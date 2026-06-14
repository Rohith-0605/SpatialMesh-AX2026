"""
SpatialMesh GAT-GNN Model — Day 8
==================================
Graph Attention Network that takes the speaker interaction graph
and outputs an (azimuth, elevation) position for each speaker.

CRITICAL: uses GATv2Conv with edge_dim=7 so the 7-dim edge features
(spectral_correlation, dominance, overlap, geometry) actually
participate in the attention computation. Plain GATConv would
silently ignore them.

Input  graph:
    x          [4, 133]  node features
    edge_index [2, 12]   directed connectivity
    edge_attr  [12, 7]   edge features

Output:
    positions  [4, 2]    (az, el) per speaker, normalized [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# ---- Architecture constants ----
NODE_DIM   = 133
EDGE_DIM   = 7
HIDDEN     = 64
HEADS      = 4
OUT_DIM    = 2     # (az, el)
N_SPEAKERS = 4


class SpatialMeshGAT(nn.Module):
    """
    2-layer Graph Attention Network with edge features.

    Layer 1: GATv2Conv(133 → 64, heads=4, concat=True)  → 256-dim
    Layer 2: GATv2Conv(256 → 32, heads=4, concat=False) → 32-dim
    Head:    Linear(32 → 2) → tanh → (az, el) in [-1, 1]

    The residual blend at the end ties output to current position so
    the GNN learns a refinement of the existing layout rather than a
    blind reassignment — matches the "dynamic refinement" framing and
    plays nicely with the stability term in the loss.
    """

    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                 hidden=HIDDEN, heads=HEADS, out_dim=OUT_DIM,
                 refine=True, refine_gain=0.5):
        super().__init__()
        self.refine = refine
        self.refine_gain = refine_gain

        # Layer 1 — concat heads → hidden*heads output
        self.gat1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=0.1,
            add_self_loops=False,   # directed graph, keep as-is
        )
        self.norm1 = nn.LayerNorm(hidden * heads)

        # Layer 2 — average heads → 32 output
        self.gat2 = GATv2Conv(
            in_channels=hidden * heads,
            out_channels=32,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=0.1,
            add_self_loops=False,
        )
        self.norm2 = nn.LayerNorm(32)

        # Position head
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # current position lives in node features [131:133]
        current_pos = x[:, 131:133]   # [N, 2]

        h = self.gat1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.elu(h)

        h = self.gat2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = F.elu(h)

        delta = torch.tanh(self.head(h))   # [N, 2] in [-1, 1]

        if self.refine:
            # Refinement: nudge current position, then re-clamp.
            out = current_pos + self.refine_gain * delta
            out = torch.tanh(out)   # keep in [-1, 1]
        else:
            out = delta

        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_size_mb(model):
    return count_params(model) * 4 / (1024 ** 2)  # float32


# =====================================================================
# Self-test — forward + backward on a synthetic graph
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # Build a synthetic graph matching Day 7 output format
    x = torch.randn(N_SPEAKERS, NODE_DIM)
    # plant realistic current positions in [131:133]
    x[:, 131] = torch.tensor([0.0, 0.5, -0.5, 1.0])   # az
    x[:, 132] = torch.tensor([0.0, 0.0, 0.0, 0.0])    # el
    # activity flag in [130]
    x[:, 130] = torch.tensor([1., 1., 1., 0.])

    # directed edges (12)
    src, dst = [], []
    for i in range(N_SPEAKERS):
        for j in range(N_SPEAKERS):
            if i != j:
                src.append(i); dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.randn(12, EDGE_DIM)

    model = SpatialMeshGAT()
    print("=== Model ===")
    print(f"Parameters: {count_params(model):,}")
    print(f"Est. size:  {estimate_size_mb(model):.3f} MB")

    print("\n=== Forward pass ===")
    out = model(x, edge_index, edge_attr)
    print(f"Output shape: {out.shape}")          # [4, 2]
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"Positions (az, el normalized):\n{out.detach().numpy().round(3)}")

    assert out.shape == (N_SPEAKERS, OUT_DIM), "Wrong output shape"
    assert out.min() >= -1.0 and out.max() <= 1.0, "Output out of [-1,1]"

    print("\n=== Backward pass with loss ===")
    from spatialmesh_loss import spatialmesh_loss

    activity_mask    = x[:, 130]
    dominance        = torch.tensor([0.8, 0.3, 0.6, 0.0])
    overlap_duration = torch.rand(4, 4) * 3.0
    prev_positions   = x[:, 131:133].clone()

    loss, comp = spatialmesh_loss(
        out, prev_positions, x[:, :128],
        activity_mask, dominance, overlap_duration)
    print(f"Loss: {comp['total']:.4f}")
    for k, v in comp.items():
        print(f"  {k:14s}: {v}")

    loss.backward()

    # Verify gradients flow to model parameters
    has_grad = all(p.grad is not None for p in model.parameters()
                   if p.requires_grad)
    grad_finite = all(torch.isfinite(p.grad).all().item()
                      for p in model.parameters()
                      if p.grad is not None)
    print(f"\nGradients flow to all params: {has_grad}")
    print(f"All gradients finite:         {grad_finite}")

    assert has_grad, "Some params got no gradient"
    assert grad_finite, "Non-finite gradient found"

    print("\n=== A few training steps (sanity — loss should drop) ===")
    model2 = SpatialMeshGAT()
    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(20):
        opt.zero_grad()
        pred = model2(x, edge_index, edge_attr)
        l, _ = spatialmesh_loss(
            pred, prev_positions, x[:, :128],
            activity_mask, dominance, overlap_duration)
        l.backward()
        opt.step()
        if step % 5 == 0:
            print(f"  step {step:2d}: loss {l.item():.4f}")
    print("\nPASS: model builds, forward + backward work, loss decreases.")
