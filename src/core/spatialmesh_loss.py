"""
SpatialMesh Loss Function — v2 (Jun 14)
=========================================
Fixes over v1:
  1. acos removed → smooth angular_proximity (gradient-stable everywhere)
  2. elevation_spread term added → enforces true 3D use, not flat layouts
  3. perceptual_separability_score() → bridges loss to MOS for validation

Locked weights (tune on Day 10, report final values):
  interference : 1.00   primary — minimize perceptual masking
  repulsion    : 0.05   light prior — prevent global clustering
  elevation    : 0.08   NEW — reward vertical spread when crowded
  comfort      : 0.05   reduced from 0.10 — stop fighting separation
  stability    : 0.10   smooth frame-to-frame transitions
"""

import torch
import torch.nn.functional as F
from itertools import permutations

# ---- Constants (locked) ----
N_SPEAKERS      = 4
MIN_GLOBAL_SEP  = 30.0    # degrees — minimum comfortable separation
TAU             = 45.0    # repulsion decay constant (degrees)
ALPHA_SIGMOID   = 5.0     # dominance asymmetry sharpness
AZ_COMFORT      = 110.0   # degrees — comfortable azimuth half-range
                          # widened from 60: 4 speakers cannot separate
                          # within +/-60. 110 still excludes the rear
                          # cone (|az|>110) which is disorienting.
EL_COMFORT      = 45.0    # degrees — comfortable elevation half-range
OVERLAP_WINDOW  = 5.0     # seconds — overlap normalization
EPS             = 1e-8

# ---- Loss weights ----
W_INTERFERENCE  = 1.00
W_REPULSION     = 0.05
W_ELEVATION     = 0.08
W_COMFORT       = 0.05
W_STABILITY     = 0.10


# =====================================================================
# FIX 1 — Smooth angular proximity (replaces acos great-circle distance)
# =====================================================================
def angular_proximity(pos_i, pos_j):
    """
    Smooth proximity between two positions on the sphere.

    Returns:
        proximity in [0, 1]
            1.0  → same direction (maximum crowding / collision)
            0.5  → 90 degrees apart
            0.0  → diametrically opposite (maximum separation)

    Why this replaces acos:
        acos(cos_sim) has infinite gradient at cos_sim = ±1, exactly
        where speakers collide or oppose — the cases the GNN visits most
        during early training. This formulation uses only sin/cos/linear
        ops, so gradients stay finite and smooth across the entire sphere.

    pos format: [az, el] normalized to [-1, 1]
        az: [-1,1] → [-180, 180] deg → [-pi, pi] rad
        el: [-1,1] → [-90, 90] deg  → [-pi/2, pi/2] rad
    """
    az_i = pos_i[0] * torch.pi
    el_i = pos_i[1] * (torch.pi / 2)
    az_j = pos_j[0] * torch.pi
    el_j = pos_j[1] * (torch.pi / 2)

    # Cosine of the angle between the two direction vectors on a sphere
    cos_angle = (torch.sin(el_i) * torch.sin(el_j) +
                 torch.cos(el_i) * torch.cos(el_j) *
                 torch.cos(az_i - az_j))

    # Remap [-1, 1] → [0, 1]; no acos, fully differentiable
    proximity = (cos_angle + 1.0) / 2.0
    return proximity


def proximity_to_degrees(proximity):
    """
    Convert smooth proximity back to approximate angular separation
    in degrees — for logging / interpretability only, NOT used in
    gradient path.
    """
    cos_angle = proximity * 2.0 - 1.0
    cos_angle = torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.acos(cos_angle))


# =====================================================================
# Main loss
# =====================================================================
def spatialmesh_loss(pred_positions,       # [4, 2] normalized [-1,1]
                     previous_positions,    # [4, 2]
                     embeddings,            # [4, 128] CNN embeddings
                     activity_mask,         # [4] binary
                     dominance,             # [4] scalar
                     overlap_duration,      # [4, 4] seconds
                     return_components=True):
    """
    Returns total_loss (scalar) and optionally a dict of components.

    All terms are differentiable and gradient-stable.
    """
    device = pred_positions.device

    interference_loss = torch.tensor(0.0, device=device)
    repulsion_loss    = torch.tensor(0.0, device=device)
    elevation_loss    = torch.tensor(0.0, device=device)
    comfort_loss      = torch.tensor(0.0, device=device)

    active_pair_count = 0

    # Normalize overlap to [0,1]
    overlap_norm = torch.clamp(overlap_duration / OVERLAP_WINDOW, 0.0, 1.0)

    # --- Pairwise terms ---
    for i in range(N_SPEAKERS):
        for j in range(N_SPEAKERS):
            if i == j:
                continue

            proximity = angular_proximity(pred_positions[i],
                                          pred_positions[j])

            # ---- Repulsion prior (all pairs) ----
            # exp(-sep/TAU). Using proximity: high proximity → high penalty.
            # Approximate angular sep from proximity for the decay shape.
            approx_sep = (1.0 - proximity) * 180.0  # 0..180 deg proxy
            repulsion_loss = repulsion_loss + torch.exp(-approx_sep / TAU)

            # ---- Interference (active pairs only) ----
            if activity_mask[i] == 0 or activity_mask[j] == 0:
                continue

            # Perceptual similarity from CNN embeddings — masking risk
            spectral_sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).clamp(0, 1).squeeze()

            # Directional dominance — sigmoid for clean asymmetry
            dom = torch.sigmoid(ALPHA_SIGMOID *
                                (dominance[i] - dominance[j]))

            # Temporal overlap
            overlap = overlap_norm[i][j]

            # Spatial crowding — directly the smooth proximity
            # high proximity = positions too close = unresolved conflict
            spatial_crowding = proximity

            # Additive interaction keeps gradients healthy even when
            # one factor is small
            conflict = (0.35 * spectral_sim +
                        0.25 * dom +
                        0.25 * overlap +
                        0.15 * spatial_crowding)

            # Scale by crowding — only penalize when positions fail
            interference_loss = interference_loss + conflict * spatial_crowding
            active_pair_count += 1

    # Normalize
    if active_pair_count > 0:
        interference_loss = interference_loss / active_pair_count
    repulsion_loss = repulsion_loss / (N_SPEAKERS * (N_SPEAKERS - 1))

    # =================================================================
    # FIX 2 — Elevation spread term
    # =================================================================
    # Problem: GNN can satisfy separation purely in azimuth and leave
    # all speakers at el=0 (flat plane). That violates the "full 3D
    # (x, y, z) spatial audio" requirement of the problem statement.
    #
    # Solution: when active speakers are azimuthally crowded, reward
    # using the elevation dimension to separate them. Only kicks in
    # when azimuth alone cannot resolve the conflict — so it does not
    # force unnatural elevation when the horizontal plane is enough.
    active_idx = [k for k in range(N_SPEAKERS) if activity_mask[k] == 1]

    if len(active_idx) >= 3:
        # Azimuth crowding signal — how packed are active speakers in az
        az_vals = torch.stack([pred_positions[k][0] for k in active_idx])
        az_spread = az_vals.std() + EPS          # low = crowded
        az_crowding = torch.exp(-az_spread * 3.0)  # high when crowded

        # Elevation usage — std of elevations among active speakers
        el_vals = torch.stack([pred_positions[k][1] for k in active_idx])
        el_spread = el_vals.std()

        # Reward elevation spread proportional to azimuth crowding.
        # If azimuth is crowded (az_crowding high) but elevation is
        # flat (el_spread low), this penalty is large → GNN learns to
        # lift speakers vertically to resolve the crowding.
        elevation_loss = az_crowding * F.relu(0.4 - el_spread)
        # 0.4 normalized ≈ 36 deg target elevation spread when crowded

    # =================================================================
    # Comfort zone (active speakers) — reduced weight
    # =================================================================
    for i in range(N_SPEAKERS):
        if activity_mask[i] == 1:
            az_deg = pred_positions[i][0] * 180.0
            el_deg = pred_positions[i][1] * 90.0
            # Normalize penalties to [0,1] scale so comfort is
            # commensurate with interference/repulsion (which are [0,1]).
            # Divide by the max possible overshoot.
            az_over = F.relu(torch.abs(az_deg) - AZ_COMFORT) / (180.0 - AZ_COMFORT)
            el_over = F.relu(torch.abs(el_deg) - EL_COMFORT) / (90.0 - EL_COMFORT)
            comfort_loss = comfort_loss + az_over + el_over
    comfort_loss = comfort_loss / max(len(active_idx), 1)

    # =================================================================
    # Stability — smooth frame-to-frame transitions
    # =================================================================
    stability_loss = F.mse_loss(pred_positions, previous_positions)

    # =================================================================
    # Total
    # =================================================================
    total_loss = (W_INTERFERENCE * interference_loss +
                  W_REPULSION    * repulsion_loss +
                  W_ELEVATION    * elevation_loss +
                  W_COMFORT      * comfort_loss +
                  W_STABILITY    * stability_loss)

    if return_components:
        return total_loss, {
            'total':        total_loss.item(),
            'interference': interference_loss.detach().item(),
            'repulsion':    repulsion_loss.detach().item(),
            'elevation':    elevation_loss.detach().item() if torch.is_tensor(elevation_loss) else float(elevation_loss),
            'comfort':      comfort_loss.detach().item() if torch.is_tensor(comfort_loss) else float(comfort_loss),
            'stability':    stability_loss.detach().item(),
            'active_pairs': active_pair_count,
        }
    return total_loss


# =====================================================================
# FIX 3 — Loss ↔ MOS correlation bridge
# =====================================================================
def perceptual_separability_score(pred_positions,
                                   embeddings,
                                   activity_mask,
                                   dominance,
                                   overlap_duration):
    """
    A 0-100 interpretable score measuring how perceptually separable
    the assigned layout is. This is the BRIDGE between the training
    loss and the MOS listening study on Day 19.

    The hypothesis to validate:
        higher separability_score  →  higher MOS rating from listeners

    On Day 19, compute this score for each listening-study clip and
    plot it against the human MOS ratings. A positive correlation
    (target Pearson r > 0.6) demonstrates the loss optimizes for what
    humans actually perceive — turning "our loss went down" into
    "our loss predicts human-rated quality."

    Score breakdown:
        100 = all active speakers perfectly separated, no masking
          0 = all active speakers collapsed to one direction
    """
    active_idx = [k for k in range(N_SPEAKERS) if activity_mask[k] == 1]
    if len(active_idx) < 2:
        return 100.0  # single speaker — trivially separable

    overlap_norm = torch.clamp(overlap_duration / OVERLAP_WINDOW, 0.0, 1.0)

    total_masking = 0.0
    pair_count = 0

    for a in range(len(active_idx)):
        for b in range(a + 1, len(active_idx)):
            i, j = active_idx[a], active_idx[b]

            proximity = angular_proximity(pred_positions[i],
                                          pred_positions[j])

            spectral_sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).clamp(0, 1).item()

            overlap = overlap_norm[i][j].item()

            # Masking for this pair: high when similar voices, overlapping,
            # AND spatially close. Symmetric for scoring (perception is
            # mutual even if dominance is directed).
            pair_masking = spectral_sim * overlap * proximity.item()
            total_masking += pair_masking
            pair_count += 1

    avg_masking = total_masking / max(pair_count, 1)  # [0, 1]
    separability = (1.0 - avg_masking) * 100.0
    return float(separability)


# =====================================================================
# Quick self-test
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # Synthetic 4-speaker scenario
    embeddings = torch.randn(4, 128)
    # Make speaker 0 and 2 similar (high masking risk)
    embeddings[2] = embeddings[0] + 0.1 * torch.randn(128)

    activity_mask    = torch.tensor([1., 1., 1., 0.])  # 3 active
    dominance        = torch.tensor([0.8, 0.3, 0.6, 0.0])
    overlap_duration = torch.tensor([
        [0., 3., 2.5, 0.],
        [3., 0., 2., 0.],
        [2.5, 2., 0., 0.],
        [0., 0., 0., 0.],
    ])
    previous_positions = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [1.0, 0.0]
    ])

    print("=== Test A: BAD layout (all crowded at front) ===")
    bad = torch.tensor([[0.0, 0.0], [0.05, 0.0],
                        [-0.05, 0.0], [1.0, 0.0]], requires_grad=True)
    loss_bad, comp_bad = spatialmesh_loss(
        bad, previous_positions, embeddings,
        activity_mask, dominance, overlap_duration)
    sep_bad = perceptual_separability_score(
        bad, embeddings, activity_mask, dominance, overlap_duration)
    for k, v in comp_bad.items():
        print(f"  {k:14s}: {v}")
    print(f"  separability  : {sep_bad:.1f}/100")

    # Gradient check — must be finite (this is what acos broke)
    loss_bad.backward()
    print(f"  grad finite   : {torch.isfinite(bad.grad).all().item()}")
    print(f"  grad max      : {bad.grad.abs().max().item():.4f}")

    print("\n=== Test B: GOOD layout (well separated + elevation) ===")
    good = torch.tensor([[0.0, 0.2], [0.55, -0.1],
                         [-0.55, 0.3], [1.0, 0.0]], requires_grad=True)
    loss_good, comp_good = spatialmesh_loss(
        good, previous_positions, embeddings,
        activity_mask, dominance, overlap_duration)
    sep_good = perceptual_separability_score(
        good, embeddings, activity_mask, dominance, overlap_duration)
    for k, v in comp_good.items():
        print(f"  {k:14s}: {v}")
    print(f"  separability  : {sep_good:.1f}/100")
    loss_good.backward()
    print(f"  grad finite   : {torch.isfinite(good.grad).all().item()}")

    print("\n=== Validation ===")
    print(f"  BAD  total loss : {comp_bad['total']:.4f}  "
          f"sep {sep_bad:.0f}")
    print(f"  GOOD total loss : {comp_good['total']:.4f}  "
          f"sep {sep_good:.0f}")
    assert comp_good['total'] < comp_bad['total'], \
        "FAIL: good layout should have lower loss"
    assert sep_good > sep_bad, \
        "FAIL: good layout should have higher separability"
    print("  PASS: good layout has lower loss AND higher separability")
    print("  → loss and separability score agree (the MOS bridge holds)")
