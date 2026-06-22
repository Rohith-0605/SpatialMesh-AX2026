"""
spatialmesh_core.py -- shared, verified pipeline.
Both test_drop.py and app.py import from here so they never diverge.
Contains the exact tested CNN, HRTF renderer, graph builder, and a
one-call run_pipeline() that does: convolve -> CNN -> GNN -> reconvolve -> mix.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
import sofar
from scipy.signal import fftconvolve
from torch_geometric.data import Data

from spatialmesh_gat import SpatialMeshGAT

# ---- Constants (locked architecture, matches Day 5/7/8) -------------------
SR              = 48000
SEGMENT_SEC     = 8.0
OVERLAP_WINDOW  = 5.0
ACTIVITY_THRESH = 0.01
ALPHA_SIGMOID   = 5.0
N_SPK           = 4
AZ_MAX, EL_MAX  = 110.0, 45.0
EPS             = 1e-8
SPK_LABELS      = ["A", "B", "C", "D"]
DEVICE = "cpu"

# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128, n_input_channels=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_input_channels, 64, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, embedding_dim, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(embedding_dim), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        features = self.backbone(x)              # (batch, 128, T)
        pooled = self.pool(features)              # (batch, 128, 1)
        embedding = pooled.squeeze(-1)             # (batch, 128)
        return F.normalize(embedding, p=2, dim=1)  # L2 normalize -> unit sphere


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_speaker_audio(flac_path, sr=SR, duration=SEGMENT_SEC):
    audio, orig_sr = sf.read(flac_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    n = int(duration * sr)
    audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))
    return audio.astype(np.float32)


def load_four_clips(spk_dir):
    flacs = sorted(Path(spk_dir).glob("*.flac"))
    if len(flacs) < 4:
        raise FileNotFoundError(
            f"Need 4 .flac files in {spk_dir}, found {len(flacs)}. "
            f"Place 4 different clips from ONE LibriSpeech speaker here."
        )
    chosen = flacs[:4]
    print(f"Clips loaded ({Path(spk_dir).name}):")
    for label, p in zip(SPK_LABELS, chosen):
        print(f"  {label}: {p.name}")
    return [load_speaker_audio(p) for p in chosen]


# ---------------------------------------------------------------------------
# SONICOM HRTF -- single subject, nearest-angle lookup
# ---------------------------------------------------------------------------
class HRTFRenderer:
    def __init__(self, sofa_path):
        sofa = sofar.read_sofa(sofa_path)
        self.src = sofa.SourcePosition[:, :2]      # [M,2] az, el (degrees)
        self.hrir = sofa.Data_IR                    # [M,2,N] ch0=L, ch1=R
        print(f"HRTF loaded: {self.src.shape[0]} measured positions")

    def nearest(self, az_deg, el_deg):
        a, e = np.radians(self.src[:, 0]), np.radians(self.src[:, 1])
        ta, te = np.radians(az_deg), np.radians(el_deg)
        dot = np.cos(e) * np.cos(te) * np.cos(a - ta) + np.sin(e) * np.sin(te)
        return int(np.argmax(dot))

    def convolve(self, mono, az_deg, el_deg):
        i = self.nearest(az_deg, el_deg)
        L = fftconvolve(mono, self.hrir[i, 0])
        R = fftconvolve(mono, self.hrir[i, 1])
        n = len(mono)
        return np.stack([L[:n], R[:n]], axis=0).astype(np.float32)  # [2, n]


# ---------------------------------------------------------------------------
# Graph-builder pieces -- exact Day-7 logic, adapted for variable activity
# ---------------------------------------------------------------------------
def compute_conversational_features(audios, activity_mask, sr=SR,
                                     activity_thresh=ACTIVITY_THRESH):
    """audios: list of mono [samples]. activity_mask: [N] 0/1, speakers with
    0 are excluded from dominance/overlap computation (matches sample_scene_config)."""
    N = len(audios)
    frame_len, hop_len = int(0.02 * sr), int(0.01 * sr)
    frame_activity, rms_values = [], []
    for audio, active in zip(audios, activity_mask):
        if active == 0:
            n_frames = 1 + max(0, (len(audio) - frame_len) // hop_len)
            frame_activity.append(np.zeros(max(n_frames, 1)))
            rms_values.append(0.0)
            continue
        frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))
        frame_activity.append((rms_per_frame > activity_thresh).astype(float))
        rms_values.append(np.sqrt(np.mean(audio ** 2)))

    max_len = max(len(f) for f in frame_activity)
    frame_activity = np.array([np.pad(f, (0, max_len - len(f))) for f in frame_activity])
    rms_values = np.array(rms_values)
    rms_norm = rms_values / (rms_values.max() + EPS)

    dominance_score = frame_activity.mean(axis=1)
    dominance_score[activity_mask == 0] = 0.0
    dominance_norm = dominance_score / (dominance_score.max() + EPS)

    frame_duration = hop_len / sr
    overlap_duration = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and activity_mask[i] and activity_mask[j]:
                overlap_duration[i][j] = (frame_activity[i] * frame_activity[j]).sum() * frame_duration

    return {
        "rms_energy": rms_norm,
        "activity_flag": activity_mask.astype(float),
        "dominance_score": dominance_norm,
        "overlap_duration": overlap_duration,
    }


def extract_cnn_embeddings(stereo_list, encoder, device=DEVICE):
    """stereo_list: list of [2, samples] HRTF-convolved arrays (real convolution,
    not mono-replicated -- this is the Day-12 fix applied to this test)."""
    embeddings = []
    with torch.no_grad():
        for stereo in stereo_list:
            tensor = torch.tensor(stereo).unsqueeze(0).to(device)  # [1,2,samples]
            embeddings.append(encoder(tensor).squeeze(0).cpu())
    return torch.stack(embeddings)


def build_node_features(embeddings, conv_features, current_positions):
    rms = torch.tensor(conv_features["rms_energy"], dtype=torch.float32).unsqueeze(1)
    dominance = torch.tensor(conv_features["dominance_score"], dtype=torch.float32).unsqueeze(1)
    activity = torch.tensor(conv_features["activity_flag"], dtype=torch.float32).unsqueeze(1)
    az = current_positions[:, 0].unsqueeze(1)
    el = current_positions[:, 1].unsqueeze(1)
    return torch.cat([embeddings, rms, dominance, activity, az, el], dim=1)


def build_edge_index(n=N_SPK):
    src, dst, pairs = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i); dst.append(j); pairs.append((i, j))
    return torch.tensor([src, dst], dtype=torch.long), pairs


def compute_edge_features(embeddings, conv_features, current_positions, edge_pairs):
    rms = torch.tensor(conv_features["rms_energy"], dtype=torch.float32)
    dominance = torch.tensor(conv_features["dominance_score"], dtype=torch.float32)
    activity = torch.tensor(conv_features["activity_flag"], dtype=torch.float32)
    overlap = torch.tensor(conv_features["overlap_duration"], dtype=torch.float32)
    overlap_norm = torch.clamp(overlap / OVERLAP_WINDOW, 0.0, 1.0)

    feats = []
    for (i, j) in edge_pairs:
        delta_az = (current_positions[i, 0] - current_positions[j, 0]).item()
        delta_el = (current_positions[i, 1] - current_positions[j, 1]).item()
        spec_corr = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        rel_db = float(np.clip((rms[i] - rms[j]).item(), -1.0, 1.0))
        dom_ratio = torch.sigmoid(ALPHA_SIGMOID * (dominance[i] - dominance[j])).item()
        ovlp_flag = float(activity[i].item() == 1.0 and activity[j].item() == 1.0)
        ovlp_dur = overlap_norm[i][j].item()
        feats.append([delta_az, delta_el, spec_corr, rel_db, dom_ratio, ovlp_flag, ovlp_dur])
    return torch.tensor(feats, dtype=torch.float32)


def build_graph(stereo_list, activity_mask, cnn_encoder, prev_positions):
    raw_for_features = [s.mean(axis=0) for s in stereo_list]  # mono proxy for RMS/activity timing
    conv_features = compute_conversational_features(raw_for_features, activity_mask)
    embeddings = extract_cnn_embeddings(stereo_list, cnn_encoder)
    node_features = build_node_features(embeddings, conv_features, prev_positions)
    edge_index, edge_pairs = build_edge_index(N_SPK)
    edge_attr = compute_edge_features(embeddings, conv_features, prev_positions, edge_pairs)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr), conv_features


def random_init_positions(n=N_SPK, max_tries=20):
    """Matches training scene sampler exactly -- collision-rejected random layout."""
    for _ in range(max_tries):
        az = np.random.uniform(-1, 1, n)
        el = np.random.uniform(-0.3, 0.3, n)
        ok = all(not (abs(az[i] - az[j]) < 0.1 and abs(el[i] - el[j]) < 0.1)
                 for i in range(n) for j in range(i + 1, n))
        if ok:
            return torch.tensor(np.stack([az, el], axis=1), dtype=torch.float32)
    return torch.tensor([[-0.7, 0.0], [-0.2, 0.0], [0.3, 0.0], [0.8, 0.0]], dtype=torch.float32)


def denorm(pos_tensor):
    return [(p[0].item() * AZ_MAX, p[1].item() * EL_MAX) for p in pos_tensor]


def render_mix(stereo_list, activity_mask, gains=None):
    if gains is None:
        gains = [1.0] * len(stereo_list)
    n = max(s.shape[1] for s in stereo_list)
    gap = np.zeros((2, int(0.3 * SR)), dtype=np.float32)  # 0.3s silence between speakers
    
    # First: play each speaker solo briefly so listener can localize
    intro_parts = []
    for i, (s, active, g) in enumerate(zip(stereo_list, activity_mask, gains)):
        if active:
            chunk = s * float(g)
            intro_parts.append(chunk[:, :int(1.5 * SR)])  # 1.5s per speaker
            intro_parts.append(gap)
    
    # Then: full mix
    full = np.zeros((2, n), dtype=np.float32)
    for s, active, g in zip(stereo_list, activity_mask, gains):
        if active:
            full[:, :s.shape[1]] += s * float(g)

    # Concatenate intro + full mix
    out = np.concatenate(intro_parts + [gap, full], axis=1)
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.9).T

def render_solo(stereo, label, out_dir):
    """Write one speaker alone, normalized -- for confirming direction by ear
    before judging the cluttered full mix."""
    peak = np.max(np.abs(stereo)) or 1.0
    sig = (stereo / peak * 0.9).T
    path = f"{out_dir}/solo_{label}.wav"
    sf.write(path, sig, SR)
    return path


# ---------------------------------------------------------------------------
# Main test


# ===========================================================================
# HIGH-LEVEL HELPERS for the app -- wrap the verified functions above
# ===========================================================================
_cnn = None
_gat = None
_hrtf = None


def load_models(cnn_path, gat_path, sofa_path):
    """Load CNN + GAT + HRTF once. Call at app startup."""
    global _cnn, _gat, _hrtf
    _cnn = CNNEncoder().to(DEVICE)
    _cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    _cnn.eval()
    _gat = SpatialMeshGAT().to(DEVICE)
    _gat.load_state_dict(torch.load(gat_path, map_location=DEVICE))
    _gat.eval()
    _hrtf = HRTFRenderer(sofa_path)
    return _cnn, _gat, _hrtf

def _enforce_separation(new_deg, activity_mask, min_az_sep=30.0):
    """Spread ONLY active speakers so no two cluster. Preserves the GNN's
    left-to-right ordering (its separability decision) but guarantees a clean,
    evenly-fanned layout. Muted speakers are left untouched (not rendered)."""
    new_deg = list(new_deg)
    active = [i for i in range(len(new_deg)) if activity_mask[i] == 1]
    if len(active) <= 1:
        return new_deg  # 0 or 1 active speaker -> nothing to separate

    # order active speakers left -> right by the GNN's azimuth
    active.sort(key=lambda i: new_deg[i][0])
    n = len(active)

    # evenly spaced azimuth fan across a comfortable arc
    spread = min(AZ_MAX, max(45.0, (n - 1) * min_az_sep / 2))
    targets_az = np.linspace(-spread, spread, n)
    el_variety = [15.0, -10.0, 12.0, -15.0]  # vertical spread for realism

    for k, i in enumerate(active):
        az = float(targets_az[k])
        el = float(np.clip(el_variety[k], -EL_MAX, EL_MAX))
        new_deg[i] = (az, el)
    return new_deg


def run_pipeline(clips, activity_mask, prev_positions_norm, gains=None):
    """One full cycle. Returns (positions_deg [N,2], out_norm, stereo_mix [samples,2])."""
# NEW — always start from a wide spread so GNN doesn't collapse
    import torch
    forced_init = torch.tensor([
        [-0.8,  0.1],
        [-0.3, -0.1],
        [ 0.3,  0.1],
        [ 0.8, -0.1],
    ], dtype=torch.float32)
# only use forced init if current positions are clustered
    pos_spread = float(prev_positions_norm[:, 0].max() - prev_positions_norm[:, 0].min())
    if pos_spread < 0.5:  # clustered — reset to wide spread
        print(f"[spread reset] pos_spread={pos_spread:.2f} < 0.5, resetting to wide init")
        prev_positions_norm = forced_init
    prev_deg = denorm(prev_positions_norm)
    # 1) convolve each clip at its current position
    stereo_in = [_hrtf.convolve(clips[i], *prev_deg[i]) for i in range(len(clips))]
    # 2+3) build graph (CNN inside) -> GAT -> new normalized positions
    graph, _ = build_graph(stereo_in, activity_mask, _cnn, prev_positions_norm)

    
    with torch.no_grad():
        out_norm = _gat(graph.x, graph.edge_index, graph.edge_attr)

     
    new_deg = denorm(out_norm)

    # --- separation post-filter: spread ACTIVE speakers only ---
    # new_deg = _enforce_separation(new_deg, activity_mask, min_az_sep=30.0)

    # --- debug ---

    # 4) re-convolve at corrected positions, mix only active speakers
    stereo_out = [_hrtf.convolve(clips[i], *new_deg[i]) for i in range(len(clips))]
    mix = render_mix(stereo_out, activity_mask, gains)
    return new_deg, out_norm, mix