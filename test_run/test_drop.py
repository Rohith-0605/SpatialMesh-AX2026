"""
SpatialMesh — practical 4->3 speaker drop test
=================================================
Tests the exact loop:
  4 clips (same speaker, hard case) -> random init positions ->
  HRTF-convolve each at its own angle -> CNN embed -> GNN frame 1
  -> drop speaker D (activity=0, dominance recomputed) -> GNN frame 2
  -> compare A/B/C positions before vs after the drop
  -> render both frames to binaural WAV so you can listen to the difference

Set the three paths below, then:  python test_drop.py

Outputs (4 things):
  1. Frame-1 positions (console)
  2. Frame-2 positions, speaker D silenced (console)
  3. Delta table for A/B/C -- did they re-spread? (console)
  4. frame1_all4.wav and frame2_3active.wav written to disk
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

# ---------------------------------------------------------------------------
# PATHS -- set these to your real folders
# ---------------------------------------------------------------------------
LIBRISPEECH_SPK_DIR = r"librispeech/spk_237"          # 4 .flac clips, same speaker
SONICOM_SOFA_PATH   = r"sonicom\P0001_FreeFieldComp_48kHz.sofa"
CNN_PATH             = r"models/cnn_encoder_best.pt"
GAT_PATH             = r"models/gat_best.pt"
OUT_DIR              = r"test_out"

# ---------------------------------------------------------------------------
# Constants -- locked architecture, matches Day 5/7/8 exactly
# ---------------------------------------------------------------------------
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
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
 
 
# ---------------------------------------------------------------------------
# CNN encoder -- EXACT Day-5 architecture (matches cnn_encoder_best.pt)
# backbone: 2->64->128->256->128, separate AdaptiveAvgPool, L2-normalized output
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
 
 
def render_mix(stereo_list, activity_mask):
    """Sums only active speakers' already-HRTF-convolved stereo streams."""
    n = max(s.shape[1] for s in stereo_list)
    out = np.zeros((2, n), dtype=np.float32)
    for s, active in zip(stereo_list, activity_mask):
        if active:
            out[:, :s.shape[1]] += s
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.9).T  # [samples, 2]
 
 
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
# ---------------------------------------------------------------------------
def main():
    Path(OUT_DIR).mkdir(exist_ok=True)
 
    print("=" * 60)
    print("STEP 0 -- load models, clips, HRTF")
    print("=" * 60)
    cnn = CNNEncoder().to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn.eval()
 
    gat = SpatialMeshGAT().to(DEVICE)
    gat.load_state_dict(torch.load(GAT_PATH, map_location=DEVICE))
    gat.eval()
 
    clips = load_four_clips(LIBRISPEECH_SPK_DIR)        # 4 mono, same speaker (hard case)
    hrtf = HRTFRenderer(SONICOM_SOFA_PATH)
 
    # ---- FRAME 1: all 4 active -------------------------------------------
    print("\n" + "=" * 60)
    print("FRAME 1 -- all 4 speakers active")
    print("=" * 60)
    init_pos_norm = random_init_positions(N_SPK)
    init_pos_deg = denorm(init_pos_norm)
    print("Init positions (random, collision-rejected):")
    for label, (az, el) in zip(SPK_LABELS, init_pos_deg):
        print(f"  {label}: az={az:+.1f} deg  el={el:+.1f} deg")
 
    stereo_1 = [hrtf.convolve(clips[i], *init_pos_deg[i]) for i in range(4)]
    activity_1 = np.array([1, 1, 1, 1], dtype=np.float32)
 
    graph_1, _ = build_graph(stereo_1, activity_1, cnn, init_pos_norm)
    with torch.no_grad():
        out_1 = gat(graph_1.x, graph_1.edge_index, graph_1.edge_attr)
    pos_1_deg = denorm(out_1)
 
    print("\n>>> OUTPUT 1 -- Frame-1 GNN positions:")
    for label, (az, el) in zip(SPK_LABELS, pos_1_deg):
        print(f"  {label}: az={az:+.1f} deg  el={el:+.1f} deg")
 
    # Solo renders -- listen to ONE speaker at a time before the full mix.
    # This is what actually fixes "I can't tell directions" -- a 4-way sum
    # is hard to parse by ear no matter the duration.
    print("\nSolo files (listen one at a time, headphones, before the mix):")
    for i, label in enumerate(SPK_LABELS):
        path = render_solo(stereo_1[i], label, OUT_DIR)
        print(f"  {path}  -> should sound like it's coming from az={pos_1_deg[i][0]:+.0f} deg")
 
    mix_1 = render_mix(stereo_1, activity_1)
    sf.write(f"{OUT_DIR}/frame1_all4.wav", mix_1, SR)
 
    # ---- FRAME 2: D drops --------------------------------------------------
    print("\n" + "=" * 60)
    print("FRAME 2 -- Speaker D silenced (activity=0)")
    print("=" * 60)
    activity_2 = np.array([1, 1, 1, 0], dtype=np.float32)
    # re-convolve A/B/C at their frame-1 output positions (closes the loop);
    # D's stream is irrelevant now but kept for shape -- render_mix skips it
    stereo_2 = [hrtf.convolve(clips[i], *pos_1_deg[i]) for i in range(3)] + [stereo_1[3]]
 
    graph_2, conv_feat_2 = build_graph(stereo_2, activity_2, cnn, out_1)
    with torch.no_grad():
        out_2 = gat(graph_2.x, graph_2.edge_index, graph_2.edge_attr)
    pos_2_deg = denorm(out_2)
 
    print("Dominance after D drops (recomputed, renormalized over A/B/C):")
    print(f"  {conv_feat_2['dominance_score'].round(3)}")
 
    print("\n>>> OUTPUT 2 -- Frame-2 GNN positions (D silenced):")
    for label, (az, el) in zip(SPK_LABELS, pos_2_deg):
        tag = " (silent)" if label == "D" else ""
        print(f"  {label}: az={az:+.1f} deg  el={el:+.1f} deg{tag}")
 
    mix_2 = render_mix(stereo_2, activity_2)
    sf.write(f"{OUT_DIR}/frame2_3active.wav", mix_2, SR)
 
    # ---- OUTPUT 3: delta table ----------------------------------------------
    print("\n" + "=" * 60)
    print(">>> OUTPUT 3 -- Did A/B/C re-spread when D went silent?")
    print("=" * 60)
    print(f"{'Spk':>4} {'Frame1 az':>10} {'Frame2 az':>10} {'D-az':>8}   "
          f"{'Frame1 el':>10} {'Frame2 el':>10} {'D-el':>8}")
    for i in range(3):
        a1, e1 = pos_1_deg[i]
        a2, e2 = pos_2_deg[i]
        print(f"{SPK_LABELS[i]:>4} {a1:10.1f} {a2:10.1f} {a2-a1:8.1f}   "
              f"{e1:10.1f} {e2:10.1f} {e2-e1:8.1f}")
 
    print(f"\n>>> OUTPUT 4 -- Audio written:")
    print(f"  {OUT_DIR}/solo_A.wav ... solo_D.wav   (ONE speaker each, frame 1 -- start here)")
    print(f"  {OUT_DIR}/frame1_all4.wav    (4 speakers mixed, listen for spatial separation)")
    print(f"  {OUT_DIR}/frame2_3active.wav (A/B/C only, D dropped -- listen for the shift)")
    print("\nDone. Compare the delta table above against what you hear.")
 
 
if __name__ == "__main__":
    main()
 










