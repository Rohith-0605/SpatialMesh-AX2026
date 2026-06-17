"""
SpatialMesh - Day 1 demo scaffold
Globe (Plotly) + Auto-assign (GNN) + binaural playback.

Runs out-of-the-box with a MOCK GNN and a PLACEHOLDER HRTF render so you can
see/hear the full loop today. Swap in your real model + SONICOM render at the
two marked spots:  [PLUG 1]  and  [PLUG 2].

Local:  pip install gradio plotly numpy && python app.py
Colab:  same, but demo.launch(share=True)
"""

import numpy as np
import plotly.graph_objects as go
import gradio as gr

FS = 44100
DUR = 3.0
N_SPK = 4
SPK_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#185FA5"]
SPK_NAMES = ["Speaker A", "Speaker B", "Speaker C", "Speaker D"]


# ---------------------------------------------------------------------------
# [PLUG 1]  REAL GNN GOES HERE
# Replace the body with: load SpatialMeshGAT, build graph from features,
# run forward, denormalize tanh output [-1,1] -> degrees.
# Must return a list of (azimuth_deg, elevation_deg), one per active speaker.
# ---------------------------------------------------------------------------
def assign_positions_gnn(n_active=4):
    # MOCK: spreads speakers with a bit of elevation variation so the globe
    # looks like a smart assignment. DELETE once gat_best.pt is wired in.
    rng = np.random.default_rng()
    base = np.linspace(-120, 120, n_active)
    az = base + rng.uniform(-8, 8, n_active)
    el = rng.uniform(-15, 25, n_active)
    return list(zip(az.tolist(), el.tolist()))


# ---------------------------------------------------------------------------
# [PLUG 2]  REAL SONICOM HRTF RENDER GOES HERE
# Replace with your nearest-angle HRIR lookup + fftconvolve per ear, summed.
# Inputs: list of mono np arrays + their (az,el). Output: stereo (N,2) float.
# ---------------------------------------------------------------------------
def render_binaural(streams, positions):
    # PLACEHOLDER: constant-power pan (ILD) + small ITD per source. Audibly
    # spatial, but NOT your HRTF. Swap for SONICOM convolution.
    out = np.zeros((int(FS * DUR), 2), dtype=np.float32)
    itd_max = int(0.0007 * FS)  # ~0.7 ms
    for sig, (az, el) in zip(streams, positions):
        pan = np.sin(np.radians(az))           # -1 left .. +1 right
        gl = np.sqrt(0.5 * (1 - pan))
        gr_ = np.sqrt(0.5 * (1 + pan))
        d = int(itd_max * pan)                 # >0: right leads, delay left
        L = sig.copy(); R = sig.copy()
        if d > 0:
            L = np.concatenate([np.zeros(d, np.float32), L])[: len(sig)]
        elif d < 0:
            R = np.concatenate([np.zeros(-d, np.float32), R])[: len(sig)]
        n = min(len(sig), len(out))
        out[:n, 0] += gl * L[:n]
        out[:n, 1] += gr_ * R[:n]
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.9).astype(np.float32)


# --- stand-in audio: 4 distinct tones (replace with LibriSpeech mono) -------
def dummy_streams(n):
    t = np.linspace(0, DUR, int(FS * DUR), endpoint=False)
    freqs = [180, 240, 320, 400]
    return [(0.4 * np.sin(2 * np.pi * freqs[i] * t)).astype(np.float32)
            for i in range(n)]


# --- globe -----------------------------------------------------------------
def sph_to_xyz(az, el, r=1.0):
    a, e = np.radians(az), np.radians(el)
    return r * np.cos(e) * np.sin(a), r * np.cos(e) * np.cos(a), r * np.sin(e)


def _wireframe_lines(r=1.0, n_lat=9, n_lon=12, pts_per_line=60):
    """Lat/long wireframe as Scatter3d line segments (one trace, NaN-separated)."""
    xs, ys, zs = [], [], []
    # latitude rings (el = const, az sweeps 0..360)
    for el_deg in np.linspace(-80, 80, n_lat):
        e = np.radians(el_deg)
        a = np.linspace(0, 2 * np.pi, pts_per_line)
        xs += list(r * np.cos(e) * np.sin(a)) + [None]
        ys += list(r * np.cos(e) * np.cos(a)) + [None]
        zs += list(r * np.sin(e) * np.ones_like(a)) + [None]
    # longitude meridians (az = const, el sweeps -90..90)
    for az_deg in np.linspace(0, 360, n_lon, endpoint=False):
        a = np.radians(az_deg)
        e = np.linspace(-np.pi / 2, np.pi / 2, pts_per_line)
        xs += list(r * np.cos(e) * np.sin(a)) + [None]
        ys += list(r * np.cos(e) * np.cos(a)) + [None]
        zs += list(r * np.sin(e)) + [None]
    return xs, ys, zs


def _degree_dot_lattice(r=1.0, step=15):
    """Dot at every `step` degrees of (az, el) -- a visible measurement grid,
    evocative of the discrete SONICOM HRTF sampling positions."""
    az = np.arange(-180, 180, step)
    el = np.arange(-80, 81, step)
    AZ, EL = np.meshgrid(az, el)
    x, y, z = sph_to_xyz(AZ.ravel(), EL.ravel(), r)
    return x, y, z


def make_globe(positions, radii=None):
    if radii is None:
        radii = [1.0] * len(positions)
    fig = go.Figure()

    # faint solid sphere underneath, for depth/shading cues
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    fig.add_surface(
        x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v),
        opacity=0.05, showscale=False,
        colorscale=[[0, "#9aa5b1"], [1, "#9aa5b1"]], hoverinfo="skip")

    # lat/long wireframe
    wx, wy, wz = _wireframe_lines()
    fig.add_scatter3d(x=wx, y=wy, z=wz, mode="lines",
                      line=dict(color="#b9c0c8", width=1),
                      hoverinfo="skip", showlegend=False)

    # per-degree measurement dot lattice (HRTF sampling grid feel)
    dx, dy, dz = _degree_dot_lattice(step=15)
    fig.add_scatter3d(x=dx, y=dy, z=dz, mode="markers",
                      marker=dict(size=1.6, color="#7c8794", opacity=0.55),
                      hoverinfo="skip", showlegend=False)

    # listener at center
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                      marker=dict(size=7, color="#111", symbol="diamond"),
                      text=["You"], textposition="top center",
                      hoverinfo="skip", showlegend=False)
    # small "nose" tick toward +y (az=0, el=0) so front is unambiguous
    fig.add_scatter3d(x=[0, 0], y=[0, 1.15], z=[0, 0], mode="lines+text",
                      line=dict(color="#111", width=3),
                      text=["", "front (0°)"], textposition="top center",
                      hoverinfo="skip", showlegend=False)

    for i, ((az, el), r) in enumerate(zip(positions, radii)):
        x, y, z = sph_to_xyz(az, el, r)
        fig.add_scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines",
                          line=dict(color=SPK_COLORS[i], width=5),
                          hoverinfo="skip", showlegend=False)
        fig.add_scatter3d(
            x=[x], y=[y], z=[z], mode="markers+text",
            marker=dict(size=11, color=SPK_COLORS[i],
                       line=dict(color="white", width=1)),
            text=[f"{SPK_NAMES[i]}<br>{az:.0f}°,{el:.0f}°"],
            textposition="top center", name=SPK_NAMES[i], showlegend=False)

    # fixed, known camera: looking along -y toward origin, slightly elevated.
    # With this camera: +x -> screen-right, +z -> screen-up, az=0 -> "into" the screen (front).
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=-2.2, z=0.9),
                       up=dict(x=0, y=0, z=1))),
        margin=dict(l=0, r=0, t=0, b=0), height=520, showlegend=False)
    return fig


# --- callback --------------------------------------------------------------
def run_auto():
    pos = assign_positions_gnn(N_SPK)
    streams = dummy_streams(N_SPK)
    stereo = render_binaural(streams, pos)
    audio = (FS, (stereo * 32767).astype(np.int16))
    label = "  |  ".join(f"{SPK_NAMES[i]}: {a:.0f}°/{e:.0f}°"
                         for i, (a, e) in enumerate(pos))
    return make_globe(pos), audio, label


with gr.Blocks(title="SpatialMesh") as demo:
    gr.Markdown("## SpatialMesh — 3D conversation soundscape\n"
                "Auto mode: the GNN places each speaker for maximum "
                "separability. Hit **Auto-assign**, then play the binaural mix.")
    with gr.Row():
        globe = gr.Plot(make_globe(assign_positions_gnn(N_SPK)))
    with gr.Row():
        btn = gr.Button("Auto-assign (GNN)", variant="primary")
    audio = gr.Audio(label="Binaural output (use headphones)", type="numpy")
    info = gr.Markdown()
    btn.click(run_auto, outputs=[globe, audio, info])

if __name__ == "__main__":
    demo.launch()