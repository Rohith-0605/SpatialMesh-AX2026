"""
SpatialMesh -- Gradio demo with Spatial Context Agent.
Agent runs in background every 3s, monitors:
  1. Speaker separation too tight  -> trigger_gnn_reassign()
  2. Activity changed (mute/unmute) -> trigger_gnn_reassign()
  3. Noise spike                   -> boost_ild_separation()
  4. Head yaw changed >15deg       -> update_world_lock(yaw)
  5. One speaker dominates         -> set_speaker_priority(idx)
"""

import threading, time, math
import numpy as np
import plotly.graph_objects as go
import gradio as gr

import spatialmesh_core as core

# ---- paths -----------------------------------------------------------------
CNN_PATH  = "models/cnn_encoder_best.pt"
GAT_PATH  = "models/gat_best.pt"
SOFA_PATH = "sonicom/P0001_FreeFieldComp_48kHz.sofa"
CLIPS_DIR = "librispeech/spk_237"

SPK_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#185FA5"]
SPK_NAMES  = ["A", "B", "C", "D"]

# ---- load once -------------------------------------------------------------
core.load_models(CNN_PATH, GAT_PATH, SOFA_PATH)
CLIPS = core.load_four_clips(CLIPS_DIR)

# ---- shared state ----------------------------------------------------------
state = {
    "pos_norm"    : core.random_init_positions(core.N_SPK),
    "pos_deg"     : None,          # filled after first run
    "activity"    : np.ones(4, dtype=np.float32),
    "prev_activity": np.ones(4, dtype=np.float32),
    "mix"         : None,
    "head_yaw"    : 0.0,           # degrees, set by slider
    "prev_yaw"    : 0.0,
    "noise_level" : 0.0,           # 0-1, set by slider
    "agent_log"   : [],            # list of strings shown in UI
    "lock"        : threading.Lock(),
    "dirty"       : False,         # True when agent updated state
}


# ===========================================================================
# AGENT TOOL CALLS
# ===========================================================================
def trigger_gnn_reassign(reason):
    """Re-run full pipeline with current clips + activity."""
    with state["lock"]:
        act = state["activity"].copy()
        prev = state["pos_norm"]
    deg, norm, mix = core.run_pipeline(CLIPS, act, prev)
    with state["lock"]:
        state["pos_norm"] = norm
        state["pos_deg"]  = deg
        state["mix"]      = mix
        state["dirty"]    = True
    return f"↺ reassign ({reason})"


def update_world_lock(yaw_deg):
    """Rotate all speaker positions by -yaw so soundfield stays world-locked."""
    with state["lock"]:
        norm = state["pos_norm"].clone()
    deg = core.denorm(norm)
    # subtract head yaw from azimuth of every speaker
    for i in range(len(deg)):
        deg[i] = (deg[i][0] - yaw_deg, deg[i][1])
    # re-normalize back into [-1,1]
    import torch
    new_norm = torch.tensor(
        [[az / core.AZ_MAX, el / core.EL_MAX] for az, el in deg],
        dtype=torch.float32)
    new_norm = new_norm.clamp(-1, 1)
    with state["lock"]:
        state["pos_norm"] = new_norm
        state["pos_deg"]  = deg
        state["dirty"]    = True
    return f"🔒 world-lock yaw={yaw_deg:.0f}°"


def boost_ild_separation():
    """Widen azimuth spread by 20% to fight noise masking."""
    with state["lock"]:
        norm = state["pos_norm"].clone()
    deg = core.denorm(norm)
    for i in range(len(deg)):
        deg[i] = (deg[i][0] * 1.2, deg[i][1])
    import torch
    new_norm = torch.tensor(
        [[az / core.AZ_MAX, el / core.EL_MAX] for az, el in deg],
        dtype=torch.float32)
    new_norm = new_norm.clamp(-1, 1)
    with state["lock"]:
        state["pos_norm"] = new_norm
        state["pos_deg"]  = deg
        state["dirty"]    = True
    return "📢 ILD boost (noise detected)"


def set_speaker_priority(idx):
    """Move dominant speaker toward front-center (az~0, el~0)."""
    with state["lock"]:
        norm = state["pos_norm"].clone()
    deg = list(core.denorm(norm))
    az, el = deg[idx]
    deg[idx] = (az * 0.4, el * 0.5)   # pull toward center
    import torch
    new_norm = torch.tensor(
        [[a / core.AZ_MAX, e / core.EL_MAX] for a, e in deg],
        dtype=torch.float32)
    new_norm = new_norm.clamp(-1, 1)
    with state["lock"]:
        state["pos_norm"] = new_norm
        state["pos_deg"]  = deg
        state["dirty"]    = True
    return f"⭐ priority → Spk {SPK_NAMES[idx]}"


# ===========================================================================
# AGENT LOOP  (monitor → reason → act)
# ===========================================================================
SEP_THRESHOLD  = 25.0   # deg — if min pairwise azimuth gap < this, re-spread
NOISE_THRESH   = 0.6    # noise slider value above which ILD boost fires
YAW_THRESH     = 15.0   # deg head rotation before world-lock fires
AGENT_INTERVAL = 3.0    # seconds between checks


def min_az_separation(deg):
    """Minimum pairwise azimuth gap between ACTIVE speakers."""
    active = [i for i in range(4) if state["activity"][i] == 1]
    if len(active) < 2:
        return 999.0
    gaps = []
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            gaps.append(abs(deg[active[i]][0] - deg[active[j]][0]))
    return min(gaps)


def agent_step():
    actions = []

    with state["lock"]:
        act      = state["activity"].copy()
        prev_act = state["prev_activity"].copy()
        deg      = state["pos_deg"]
        yaw      = state["head_yaw"]
        prev_yaw = state["prev_yaw"]
        noise    = state["noise_level"]

    # --- REASON ---------------------------------------------------------
    activity_changed = not np.array_equal(act, prev_act)
    needs_reassign   = deg is not None and min_az_separation(deg) < SEP_THRESHOLD
    noise_spike      = noise > NOISE_THRESH
    yaw_shifted      = abs(yaw - prev_yaw) > YAW_THRESH

    # dominant speaker: most energy (simplified: first active)
    active_idx = [i for i in range(4) if act[i] == 1]

    # --- ACT (priority order) -------------------------------------------
    if activity_changed or needs_reassign:
        reason = "mute change" if activity_changed else "tight cluster"
        actions.append(trigger_gnn_reassign(reason))

    if noise_spike:
        actions.append(boost_ild_separation())

    if yaw_shifted:
        actions.append(update_world_lock(yaw - prev_yaw))

    if not actions and active_idx:
        # quietly check if dominant speaker is too central
        if deg is not None:
            dom = active_idx[0]
            if abs(deg[dom][0]) < 10 and len(active_idx) > 1:
                actions.append(set_speaker_priority(dom))

    # update prev state
    with state["lock"]:
        state["prev_activity"] = act.copy()
        state["prev_yaw"]      = yaw
        if actions:
            log = state["agent_log"]
            ts  = time.strftime("%H:%M:%S")
            for a in actions:
                log.append(f"[{ts}] {a}")
            state["agent_log"] = log[-6:]   # keep last 6 entries

    return actions


def agent_thread():
    while True:
        time.sleep(AGENT_INTERVAL)
        try:
            agent_step()
        except Exception as e:
            with state["lock"]:
                state["agent_log"].append(f"[agent err] {e}")


# start agent in background
threading.Thread(target=agent_thread, daemon=True).start()


# ===========================================================================
# GLOBE + UI HELPERS
# ===========================================================================
def sph_to_xyz(az, el, r=1.0):
    a, e = np.radians(az), np.radians(el)
    return r*np.cos(e)*np.sin(a), r*np.cos(e)*np.cos(a), r*np.sin(e)


def make_globe(positions_deg, activity):
    fig = go.Figure()
    xs, ys, zs = [], [], []
    for el in np.linspace(-80, 80, 9):
        a = np.linspace(0, 2*np.pi, 60)
        xs += list(np.cos(np.radians(el))*np.sin(a))+[None]
        ys += list(np.cos(np.radians(el))*np.cos(a))+[None]
        zs += list(np.sin(np.radians(el))*np.ones_like(a))+[None]
    for az in np.linspace(0, 360, 12, endpoint=False):
        e = np.linspace(-np.pi/2, np.pi/2, 60)
        xs += list(np.cos(e)*np.sin(np.radians(az)))+[None]
        ys += list(np.cos(e)*np.cos(np.radians(az)))+[None]
        zs += list(np.sin(e))+[None]
    fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines",
                      line=dict(color="#b9c0c8", width=1),
                      hoverinfo="skip", showlegend=False)
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                      marker=dict(size=6, color="#111"), text=["You"],
                      textposition="top center", hoverinfo="skip", showlegend=False)
    fig.add_scatter3d(x=[0,0], y=[0,1.15], z=[0,0], mode="lines+text",
                      line=dict(color="#111", width=3),
                      text=["","front 0°"], textposition="top center",
                      hoverinfo="skip", showlegend=False)
    for i, (az, el) in enumerate(positions_deg):
        on = activity[i] == 1
        x, y, z = sph_to_xyz(az, el)
        col = SPK_COLORS[i] if on else "#cccccc"
        fig.add_scatter3d(x=[0,x], y=[0,y], z=[0,z], mode="lines",
                          line=dict(color=col, width=5 if on else 2),
                          hoverinfo="skip", showlegend=False)
        tag = SPK_NAMES[i] + ("" if on else " ✗")
        fig.add_scatter3d(x=[x], y=[y], z=[z], mode="markers+text",
                          marker=dict(size=11 if on else 7, color=col),
                          text=[f"{tag}<br>{az:.0f}°,{el:.0f}°"],
                          textposition="top center", showlegend=False)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="cube",
                   camera=dict(eye=dict(x=1.6, y=-2.2, z=0.9))),
        margin=dict(l=0, r=0, t=0, b=0), height=520, showlegend=False)
    return fig


def _current_deg():
    with state["lock"]:
        d = state["pos_deg"]
    if d is None:
        d = core.denorm(state["pos_norm"])
    return d


# ===========================================================================
# UI CALLBACKS
# ===========================================================================
def on_mute(m_a, m_b, m_c, m_d):
    """User toggled a mute checkbox — update activity; agent will auto-reassign."""
    act = np.array([0.0 if m else 1.0 for m in (m_a,m_b,m_c,m_d)],
                   dtype=np.float32)
    with state["lock"]:
        state["activity"] = act
    # return immediately — agent picks it up in ≤3s
    deg = _current_deg()
    label = "Agent will auto-reassign... " + "  |  ".join(
        f"{SPK_NAMES[i]}: {deg[i][0]:.0f}°" +
        ("" if act[i] else " ✗") for i in range(4))
    return make_globe(deg, act), label


def on_reassign(m_a, m_b, m_c, m_d):
    """Manual Re-assign button — immediate pipeline run."""
    act = np.array([0.0 if m else 1.0 for m in (m_a,m_b,m_c,m_d)],
                   dtype=np.float32)
    with state["lock"]:
        state["activity"] = act
    deg, norm, mix = core.run_pipeline(CLIPS, act, state["pos_norm"])
    with state["lock"]:
        state["pos_norm"] = norm
        state["pos_deg"]  = deg
        state["mix"]      = mix
    audio = (core.SR, (mix * 32767).astype(np.int16))
    label = "  |  ".join(
        f"{SPK_NAMES[i]}: {deg[i][0]:.0f}°,{deg[i][1]:.0f}°" +
        ("" if act[i] else " ✗") for i in range(4))
    return make_globe(deg, act), audio, label


def on_yaw(yaw):
    with state["lock"]:
        state["head_yaw"] = float(yaw)
    deg = _current_deg()
    act = state["activity"]
    return make_globe(deg, act)


def on_noise(noise):
    with state["lock"]:
        state["noise_level"] = float(noise)
    return f"Noise level: {noise:.2f}"


def poll_agent():
    """Called by Gradio timer — returns agent log."""
    with state["lock"]:
        log  = state["agent_log"].copy()
        dirty = state["dirty"]
        deg  = state["pos_deg"]
        act  = state["activity"].copy()
        mix  = state["mix"]
        state["dirty"] = False
    log_str = "\n".join(log) if log else "Agent monitoring..."
    if dirty and deg is not None and mix is not None:
        audio = (core.SR, (mix * 32767).astype(np.int16))
        return make_globe(deg, act), audio, log_str
    deg = _current_deg()
    return make_globe(deg, act), gr.update(), log_str


# ===========================================================================
# UI LAYOUT
# ===========================================================================
init_deg = core.denorm(state["pos_norm"])
init_act = np.ones(4, dtype=np.float32)

with gr.Blocks(title="SpatialMesh", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎧 SpatialMesh — Immersive Spatial Voice Call\n"
                "**Spatial Context Agent** runs every 3 s — mute a speaker "
                "and watch the GNN auto-reassign. Headphones on.")

    with gr.Row():
        with gr.Column(scale=3):
            globe = gr.Plot(make_globe(init_deg, init_act), label="3D Soundscape")
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            m_a = gr.Checkbox(label="Mute A", value=False)
            m_b = gr.Checkbox(label="Mute B", value=False)
            m_c = gr.Checkbox(label="Mute C", value=False)
            m_d = gr.Checkbox(label="Mute D", value=False)
            btn = gr.Button("Re-assign now (GNN)", variant="primary")
            gr.Markdown("---")
            yaw_sl  = gr.Slider(-90, 90, value=0, step=1,
                                label="Head yaw (°) — world-lock")
            noise_sl = gr.Slider(0, 1, value=0, step=0.05,
                                 label="Noise level — ILD boost")
            noise_info = gr.Markdown("Noise level: 0.00")

    audio  = gr.Audio(label="Binaural output (headphones)", type="numpy")
    info   = gr.Markdown("Press Re-assign or mute a speaker to start.")
    agent_box = gr.Textbox(label="🤖 Spatial Context Agent log",
                           lines=6, interactive=False,
                           value="Agent monitoring...")

    # mute checkboxes → update activity (agent auto-fires)
    for cb in (m_a, m_b, m_c, m_d):
        cb.change(on_mute, inputs=[m_a,m_b,m_c,m_d],
                  outputs=[globe, info])

    # manual button → immediate
    btn.click(on_reassign, inputs=[m_a,m_b,m_c,m_d],
              outputs=[globe, audio, info])

    # sliders
    yaw_sl.change(on_yaw,   inputs=[yaw_sl],   outputs=[globe])
    noise_sl.change(on_noise, inputs=[noise_sl], outputs=[noise_info])

    # poll agent every 3s → refresh globe + audio if agent fired
    timer = gr.Timer(3.0)
    timer.tick(poll_agent, outputs=[globe, audio, agent_box])

if __name__ == "__main__":
    demo.launch()