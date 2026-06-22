"""
SpatialMesh -- Immersive Spatial Voice Call (full demo).

ANGLE -> GNN auto-assigns (az, el) for separability (GNN owns this).
The Spatial Context Agent (background, every 3s) reacts to EXTERNAL context:
  - mute/unmute   -> trigger_gnn_reassign()   (re-fires GNN)
  - noise spike   -> boost_ild_separation()   (widen azimuth + radial push-back)
  - head yaw shift -> update_world_lock()
  - dominant spk  -> set_speaker_priority()   (pull closer + nudge front)

A radial "distance" gain layer still exists internally (the agent uses it),
but there are NO manual distance sliders in the UI -- distance is agent-driven.

DEMO tab: 7 one-click scenarios that set the activity mask, re-fire the GNN,
update the globe, and play the binaural mix -- ideal for the demo video.
"""

import threading, time, urllib.request, os
import numpy as np
import plotly.graph_objects as go
import gradio as gr
import cv2
import mediapipe as mp
import torch
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
# distance: 1.0 = reference. <1 = farther (quieter), >1 = closer (louder).
# (agent-driven only; no UI sliders)

# NEW — fixed wide spread, always in-distribution for the GNN

FIXED_INIT = torch.tensor([
    [-0.8,  0.1],
    [-0.3, -0.1],
    [ 0.3,  0.1],
    [ 0.8, -0.1],
], dtype=torch.float32)

state = {
    "pos_norm" : FIXED_INIT.clone(),
    "pos_deg"      : None,
    "activity"     : np.ones(4, dtype=np.float32),
    "prev_activity": np.ones(4, dtype=np.float32),
    "distance"     : np.ones(4, dtype=np.float32),   # radial, per speaker
    "mix"          : None,
    "head_yaw"     : 0.0,
    "prev_yaw"     : 0.0,
    "noise_level"  : 0.0,
    "agent_log"    : [],
    "lock"         : threading.Lock(),
    "dirty"        : False,
}


def dist_to_gain(d):
    """Distance multiplier -> linear gain. d>1 closer/louder, d<1 farther/quieter."""
    return float(np.clip(d, 0.25, 2.0))


# ===========================================================================
# AGENT TOOL CALLS
# ===========================================================================
def trigger_gnn_reassign(reason):
    with state["lock"]:
        act   = state["activity"].copy()
        prev  = state["pos_norm"]
        gains = [dist_to_gain(d) for d in state["distance"]]
    deg, norm, mix = core.run_pipeline(CLIPS, act, prev, gains)
    with state["lock"]:
        state["pos_norm"] = norm
        state["pos_deg"]  = deg
        state["mix"]      = mix
        state["dirty"]    = True
    return f"↺ GNN reassign ({reason})"


def boost_ild_separation():
    """Noise response: widen azimuth on current positions, then re-fire GNN for new mix."""
    with state["lock"]:
        norm = state["pos_norm"].clone()
        act  = state["activity"].copy()

    # widen azimuth of current positions
    deg = core.denorm(norm)
    for i in range(len(deg)):
        if act[i] == 1:
            deg[i] = (float(np.clip(deg[i][0] * 1.2, -core.AZ_MAX, core.AZ_MAX)), deg[i][1])

    new_norm = _renorm(deg)
    with state["lock"]:
        state["pos_norm"] = new_norm  # GNN will start from widened positions

    # now re-fire GNN from the widened seed -> new mix produced
    result = trigger_gnn_reassign("noise spike → ILD widen")
    return f"📢 {result}"


def update_world_lock(yaw_delta):
    with state["lock"]:
        norm = state["pos_norm"].clone()
    deg = core.denorm(norm)
    for i in range(len(deg)):
        deg[i] = (deg[i][0] - yaw_delta, deg[i][1])
    with state["lock"]:
        state["pos_norm"] = _renorm(deg)
        state["pos_deg"]  = deg
        state["dirty"]    = True
    return f"🔒 world-lock (yaw {yaw_delta:+.0f}°)"


def set_speaker_priority(idx):
    """Dominant speaker: nudge toward front in the MIX only.
    Does NOT update pos_norm so GNN seed stays clean."""
    with state["lock"]:
        deg  = list(state["pos_deg"]) if state["pos_deg"] else list(core.denorm(state["pos_norm"]))
        act  = state["activity"].copy()

    # nudge dominant speaker toward front — temporary render only
    priority_deg = list(deg)  # copy, don't modify state["pos_deg"]
    az, el = priority_deg[idx]
    priority_deg[idx] = (az * 0.6, el * 0.5)  # pull toward center

    # re-render mix at priority positions WITHOUT updating pos_norm
    from spatialmesh_core import _hrtf
    stereo_out = [_hrtf.convolve(CLIPS[i], *priority_deg[i]) for i in range(4)]
    mix = core.render_mix(stereo_out, act)

    with state["lock"]:
        # only update mix and pos_deg for display
        # pos_norm stays UNTOUCHED so GNN seed is never corrupted
        state["pos_deg"] = priority_deg
        state["mix"]     = mix
        state["dirty"]   = True

    return f"⭐ priority → {SPK_NAMES[idx]} (nudged front, GNN seed preserved)"


def _renorm(deg):
    import torch
    t = torch.tensor([[az / core.AZ_MAX, el / core.EL_MAX] for az, el in deg],
                     dtype=torch.float32)
    return t.clamp(-1, 1)


# ===========================================================================
# AGENT LOOP
# ===========================================================================
NOISE_THRESH    = 0.6
YAW_THRESH      = 35.0     # deliberate head turn, not fidgeting (was 15.0)
AGENT_INTERVAL  = 3.0
ACTION_COOLDOWN = 10.0
_last_action = {}


def can_fire(name):
    now = time.time()
    if now - _last_action.get(name, 0) < ACTION_COOLDOWN:
        return False
    _last_action[name] = now
    return True


def agent_step():
    actions = []
    with state["lock"]:
        act      = state["activity"].copy()
        prev_act = state["prev_activity"].copy()
        deg      = state["pos_deg"]
        yaw      = state["head_yaw"]
        prev_yaw = state["prev_yaw"]
        noise    = state["noise_level"]

    activity_changed = not np.array_equal(act, prev_act)
    noise_spike      = noise > NOISE_THRESH
    yaw_shifted      = abs(yaw - prev_yaw) > YAW_THRESH
    active_idx       = [i for i in range(4) if act[i] == 1]

    # GNN only re-fires on external activity change
    if activity_changed and can_fire("reassign"):
        actions.append(trigger_gnn_reassign("mute change"))

    # noise -> radial + angular widen (no GNN)
    if noise_spike and can_fire("ild_boost"):
        actions.append(boost_ild_separation())

    # head turn -> world lock (no GNN)
    if yaw_shifted and can_fire("world_lock"):
        actions.append(update_world_lock(yaw - prev_yaw))

    # dominant speaker -> radial pull closer (no GNN)
    if not actions and active_idx and can_fire("priority"):
     if deg is not None and len(active_idx) > 1:
        with state["lock"]:
            dom = state["distance"].copy()
        # pick the speaker with highest rms — proxy for dominance
        rms_vals = [float(np.sqrt(np.mean(CLIPS[i]**2))) if act[i] else -1
                    for i in range(4)]
        dominant_idx = int(np.argmax(rms_vals))
        if act[dominant_idx] == 1:
            actions.append(set_speaker_priority(dominant_idx))
  

    with state["lock"]:
        state["prev_activity"] = act.copy()
        state["prev_yaw"]      = yaw
        if actions:
            ts = time.strftime("%H:%M:%S")
            state["agent_log"] = (state["agent_log"] +
                                  [f"[{ts}] {a}" for a in actions])[-6:]
    return actions


def agent_thread():
    while True:
        time.sleep(AGENT_INTERVAL)
        try:
            agent_step()
        except Exception as e:
            with state["lock"]:
                state["agent_log"].append(f"[agent err] {e}")


threading.Thread(target=agent_thread, daemon=True).start()


# ---- MediaPipe face landmarker (head yaw) ----------------------------------
_TASK_PATH = "face_landmarker.task"
if not os.path.exists(_TASK_PATH):
    print("Downloading face_landmarker.task (~3.6MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task", _TASK_PATH)
    print("Downloaded.")

_fl_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=_TASK_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_facial_transformation_matrixes=False,
    output_face_blendshapes=False,
)
_detector = mp.tasks.vision.FaceLandmarker.create_from_options(_fl_options)


def _get_yaw(landmarks):
    """True ±90° yaw using arctan2 -- covers full head-turn range.

    The old (nose.x - center_x)*100 formula saturates badly: a full 90° turn
    only produced ~11°, because nose screen-x barely moves at large angles.
    arctan2(nose_offset, half_face_width) maps the offset to a real angle that
    spans the full range without saturating.
    """
    nose       = landmarks[1]
    left_face  = landmarks[234]
    right_face = landmarks[454]
    face_width  = right_face.x - left_face.x
    nose_offset = nose.x - (left_face.x + right_face.x) / 2
    # half_face_width as the adjacent leg -> arctan gives a true angle
    return float(np.degrees(np.arctan2(nose_offset, face_width * 0.5)))


def _yaw_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[MediaPipe] ⚠ Webcam not available — head yaw locked at 0°")
        return
    print("[MediaPipe] ✅ Webcam opened, head tracking active")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _detector.detect(mp_img)
        if result.face_landmarks:
            yaw = _get_yaw(result.face_landmarks[0])
            with state["lock"]:
                state["head_yaw"] = yaw
        time.sleep(0.1)   # 10fps is enough for world-lock


threading.Thread(target=_yaw_thread, daemon=True).start()


# ===========================================================================
# GLOBE -- distance shown as marker SIZE
# ===========================================================================
def sph_to_xyz(az, el, r=1.0):
    a, e = np.radians(az), np.radians(el)
    return r*np.cos(e)*np.sin(a), r*np.cos(e)*np.cos(a), r*np.sin(e)


def make_globe(positions_deg, activity, distance):
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
                      line=dict(color="#c2c8cf", width=1),
                      hoverinfo="skip", showlegend=False)
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                      marker=dict(size=6, color="#111"), text=["You"],
                      textposition="bottom center", hoverinfo="skip",
                      showlegend=False)
    fig.add_scatter3d(x=[0,0], y=[0,1.15], z=[0,0], mode="lines+text",
                      line=dict(color="#111", width=3),
                      text=["","front 0°"], textposition="top center",
                      hoverinfo="skip", showlegend=False)
    for i, (az, el) in enumerate(positions_deg):
        on = activity[i] == 1
        d  = float(np.clip(distance[i], 0.25, 2.0))
        x, y, z = sph_to_xyz(az, el, 1.0)
        col = SPK_COLORS[i] if on else "#cfcfcf"
        fig.add_scatter3d(x=[0,x], y=[0,y], z=[0,z], mode="lines",
                          line=dict(color=col, width=5 if on else 2),
                          hoverinfo="skip", showlegend=False)
        size = (8 + 10*(d-0.25)/1.75) if on else 6   # bigger = closer/louder
        tag = SPK_NAMES[i] + ("" if on else " ✗")
        dist_word = "near" if d > 1.1 else ("far" if d < 0.9 else "mid")
        fig.add_scatter3d(x=[x], y=[y], z=[z], mode="markers+text",
                          marker=dict(size=size, color=col,
                                      line=dict(color="#fff", width=1)),
                          text=[f"{tag}<br>{az:.0f}°,{el:.0f}° · {dist_word}"],
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
    return d if d is not None else core.denorm(state["pos_norm"])


# ===========================================================================
# CORE UI CALLBACKS
# ===========================================================================
def on_mute(ma, mb, mc, md):
    act = np.array([0.0 if m else 1.0 for m in (ma,mb,mc,md)], dtype=np.float32)
    with state["lock"]:
        state["activity"] = act
        dist = state["distance"].copy()
    deg = _current_deg()
    return (make_globe(deg, act, dist),
            "Agent will auto-reassign in ≤3s...")


def on_reassign(ma, mb, mc, md):
    act = np.array([0.0 if m else 1.0 for m in (ma,mb,mc,md)], dtype=np.float32)
    with state["lock"]:
        state["activity"] = act
        prev  = state["pos_norm"]
        gains = [dist_to_gain(d) for d in state["distance"]]
        dist  = state["distance"].copy()
    deg, norm, mix = core.run_pipeline(CLIPS, act, prev, gains)
    with state["lock"]:
        state["pos_norm"] = norm
        state["pos_deg"]  = deg
        state["mix"]      = mix
    audio = (core.SR, (mix * 32767).astype(np.int16))
    label = "  |  ".join(
        f"{SPK_NAMES[i]}: {deg[i][0]:.0f}°,{deg[i][1]:.0f}°" +
        ("" if act[i] else " ✗") for i in range(4))
    return make_globe(deg, act, dist), audio, label


def on_noise(noise):
    with state["lock"]:
        state["noise_level"] = float(noise)
    return f"Noise level: {noise:.2f}  (>0.6 triggers agent)"


def poll_agent():
    with state["lock"]:
        log   = state["agent_log"].copy()
        dirty = state["dirty"]
        deg   = state["pos_deg"]
        act   = state["activity"].copy()
        dist  = state["distance"].copy()
        mix   = state["mix"]
        yaw   = state["head_yaw"]
        state["dirty"] = False
    log_str = "\n".join(log) if log else "Agent monitoring..."
    yaw_str = f"**Head yaw:** {yaw:+.1f}° (webcam tracking)"
    if dirty and deg is not None and mix is not None:
        audio = (core.SR, (mix * 32767).astype(np.int16))
        return make_globe(deg, act, dist), audio, log_str, yaw_str
    return make_globe(_current_deg(), act, dist), gr.update(), log_str, yaw_str


# ===========================================================================
# DEMO SCENARIO PLAYER
# ===========================================================================
# (label, description, activity_mask)
DEMO_SCENARIOS = [
    ("S1 · Only A",        "Only Speaker A is active.",                 [1, 0, 0, 0]),
    ("S2 · Only B",        "Only Speaker B is active.",                 [0, 1, 0, 0]),
    ("S3 · C + D",         "Speakers C and D active, A and B silent.",  [0, 0, 1, 1]),
    ("S4 · All four",      "All four speakers active simultaneously.",  [1, 1, 1, 1]),
    ("S5 · A muted",       "A muted → GNN reassigns B, C, D.",          [0, 1, 1, 1]),
    ("S6 · B muted",       "B muted → GNN reassigns A, C, D.",          [1, 0, 1, 1]),
    ("S7 · A + B muted",   "A and B muted → only C and D remain.",      [0, 0, 1, 1]),
]


def run_demo_scenario(scenario_idx):
    """Set the activity mask for a scenario, re-fire the GNN, and play the mix.
    Uses the CURRENT layout as prev so the rearrangement is visible (real-time
    adaptation story), exactly as the agent would do on a live mute change."""
    label, description, mask_list = DEMO_SCENARIOS[scenario_idx]
    act = np.array(mask_list, dtype=np.float32)

    with state["lock"]:
        state["activity"] = act
        # reset distance to reference so demo audio is clean/comparable
        state["distance"] = np.ones(4, dtype=np.float32)
        prev = state["pos_norm"]
    deg, norm, mix = core.run_pipeline(CLIPS, act, prev)
    with state["lock"]:
        state["pos_norm"] = norm
        state["pos_deg"]  = deg
        state["mix"]      = mix
        dist = state["distance"].copy()

    audio = (core.SR, (mix * 32767).astype(np.int16))

    # Build a readable position/status table for the video overlay
    rows = []
    for i in range(4):
        status = "ACTIVE" if mask_list[i] else "muted"
        rows.append(
            f"**{SPK_NAMES[i]}** [{status}] — az `{deg[i][0]:+.0f}°`, el `{deg[i][1]:+.0f}°`")
    table = "  \n".join(rows)

    info_md = (
        f"### ▶ {label}\n"
        f"{description}\n\n"
        f"{table}\n\n"
        f"_Audio plays each active speaker solo first, then the full mix._")

    # sync the mute checkboxes so the UI stays consistent
    mute_states = [bool(1 - m) for m in mask_list]  # checkbox = muted = not active

    return (make_globe(deg, act, dist), audio, info_md,
            mute_states[0], mute_states[1], mute_states[2], mute_states[3])


# ===========================================================================
# UI LAYOUT
# ===========================================================================
init_deg  = core.denorm(state["pos_norm"])
init_act  = np.ones(4, dtype=np.float32)
init_dist = np.ones(4, dtype=np.float32)

with gr.Blocks(title="SpatialMesh", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎧 SpatialMesh — Immersive Spatial Voice Call\n"
        "GNN auto-assigns each speaker an (az, el) for maximum separability. "
        "Mute a speaker → the Spatial Context Agent re-fires the GNN within ~3s. "
        "**Put headphones on.**")

    with gr.Tabs():
        # ---------------------------------------------------------------
        # TAB 1 — Live
        # ---------------------------------------------------------------
        with gr.Tab("Live"):
            with gr.Row():
                with gr.Column(scale=3):
                    globe = gr.Plot(make_globe(init_deg, init_act, init_dist),
                                    label="3D Soundscape")
                with gr.Column(scale=1):
                    gr.Markdown("### Mute speakers")
                    ma = gr.Checkbox(label="Mute A")
                    mb = gr.Checkbox(label="Mute B")
                    mc = gr.Checkbox(label="Mute C")
                    md = gr.Checkbox(label="Mute D")
                    btn = gr.Button("Re-assign now (GNN)", variant="primary")

            with gr.Row():
                yaw_display = gr.Markdown("**Head yaw:** 0.0° (webcam tracking)")
                noise_sl    = gr.Slider(0, 1, value=0, step=0.05,
                                        label="Noise level — agent widen + push-back")
            noise_info = gr.Markdown("Noise level: 0.00  (>0.6 triggers agent)")

            audio = gr.Audio(label="Binaural output (headphones)", type="numpy")
            info  = gr.Markdown("Press Re-assign or mute a speaker to start.")
            agent_box = gr.Textbox(label="🤖 Spatial Context Agent log", lines=6,
                                   interactive=False, value="Agent monitoring...")

        # ---------------------------------------------------------------
        # TAB 2 — Demo (one-click scenarios)
        # ---------------------------------------------------------------
        with gr.Tab("Demo"):
            gr.Markdown(
                "### Scripted scenario walkthrough\n"
                "Click each scenario in order. The globe rearranges, the agent "
                "log updates, and the binaural mix plays each active speaker "
                "solo (1.5s) before the full mix — perfect for the demo video.")
            with gr.Row():
                demo_btns = [gr.Button(DEMO_SCENARIOS[i][0],
                                       variant="secondary")
                             for i in range(len(DEMO_SCENARIOS))]
            demo_globe = gr.Plot(make_globe(init_deg, init_act, init_dist),
                                 label="3D Soundscape (demo)")
            demo_audio = gr.Audio(label="Binaural output (headphones)",
                                  type="numpy", autoplay=True)
            demo_info  = gr.Markdown("Pick a scenario above to begin.")

    # ---- wiring: Live tab ----
    mute_inputs = [ma, mb, mc, md]
    for cb in mute_inputs:
        cb.change(on_mute, inputs=mute_inputs, outputs=[globe, info])
    btn.click(on_reassign, inputs=mute_inputs, outputs=[globe, audio, info])
    noise_sl.change(on_noise, inputs=[noise_sl], outputs=[noise_info])

    timer = gr.Timer(3.0)
    timer.tick(poll_agent, outputs=[globe, audio, agent_box, yaw_display])

    # ---- wiring: Demo tab ----
    # each demo button also writes back into the Live-tab mute checkboxes
    for i, b in enumerate(demo_btns):
        b.click(
            lambda idx=i: run_demo_scenario(idx),
            inputs=None,
            outputs=[demo_globe, demo_audio, demo_info, ma, mb, mc, md],
        )

if __name__ == "__main__":
    demo.launch()