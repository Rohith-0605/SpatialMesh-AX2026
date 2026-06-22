# SpatialMesh — Agentic AI & Open-Weight Models Documentation
### `docs/ax.md` | Samsung EnnovateX AX Hackathon 2026 | Team 8

---

## 1. Overview

SpatialMesh is built around two distinct layers of agentic intelligence:

1. **The Spatial Context Agent** — a real-time, rule-based monitor-reason-act loop embedded in the system that makes autonomous spatial decisions every 3 seconds using four specialized tool calls.
2. **The GATv2Conv GNN** — a learned spatial reasoning engine that autonomously assigns optimal 3D positions to each speaker by reasoning over the entire call graph simultaneously.

Both layers are deliberately open-weight and locally executable — no external API calls, no LLM inference at runtime. This is a core design constraint driven by the <20ms latency KPI.

---

## 2. Open-Weight Models Used

### 2.1 GATv2Conv — Graph Attention Network (Core Spatial Brain)

The primary learned model in SpatialMesh is a 2-layer Graph Attention Network using **GATv2Conv** from **PyTorch Geometric** — a fully open-weight, locally executable graph neural network architecture.

- **Weights:** Trained from scratch on 8,000 synthetic spatial scenes
- **Parameters:** 139,000 (~0.53MB)
- **Inference:** CPU-only, 0.63ms per forward pass
- **Open-weight:** Yes — weights published on HuggingFace, fully reproducible

**Why GATv2Conv over other architectures:**

GATv2Conv was chosen over plain GATConv after discovering that standard GATConv silently ignores edge features in its attention computation. GATv2Conv computes attention as `a(W[h_i || h_j || e_ij])` — the 7-dimensional acoustic edge features directly influence which speaker relationships the network attends to. This was a critical discovery that improved separability by ~8 points.

### 2.2 CNN Audio Encoder

A 4-layer 1D Convolutional Neural Network trained from scratch:

- **Weights:** Trained on LibriSpeech + SONICOM HRTF data
- **Parameters:** ~4.25MB
- **Inference:** CPU-only, 2.35ms per 8s segment
- **Open-weight:** Yes — weights published on HuggingFace

### 2.3 MediaPipe FaceLandmarker

Open-weight face landmark detection model from Google MediaPipe:

- Used for real-time head yaw estimation (10fps)
- Landmarks 1 (nose tip), 234 (left face edge), 454 (right face edge)
- Model: `face_landmarker.task` (float16, ~3.6MB)
- Running mode: IMAGE (synchronous, low-latency)

### 2.4 SONICOM HRTF Dataset

399-subject Head-Related Transfer Function dataset used as the acoustic rendering engine:

- 793 measured positions per subject
- FreeFieldComp_48kHz.sofa format
- Used for nearest-position lookup and binaural convolution
- Fully open, academic dataset

---

## 3. Agentic AI Setup

### 3.1 Architecture Philosophy

The Spatial Context Agent is deliberately **not an LLM-based agent**. This was a reasoned architectural decision:

| Approach | Latency | Determinism | Open-weight | Decision |
|----------|---------|-------------|-------------|----------|
| LLM agent (GPT/Claude API) | 100-500ms | No | No | ❌ Rejected |
| Local LLM (Llama/Mistral) | 50-200ms | No | Yes | ❌ Rejected |
| Rule-based monitor-reason-act | <1ms | Yes | N/A | ✅ Chosen |

The <20ms inference KPI and the requirement for fully local execution made LLM-based agents incompatible. The rule-based approach is not a limitation — it is the correct tool for this problem. Spatial context changes (mute events, head turns, noise spikes) are deterministic, observable signals that do not require language understanding or chain-of-thought reasoning.

### 3.2 Agent Loop Structure

```
Every 3 seconds (background thread):
  │
  MONITOR: Read shared state
  │  - activity mask changed?
  │  - noise_level > 0.6?
  │  - |head_yaw - prev_yaw| > 35°?
  │  - 2+ active speakers, no other action?
  │
  REASON: Evaluate conditions in priority order
  │  1. Activity change → trigger_gnn_reassign (highest priority)
  │  2. Noise spike → boost_ild_separation
  │  3. Head yaw shift → update_world_lock
  │  4. Dominant speaker → set_speaker_priority (lowest priority)
  │
  ACT: Fire at most one tool per cycle
  │  (10-second cooldown per tool prevents thrashing)
  │
  UPDATE: Write agent log, update prev_activity, prev_yaw
```

### 3.3 Cooldown System

Each tool has a 10-second individual cooldown tracked via `_last_action` dictionary:

```python
def can_fire(name):
    now = time.time()
    if now - _last_action.get(name, 0) < ACTION_COOLDOWN:
        return False
    _last_action[name] = now
    return True
```

This prevents the agent from repeatedly firing the same tool on the same trigger signal. Without cooldowns, a sustained noise level above 0.6 would re-fire `boost_ild_separation` every 3 seconds, causing unstable position oscillation.

---

## 4. Agentic Workflows

### 4.1 Mute/Unmute Workflow

```
User mutes Speaker A
    │
    ▼
activity_mask: [1,1,1,1] → [0,1,1,1]
    │
    ▼ (within 3s, agent cycle fires)
agent detects: activity_changed = True
    │
    ▼
trigger_gnn_reassign("mute change")
    │
    ▼
Full pipeline re-runs with new activity_mask
CNN embeddings recomputed (muted speaker has rms=0, activity_flag=0)
GNN reasons over updated graph → new (az,el) for B, C, D
    │
    ▼
Globe updates, new binaural mix plays
Agent log: "↺ GNN reassign (mute change)"
```

**What the GNN does differently with 3 vs 4 active speakers:** With Speaker A muted, its node features carry `activity_flag=0` and `rms_energy=0`. The GNN's attention mechanism learns to down-weight edges involving inactive nodes. The three active speakers redistribute to a wider fan — the GNN autonomously fills the vacated space.

### 4.2 Noise Response Workflow

```
User slides noise level above 0.6
    │
    ▼
agent detects: noise_level > NOISE_THRESH (0.6)
    │
    ▼
boost_ild_separation() fires:
    Step 1: Read current pos_norm
    Step 2: Widen all active speaker azimuths by 20%
            az_new = clip(az_current × 1.2, -110°, +110°)
    Step 3: Write widened positions to pos_norm (GNN seed)
    Step 4: Call trigger_gnn_reassign("noise spike → ILD widen")
    │
    ▼
GNN re-fires from widened seed → produces even wider separation
New binaural mix with increased ILD rendered
Agent log: "📢 ↺ GNN reassign (noise spike → ILD widen)"
```

This is **tool chaining** — `boost_ild_separation` calls `trigger_gnn_reassign` internally. The noise tool modifies the GNN's seed state, then delegates the actual spatial reasoning to the GNN tool. This preserves the architectural invariant that only `trigger_gnn_reassign` produces new GNN outputs.

### 4.3 Head Yaw / World-Lock Workflow

```
User turns head 40° to the right
    │
    ▼
MediaPipe FaceLandmarker detects landmarks every 100ms
yaw = arctan2(nose_offset, face_width × 0.5) = +40°
    │
    ▼
agent detects: |yaw - prev_yaw| = 40° > YAW_THRESH (35°)
    │
    ▼
update_world_lock(yaw_delta=+40°)
    Step 1: Read current pos_norm
    Step 2: Subtract yaw_delta from all speaker azimuths
            az_new = az_current - 40°  (rotate soundfield left)
    Step 3: Write to pos_deg (display) only
    Step 4: Set dirty=True → UI polls and updates globe
    │
    ▼
Globe rotates, speakers appear to stay in their
world-absolute positions despite head movement
Agent log: "🔒 world-lock (yaw +40°)"
```

**Critical distinction:** `update_world_lock` does NOT re-fire the GNN. It modifies only `pos_deg` (the display positions) — not `pos_norm` (the GNN seed). This means the world-lock is a pure display and audio rendering adjustment that preserves the GNN's spatial reasoning for the next cycle.

### 4.4 Dominant Speaker Priority Workflow

```
Agent cycle fires, no other condition met
2+ speakers active
    │
    ▼
Compute RMS for each active speaker from raw clips
dominant_idx = argmax(rms_vals)
    │
    ▼
set_speaker_priority(dominant_idx)
    Step 1: Read current pos_deg (NOT pos_norm)
    Step 2: Nudge dominant speaker toward front-center
            az_new = az_current × 0.6
            el_new = el_current × 0.5
    Step 3: Re-render mix at modified positions
    Step 4: Write to pos_deg and mix only
            pos_norm remains UNTOUCHED
    │
    ▼
Dominant speaker sounds closer and more forward
GNN seed preserved for next reassign cycle
Agent log: "⭐ priority → A (nudged front)"
```

---

## 5. Tool Use & Tool Chaining

### 5.1 Tool Definitions

| Tool | Triggers | Modifies | Chains To |
|------|----------|----------|-----------|
| `trigger_gnn_reassign()` | Activity change, called by other tools | `pos_norm`, `pos_deg`, `mix` | None (terminal tool) |
| `boost_ild_separation()` | noise > 0.6 | `pos_norm` (seed only) | `trigger_gnn_reassign` |
| `update_world_lock()` | head yaw > 35° | `pos_deg` only | None |
| `set_speaker_priority()` | Dominant speaker, fallback | `pos_deg`, `mix` only | None |

### 5.2 The Critical Architectural Rule

The most important tool design decision: **`pos_norm` is exclusively owned by `trigger_gnn_reassign`**.

This rule was learned through failure. Early versions of `set_speaker_priority` wrote modified positions back to `pos_norm`. Because `pos_norm` is used as the GNN's seed for the next forward pass, this caused progressive position collapse — each agent cycle nudged positions toward center, the GNN used those positions as seed, converged to a locally collapsed solution, which the agent then nudged further toward center.

The fix was strict separation of concerns:
- `pos_norm` → GNN's territory. Written only by `trigger_gnn_reassign` after a GNN forward pass.
- `pos_deg` → Agent's territory. Display positions, can be modified by any tool.
- `mix` → Rendered audio. Written by tools that change positions without GNN.

### 5.3 Tool Chaining Example

```
noise spike detected
    │
    └→ boost_ild_separation()
            │
            ├→ modify pos_norm (widen seed)
            │
            └→ trigger_gnn_reassign("noise spike → ILD widen")
                    │
                    ├→ run_pipeline(clips, act, widened_pos_norm)
                    │       │
                    │       ├→ HRTF convolve at current positions
                    │       ├→ CNN embed all speakers
                    │       ├→ Build spatial graph
                    │       ├→ GATv2Conv forward pass
                    │       └→ Re-convolve at new positions
                    │
                    └→ update pos_norm, pos_deg, mix, dirty=True
```

Two tools chain, but only one GNN forward pass occurs. The chain is deterministic and completes in <20ms (GNN) + HRTF render time.

---

## 6. Reasoning & Planning Pipeline

### 6.1 GNN as a Reasoning Engine

The GATv2Conv GNN is itself a spatial reasoning engine — not a rule follower. Given the graph state (who is speaking, how loud, how much overlap, current positions), it reasons about the optimal global layout by:

1. **Message passing:** Each node aggregates information from all neighboring nodes weighted by attention scores
2. **Attention over acoustic relationships:** The 7 edge features tell the GNN which speaker pairs are competing (high overlap, similar timbre) and need maximum separation
3. **Global optimization:** The loss function during training encoded interference, repulsion, elevation, comfort, and stability — the GNN internalized these constraints and applies them at inference without explicit rules

The GNN's reasoning is implicit — encoded in 139,000 learned parameters — but the outcome is principled spatial planning.

### 6.2 Agent Priority Ordering as a Planning Policy

The agent's condition evaluation order is itself a planning policy:

```python
if activity_changed → reassign (structural change, highest priority)
elif noise_spike    → widen (acoustic response)
elif yaw_shifted    → world-lock (perceptual continuity)
elif fallback       → priority (fine-tuning, lowest priority)
```

This ordering reflects a deliberate reasoning about which context changes are most disruptive to spatial coherence. A mute event changes the fundamental graph structure — it must be handled first. Head rotation is a perceptual correction that should not override structural changes. Dominant speaker nudging is a refinement that only fires when nothing else needs attention.

---

## 7. What Worked

### 7.1 GATv2Conv Edge-Aware Attention
Switching from GATConv to GATv2Conv was the single highest-impact decision. The 7 acoustic edge features (spectral correlation, dominance ratio, overlap duration, etc.) gave the GNN exactly the relational information it needed. Separability improved from ~84/100 to 96/100 after this switch.

### 7.2 Repulsion Warmup Scheduling
Training the GNN with high repulsion weight (0.5) early and reducing it (0.05) after epoch 9 solved position collapse during early training without sacrificing nuanced placement in the final model. Without warmup, the GNN either collapsed (too little repulsion) or pushed all speakers to ±110° regardless of context (too much repulsion).

### 7.3 HRTF Convolution Before CNN Embedding
Requiring all audio to be HRTF-convolved before CNN embedding was a critical architectural invariant. Raw mono audio embeddings carry no spatial information — the CNN would produce near-identical embeddings for speakers at different positions. HRTF pre-convolution makes the embeddings spatially grounded, giving the GNN meaningful node features to reason over.

### 7.4 Fixed Wide Initialization
Using a fixed wide initialization `[-0.8, -0.3, +0.3, +0.8]` in normalized azimuth space rather than random initialization ensured the GNN always received in-distribution seed positions. Random initialization occasionally produced clustered seeds that led to collapsed outputs — the GNN is sensitive to its starting point because `pos_norm` feeds directly into the node features.

### 7.5 Strict pos_norm Ownership
Enforcing that only `trigger_gnn_reassign` writes to `pos_norm` eliminated a class of subtle bugs where agent tools were gradually corrupting the GNN's seed state across multiple cycles. This architectural invariant made the system stable across hours of continuous operation.

### 7.6 Sequential Solo-Intro Rendering
Adding a 1.5-second solo preview for each active speaker before the full mix dramatically improved perceptual convincingness. The listener orients to each voice's direction individually before the full scene plays. This is a psychoacoustic insight — the brain needs a reference to lock onto before it can maintain spatial separation in a dense mix.

### 7.7 arctan2 Head Yaw Formula
Replacing the saturating `(nose.x - center_x) × 100` formula with `arctan2(nose_offset, face_width × 0.5)` gave true ±90° head tracking. The old formula produced only ~11° for a full 90° head turn — making world-lock effectively non-functional for large rotations.

---

## 8. What Did Not Work

### 8.1 LLM-Based Spatial Agent
Early conceptual exploration considered using a small local LLM (Mistral 7B via Ollama) as the agent brain — feeding it the current speaker positions, activity states, and context signals, and asking it to decide which tool to call. This was rejected because:
- Inference latency: 80-200ms per decision cycle, incompatible with real-time requirements
- Non-determinism: Same context produced different tool selections across runs
- Overkill: The decision space is small and structured — language reasoning adds no value

### 8.2 Continuous Streaming Pipeline
The initial design targeted true sample-by-sample streaming — processing each 20ms audio frame continuously. This proved architecturally incompatible with the CNN encoder, which needs a minimum segment length to produce meaningful embeddings. The 4-layer CNN with stride-4 first layer requires at least ~1000 samples for stable pooled output. The solution was chunk-based processing (8s segments for demo, 20ms frames for latency measurement) which meets the KPI correctly.

### 8.3 set_speaker_priority Writing to pos_norm
The first implementation of `set_speaker_priority` wrote modified positions back to `pos_norm` to influence the next GNN cycle. This caused progressive position collapse — each agent cycle nudged the dominant speaker toward center, the GNN converged from that biased seed, and over time all speakers drifted toward front-center. Fixed by restricting the tool to write only `pos_deg` and `mix`.

### 8.4 Binaural Mic Input
Live microphone input through the Real-Time Mic tab successfully captures audio and runs it through the full pipeline. However, the binaural effect is not perceptually convincing for live mic audio because:
- Room reflections in the mic capture conflict with the HRTF's directional cues
- The brain cannot resolve spatial position when reverberant audio is convolved with an HRTF — the room's own reflections override the synthetic ITD/ILD
- Spectral subtraction de-reverberation partially mitigated this but did not eliminate it
- Resolution: anechoic or near-field microphone capture would solve this; bedroom mic capture does not

### 8.5 _enforce_separation Post-Processing
An early attempt to post-process GNN outputs by forcing minimum angular separation between speakers (`_enforce_separation`) appeared to help but was actually masking the root cause — GNN position collapse from bad seed inputs. Once the fixed initialization and pos_norm ownership rules were in place, the GNN produced well-separated outputs natively and the post-processing was removed. Lesson: fix the root cause, not the symptom.

---

## 9. Memory & Context Handling

### 9.1 Shared State Dictionary

All system components communicate through a single thread-safe shared state dictionary protected by a `threading.Lock()`:

```python
state = {
    "pos_norm"     : FIXED_INIT.clone(),   # GNN seed — GNN exclusive
    "pos_deg"      : None,                  # Display positions — agent writable
    "activity"     : np.ones(4),            # Current mute mask
    "prev_activity": np.ones(4),            # Previous cycle mute mask
    "distance"     : np.ones(4),            # Radial gain per speaker
    "mix"          : None,                  # Latest rendered binaural mix
    "head_yaw"     : 0.0,                   # Current head yaw from MediaPipe
    "prev_yaw"     : 0.0,                   # Previous cycle head yaw
    "noise_level"  : 0.0,                   # Current noise slider value
    "agent_log"    : [],                    # Last 6 agent actions
    "dirty"        : False,                 # Flag: new mix available for UI
    "mic_buffer"   : None,                  # Rolling 8s mic capture
    "mic_active"   : False,                 # Mic capture running?
}
```

### 9.2 Context Across Agent Cycles

The agent maintains context across cycles via `prev_activity` and `prev_yaw`:
- `prev_activity` enables change detection — the agent only fires `trigger_gnn_reassign` when the mask actually changed, not on every cycle
- `prev_yaw` enables delta computation — the agent responds to head movement, not absolute position

### 9.3 Rolling Mic Buffer

The real-time mic input uses a ring buffer for temporal context:

```python
ring = np.roll(ring, -len(chunk))
ring[-len(chunk):] = new_chunk
```

This maintains the last 8 seconds of mic audio in a rolling window. The most speech-dense 2-second window (highest RMS) is selected for pipeline input, ensuring the CNN receives meaningful audio rather than silence-padded segments.

---

## 10. Multi-Component Orchestration

SpatialMesh orchestrates four concurrent threads:

| Thread | Role | Interval |
|--------|------|----------|
| Main (Gradio) | UI serving, user interaction callbacks | Event-driven |
| Agent thread | Monitor-reason-act loop | Every 3s |
| MediaPipe thread | Head yaw estimation from webcam | Every 100ms (10fps) |
| Mic capture thread | Rolling buffer update from sounddevice | Every 500ms (0.5s chunks) |

All threads share state through the lock-protected dictionary. The Gradio timer (`gr.Timer(3.0)`) polls `state["dirty"]` and pushes new audio/globe updates to the UI when the agent has produced new output.

This is a lightweight multi-agent orchestration — each thread is a specialized agent with a single responsibility, communicating through shared state rather than message passing.

---

## 11. Gradio UI as Agentic Interface

The Gradio interface exposes three interaction modes:

**Live Tab:** Direct human-agent collaboration. The user controls mute states and noise level; the Spatial Context Agent responds autonomously within 3 seconds. The agent log shows exactly what action was taken and why.

**Demo Tab:** Scripted agentic walkthrough. Seven one-click scenario buttons each trigger a complete agent cycle — setting activity mask, firing GNN reassign, updating globe, playing binaural audio. Designed to demonstrate the full agentic loop in a controlled, reproducible sequence for evaluation.

**Real-Time Mic Tab:** Human-in-the-loop spatial audio. The user's voice enters the pipeline as Speaker A. The agent continues to operate on all four speakers including the live mic input.

---

## 12. Summary

SpatialMesh demonstrates that effective agentic AI does not require large language models. The system's two agentic layers — the rule-based Spatial Context Agent and the learned GATv2Conv spatial reasoner — operate at 0.63ms and <10ms respectively, enabling real-time spatial audio that no LLM-based approach could achieve within the latency constraints.

The key lessons: enforce strict ownership of shared state between agent tools, use the simplest agent architecture that correctly solves the problem, and treat the learned GNN as a reasoning engine rather than a lookup table. The GNN's 96/100 separability score on hard multi-speaker scenes demonstrates that learned spatial reasoning outperforms rule-based position assignment — but only when the graph structure, edge features, and loss function are designed to encode the right acoustic relationships.
