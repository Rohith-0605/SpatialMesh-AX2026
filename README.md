# SpatialMesh — Immersive 3D Spatial Audio for Multi-Party Voice Calls

> **Samsung EnnovateX AX Hackathon 2026 | Problem Statement 8 | Team SpatialMesh | IIT Guwahati**

---

## Submission Details

| Field | Details |
|-------|---------|
| Project Name | Immersive Spatial Voice Call Experience with AI |
| Problem Statement | 8 — Immersive Spatial Voice Call Experience with AI |
| Team Name | SpatialMesh |
| Team Members | Rohith Sai Anvesh Polisetty, Siddharth Sivapuram |
| Institute | IIT Guwahati |
---

## Demo Video

**Full Demo:** | [Watch Demo](https://youtu.be/VT_2EgvCe5k)|
🛠️ Installation Guide (YouTube) | [Watch Guide](https://youtu.be/PEdSFaae3fc) |
| 📊 Presentation (PDF) | [View Slides](https://drive.google.com/file/d/1mayaPPAfyEff-Vt1P6nm2eUfwn0psPMj/view?usp=sharing) |

*(Put headphones on before watching — binaural effect only works on headphones)*

---

## What is SpatialMesh?

Every voice call today is monaural — all speakers arrive from the same direction. When four people speak simultaneously, your brain has no spatial cues to separate them.

SpatialMesh solves this using a **Graph Attention Network (GATv2Conv)** that models the entire multi-party call as a spatial graph. Each speaker is a node carrying a learned CNN audio embedding. Each directed edge carries 7 acoustic relationship features. The GNN jointly reasons over all speakers simultaneously and outputs optimal azimuth and elevation coordinates — placing each voice at a distinct, perceptually convincing position in 3D space.

**Put your headphones on. This system requires them.**

---

## Product Vision — Built for the Samsung Ecosystem

SpatialMesh is designed to slot directly into Samsung's existing product lines:

### Galaxy Buds Pro
SpatialMesh runs entirely on-device. The 4.81MB model fits comfortably in Galaxy Buds DSP headroom. Real-time 3D spatial calls with zero cloud dependency and zero latency penalty. Every Galaxy Buds call becomes immersive.

### Samsung Meet / Galaxy AI Calls
Drop-in SDK for Samsung's call stack. Every Galaxy AI call becomes spatially immersive — placing each participant at a distinct 3D position. Differentiates Galaxy from Apple and Google in the enterprise and consumer markets.

### SmartThings Spatial
Multi-room spatial audio — different call participants placed at different room positions. Natural conversational dynamics for distributed family calls and remote team meetings across SmartThings-connected devices.

> **Open-weight · CPU-only · No cloud dependency · Deployable today**

---

## KPIs — All Met

| KPI | Target | Achieved |
|-----|--------|----------|
| GNN Inference Latency | < 20ms | **0.63ms** |
| Input Latency (20ms frame) | 40–60ms | **8.5ms** |
| Model Size | < 50MB | **4.81MB** |
| Spatial Separability | 95%+ | **96/100** |
| Concurrent Speakers | 3+ | **4** |
| UI Responsiveness | < 100ms | **~50ms** |

---

## System Architecture

```
Raw Mono Audio (4 speakers)
        │
        ▼
[1] HRTF Pre-Convolution (SONICOM, 793 positions)
        │
        ▼
[2] CNN Audio Encoder (4-layer 1D CNN → 128-dim embedding)
        │
        ▼
[3] Spatial Graph (133-dim nodes, 7-dim directed edges)
        │
        ▼
[4] GATv2Conv GNN (2 layers, 4 heads → az, el per speaker)
        │
        ▼
[5] HRTF Re-convolution at new positions
        │
        ▼
Binaural Output → Headphones
```

### Spatial Context Agent (4 Tool Calls)
A rule-based monitor-reason-act loop running every 3 seconds:
- `trigger_gnn_reassign()` — fires on mute/unmute events
- `boost_ild_separation()` — fires on noise spikes, widens azimuth
- `update_world_lock()` — fires on head yaw > 35 degrees, rotates soundfield
- `set_speaker_priority()` — nudges dominant speaker toward front

---

## Models

### Models Used
- **PyTorch Geometric GATv2Conv** — open-weight graph attention network
- **MediaPipe FaceLandmarker** — open-weight head tracking (Google)

### Models Published
| Model | Size | Link |
|-------|------|------|
| CNN Audio Encoder | 4.25MB | https://huggingface.co/rohith-1719/spatialmesh-models |
| GAT-GNN Spatial Reasoner | 0.57MB | https://huggingface.co/rohith-1719/spatialmesh-models |

---

## Datasets

### Datasets Used
| Dataset | Purpose | Link |
|---------|---------|------|
| SONICOM HRTF | Binaural rendering, CNN training | https://www.sonicom.eu/dataset |
| LibriSpeech test-clean | CNN training, demo audio | https://www.openslr.org/12 |

### Datasets Published
| Dataset | Description | Link |
|---------|-------------|------|
| SpatialMesh GNN Training Data | 8,000 synthetic spatial scenes (35% hard cases) | https://huggingface.co/datasets/rohith-1719/spatialmesh-data |

---

## Installation

### Requirements
- Python 3.10+
- Headphones (required for binaural effect)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Rohith-0605/SpatialMesh-AX2026
cd SpatialMesh-AX2026/test_run

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pretrained models
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('models', exist_ok=True)
hf_hub_download('rohith-1719/spatialmesh-models', 'cnn_encoder_best.pt', local_dir='models/')
hf_hub_download('rohith-1719/spatialmesh-models', 'gat_best.pt', local_dir='models/')
print('Models ready.')
"

# 5. Download SONICOM HRTF
# Download P0001_FreeFieldComp_48kHz.sofa from https://www.sonicom.eu/dataset
# Place at: sonicom/P0001/FreeFieldComp_48kHz.sofa

# 6. Add LibriSpeech clips
# Download any 4 flac clips from speaker 237 at https://www.openslr.org/12
# Place at: librispeech/spk_237/

# 7. Run
python app.py
```

Open your browser at `http://127.0.0.1:7860`

### File Structure
```
test_run/
├── models/
│   ├── cnn_encoder_best.pt      # Trained CNN encoder
│   └── gat_best.pt              # Trained GAT-GNN
├── sonicom/P0001/
│   └── FreeFieldComp_48kHz.sofa # HRTF measurements
├── librispeech/spk_237/         # 4 LibriSpeech flac clips
├── spatialmesh_core.py          # Full pipeline
├── spatialmesh_gat.py           # GATv2Conv model definition
├── app.py                       # Gradio UI (Live, Demo, Mic tabs)
├── kpi_final.py                 # KPI measurement script
└── requirements.txt
```

---

## Usage

### Demo Tab (Recommended for first run)
Click scenarios S1 through S7 in order. Each scenario:
1. Sets a different speaker activity mask
2. Re-fires the GNN
3. Updates the 3D globe
4. Plays solo intros per speaker then full binaural mix

### Live Tab
- Toggle mute checkboxes — agent auto-reassigns within 3 seconds
- Slide noise level above 0.6 — agent widens speaker separation
- Turn your head — MediaPipe tracks yaw, agent applies world-lock

### Real-Time Mic Tab
- Press Start Mic — your voice replaces Speaker A
- Wait 2 seconds for buffer to fill
- Press Spatialize — your voice is placed in 3D space

---

## Reproduce KPIs

```bash
python kpi_final.py
```

Output:
```
GNN Inference Latency : 0.63ms   (target <20ms)
Input Latency         : 8.5ms    (target 40-60ms)
Model Size            : 4.81MB   (target <50MB)
Spatial Separability  : 96/100   (target 95%+)
```

---

## Technical Documentation

- [`SpatialMesh_Technical_Documentation.md`](SpatialMesh_Technical_Documentation.md) — Full architecture, CNN training, GNN training, HRTF rendering, KPI results
- [`ax.md`](ax.md) — Agentic AI setup, tool calls, workflows, what worked and what did not work

---

## Open Source Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| PyTorch | Deep learning framework | https://pytorch.org |
| PyTorch Geometric | GATv2Conv graph neural network | https://pyg.org |
| Librosa | Audio processing | https://librosa.org |
| SoundFile | Audio I/O | https://pysoundfile.readthedocs.io |
| Sofar | SOFA HRTF file reader | https://github.com/spatialaudio/sofar |
| SciPy | FFT convolution | https://scipy.org |
| MediaPipe | Face landmark detection | https://mediapipe.dev |
| Gradio | Web UI | https://gradio.app |
| Plotly | 3D globe visualization | https://plotly.com |
| OpenCV | Webcam capture | https://opencv.org |
| SoundDevice | Microphone input | https://python-sounddevice.readthedocs.io |

---

## Attribution

This project was built from scratch for Samsung EnnovateX AX Hackathon 2026. No existing open source project was used as a base. All trained model weights (CNN encoder, GAT-GNN) were developed by the team and are published under MIT license on HuggingFace.

---

## License

MIT License
