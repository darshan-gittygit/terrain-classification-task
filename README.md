# Terrain Classification for Wearable Robotics
### DRDO AI/ML Internship Task

A two-phase deep learning pipeline for real-time terrain classification, designed as the perception layer for wearable robotics systems. The model identifies ground terrain type from images, enabling adaptive locomotion control.

---

## Problem Statement

Wearable robots need to identify the surface they are walking on — asphalt, grass, sand, soil, etc. — to adjust their gait and balance in real time. This project builds and compares two approaches:

- **Phase 1:** Custom CNN trained from scratch (baseline)
- **Phase 2:** MobileNetV2 transfer learning (deployment-ready)

---

## Dataset

**GTOS-Mobile** — Ground Terrain for Outdoor Scenes (Mobile)
- 31 terrain classes, ~94,000 training images, 6,066 test images
- Captured handheld with a mobile phone — mimics a wearable sensor's perspective
- Source: [Xue et al., CVPR 2018](https://github.com/jiaxue-ai/pytorch-material-classification)

**10-class subset used (wearable robotics relevant):**

| Class | Class | Class |
|---|---|---|
| asphalt | grass | sand |
| soil | shale | pebble |
| turf | stone_asphalt | stone_brick |
| stone_cement | | |

---

## Architecture

### Phase 1 — Custom CNN from Scratch

```
Input (3×224×224)
  │
  ├─ ConvBlock 1: Conv(3→32)   → BN → ReLU → Conv(32→32)   → BN → ReLU → MaxPool
  ├─ ConvBlock 2: Conv(32→64)  → BN → ReLU → Conv(64→64)   → BN → ReLU → MaxPool
  ├─ ConvBlock 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool
  ├─ ConvBlock 4: Conv(128→256)→ BN → ReLU → Conv(256→256) → BN → ReLU
  │
  ├─ Global Average Pooling
  └─ FC(256→512) → ReLU → Dropout(0.5) → FC(512→10)
```

- Parameters: ~1.3M
- He initialisation, CosineAnnealingLR scheduler
- Best validation accuracy: **69.45%**

### Phase 2 — MobileNetV2 Transfer Learning

- Pretrained on ImageNet (1.2M images, 1000 classes)
- Custom terrain head: Linear(1280→512) → ReLU → Dropout → Linear(512→10)
- Two-stage training:
  - **Stage A:** Backbone frozen, train head only (LR=1e-3)
  - **Stage B:** Unfreeze last 3 blocks, fine-tune (LR=1e-4)
- Parameters: ~3.4M (0.3M trainable in Stage A)
- Chosen for edge deployment — real-time capable on embedded hardware

---

## Results

| | Phase 1 (CNN Scratch) | Phase 2 (MobileNetV2) |
|---|---|---|
| Val Accuracy | 69.45% | In progress |
| Parameters | 1.3M | 3.4M |
| Real-time capable | Limited | ✅ Yes |
| Training time (T4) | ~30 epochs / 90 min | — |

---

## Project Structure

```
terrain-classification-task/
├── config.py                    # Central config (paths, hyperparams, class list)
├── train_phase1.py              # Phase 1 training script
├── train_phase2.py              # Phase 2 training (Stage A + B)
├── evaluate.py                  # Per-class metrics + confusion matrix
├── inference.py                 # Real-time inference (image/video/webcam)
├── models/
│   ├── cnn_scratch.py           # Custom CNN architecture
│   └── mobilenet_transfer.py    # MobileNetV2 transfer learning
└── utils/
    ├── dataset.py               # DataLoader + subset filtering
    ├── trainer.py               # Training + evaluation engine
    └── plot_results.py          # Training curve plots
```

---

## Setup

```bash
pip install torch torchvision matplotlib pillow
pip install opencv-python    # for video/webcam inference
```

Update `config.py` with your dataset path:
```python
GTOS_ROOT = "path/to/gtos-mobile"
```

---

## Usage

```bash
# Phase 1 — CNN from scratch
python train_phase1.py

# Phase 2 — MobileNetV2 transfer learning
python train_phase2.py

# Evaluate
python evaluate.py --checkpoint checkpoints/phase1_best.pth --phase 1
python evaluate.py --checkpoint checkpoints/phase2_best.pth --phase 2

# Real-time inference
python inference.py --image path/to/terrain.jpg --phase 2
python inference.py --webcam --phase 2
```

---

## Key Design Decisions

**Why MobileNetV2 for wearable robotics?**
Depthwise separable convolutions give ~8× fewer operations than standard convolutions, enabling real-time inference on low-power embedded hardware (Jetson Nano, Raspberry Pi) without sacrificing accuracy.

**Why Global Average Pooling instead of Flatten?**
GAP is input-size agnostic — critical for real-time variable crop inference — and drastically reduces parameters, preventing overfitting.

**Why two-stage fine-tuning?**
Directly fine-tuning all layers at a high LR destroys ImageNet features (catastrophic forgetting). Stage A warms up the new head, Stage B gently adapts deep features at a lower LR.

---

## Environment

- Python 3.10+
- PyTorch 2.x
- Trained on Google Colab (Tesla T4 GPU)

---

*DRDO AI/ML Internship | Terrain Classification for Wearable Robotics*
