# ─────────────────────────────────────────────
#  config.py  –  Central config for all experiments
# ─────────────────────────────────────────────

import os

# ── Dataset paths ──────────────────────────────
# Update these to match where you extracted the dataset
GTOS_ROOT = "data/gtos"          # path to GTOS  (train/ test/ inside)
GTOS_MOBILE_ROOT = "data/gtos_mobile"  # path to GTOS-Mobile (train/ test/ inside)

# ── Class selection ────────────────────────────
# Phase 1: focused subset (10 terrain-relevant classes for wearable robotics)
SUBSET_CLASSES = [
    "asphalt",
    "grass",
    "gravel",        # or pebble – whichever folder name your dataset uses
    "sand",
    "soil",
    "stone_asphalt",
    "stone_brick",
    "stone_cement",
    "pebble",
    "turf",
]

# Phase 2 (later): set USE_SUBSET = False to train on all 40 classes
USE_SUBSET = True

# ── Image settings ─────────────────────────────
IMG_SIZE    = 224          # resize to 224×224 (standard for CNNs)
MEAN        = [0.485, 0.456, 0.406]   # ImageNet mean (reuse for transfer learning)
STD         = [0.229, 0.224, 0.225]   # ImageNet std

# ── Training hyperparameters ───────────────────
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4           # dataloader workers; set to 0 on Windows if errors

# ── Paths ──────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
