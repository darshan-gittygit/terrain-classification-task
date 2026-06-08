# ─────────────────────────────────────────────
#  train_phase1.py  –  Phase 1: CNN from Scratch
# ─────────────────────────────────────────────
#
#  Usage:
#    python train_phase1.py                        # uses GTOS_ROOT from config.py
#    python train_phase1.py --data data/gtos       # override data path
#    python train_phase1.py --epochs 50            # override epochs
# ─────────────────────────────────────────────

import os
import sys
import argparse
import torch

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GTOS_ROOT, NUM_EPOCHS, LR, WEIGHT_DECAY,
    CHECKPOINT_DIR, LOG_DIR
)
from utils.dataset import get_dataloaders
from models.cnn_scratch import get_model
from utils.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 – Terrain CNN from Scratch")
    p.add_argument("--data",    type=str,   default=GTOS_ROOT,   help="Path to dataset root")
    p.add_argument("--epochs",  type=int,   default=NUM_EPOCHS,  help="Number of epochs")
    p.add_argument("--lr",      type=float, default=LR,          help="Learning rate")
    p.add_argument("--batch",   type=int,   default=None,        help="Batch size override")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────
    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else   # Apple Silicon
        "cpu"
    )
    print(f"[Device] Using: {device}")

    # ── Data ──────────────────────────────────────
    if args.batch:
        import config
        config.BATCH_SIZE = args.batch

    train_loader, test_loader, class_names, num_classes = get_dataloaders(args.data)
    print(f"[Classes] {num_classes} classes: {class_names}")

    # ── Model ─────────────────────────────────────
    model = get_model(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] TerrainCNN | Parameters: {total_params:,}")

    # ── Paths ─────────────────────────────────────
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "phase1_best.pth")
    log_path        = os.path.join(LOG_DIR, "phase1_log.csv")

    # ── Train ─────────────────────────────────────
    history = train(
        model          = model,
        train_loader   = train_loader,
        test_loader    = test_loader,
        num_epochs     = args.epochs,
        lr             = args.lr,
        weight_decay   = WEIGHT_DECAY,
        device         = device,
        checkpoint_path= checkpoint_path,
        log_path       = log_path,
    )

    print(f"\n[Done] Checkpoint saved to: {checkpoint_path}")
    print(f"[Done] Training log saved to: {log_path}")
    print("\nNext step → run train_phase2.py for MobileNetV2 transfer learning")


if __name__ == "__main__":
    main()
