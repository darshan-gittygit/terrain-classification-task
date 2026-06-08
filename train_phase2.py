# ─────────────────────────────────────────────
#  train_phase2.py  –  Phase 2: Transfer Learning (MobileNetV2)
# ─────────────────────────────────────────────
#
#  Two-stage training strategy:
#
#  Stage A — Feature Extraction (epochs 1-10):
#    Freeze backbone, train only the new classifier head.
#    Use higher LR (1e-3) since only the head is being trained.
#    → Quickly adapts the classifier to terrain classes.
#
#  Stage B — Fine-tuning (epochs 11-30):
#    Unfreeze last 3 MobileNet blocks + head.
#    Use much lower LR (1e-4) to avoid destroying pretrained features.
#    → Gently adapts deep features to terrain textures.
#
#  This two-stage approach consistently outperforms training everything
#  from scratch or fine-tuning everything at once.
#
#  Usage:
#    python train_phase2.py
#    python train_phase2.py --data data/gtos --stage_a_epochs 15 --stage_b_epochs 25
# ─────────────────────────────────────────────

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GTOS_ROOT, WEIGHT_DECAY,
    CHECKPOINT_DIR, LOG_DIR
)
from utils.dataset import get_dataloaders
from models.mobilenet_transfer import MobileNetV2Terrain
from utils.trainer import train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 – MobileNetV2 Transfer Learning")
    p.add_argument("--data",           type=str,   default=GTOS_ROOT)
    p.add_argument("--stage_a_epochs", type=int,   default=10,  help="Feature extraction epochs")
    p.add_argument("--stage_b_epochs", type=int,   default=20,  help="Fine-tuning epochs")
    p.add_argument("--lr_a",           type=float, default=1e-3, help="LR for stage A")
    p.add_argument("--lr_b",           type=float, default=1e-4, help="LR for stage B")
    return p.parse_args()


def run_stage(
    model, train_loader, test_loader,
    num_epochs, lr, weight_decay,
    device, stage_name,
    checkpoint_path, log_path, best_acc=0.0
):
    """Run one training stage and return history + best_acc."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'═'*65}")
    print(f"  {stage_name}")
    print(f"  LR={lr} | Epochs={num_epochs}")
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), \
                       sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")
    print(f"{'═'*65}")
    print(f"  {'Epoch':>6} │ {'Train Loss':>10} │ {'Train Acc':>9} │ {'Val Loss':>8} │ {'Val Acc':>8}")
    print(f"{'─'*65}")

    for epoch in range(1, num_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_m   = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["acc"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["acc"])

        print(
            f"  {epoch:>6} │ {train_m['loss']:>10.4f} │ "
            f"{train_m['acc']:>8.2f}% │ {val_m['loss']:>8.4f} │ "
            f"{val_m['acc']:>7.2f}%"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{stage_name},{epoch},{train_m['loss']:.4f},{train_m['acc']:.2f},"
                f"{val_m['loss']:.4f},{val_m['acc']:.2f}\n"
            )

        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            torch.save(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ New best → {best_acc:.2f}%")

    return history, best_acc


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[Device] Using: {device}")

    train_loader, test_loader, class_names, num_classes = get_dataloaders(args.data)
    print(f"[Classes] {num_classes} classes")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "phase2_best.pth")
    log_path        = os.path.join(LOG_DIR, "phase2_log.csv")

    with open(log_path, "w") as f:
        f.write("stage,epoch,train_loss,train_acc,val_loss,val_acc\n")

    # ── Stage A: Feature Extraction ───────────────
    model = MobileNetV2Terrain(num_classes=num_classes, mode="feature_extract").to(device)

    _, best_acc = run_stage(
        model, train_loader, test_loader,
        num_epochs      = args.stage_a_epochs,
        lr              = args.lr_a,
        weight_decay    = WEIGHT_DECAY,
        device          = device,
        stage_name      = "Stage A - Feature Extraction",
        checkpoint_path = checkpoint_path,
        log_path        = log_path,
        best_acc        = 0.0,
    )

    # ── Stage B: Fine-tuning ──────────────────────
    # Unfreeze last 3 blocks by rebuilding in finetune mode,
    # then load Stage A weights so we continue from where we left off
    print("\n[Phase2] Switching to finetune mode...")
    finetune_model = MobileNetV2Terrain(num_classes=num_classes, mode="finetune").to(device)

    # load Stage A weights into finetune model
    ckpt = torch.load(checkpoint_path, map_location=device)
    finetune_model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Phase2] Loaded Stage A best weights (val_acc={ckpt['val_acc']:.2f}%)")

    _, best_acc = run_stage(
        finetune_model, train_loader, test_loader,
        num_epochs      = args.stage_b_epochs,
        lr              = args.lr_b,
        weight_decay    = WEIGHT_DECAY,
        device          = device,
        stage_name      = "Stage B - Fine-tuning",
        checkpoint_path = checkpoint_path,
        log_path        = log_path,
        best_acc        = best_acc,
    )

    print(f"\n[Done] Best overall val accuracy: {best_acc:.2f}%")
    print(f"[Done] Checkpoint: {checkpoint_path}")
    print(f"[Done] Log: {log_path}")
    print("\nNext step → run evaluate.py to see per-class metrics & confusion matrix")


if __name__ == "__main__":
    main()
