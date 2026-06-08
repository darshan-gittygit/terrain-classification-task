# ─────────────────────────────────────────────
#  evaluate.py  –  Full evaluation with per-class metrics
# ─────────────────────────────────────────────
#
#  Outputs:
#  • Overall accuracy
#  • Per-class precision, recall, F1
#  • Confusion matrix (saved as PNG)
#  • Top-5 confused class pairs
#
#  Usage:
#    python evaluate.py --checkpoint checkpoints/phase1_best.pth --phase 1
#    python evaluate.py --checkpoint checkpoints/phase2_best.pth --phase 2
# ─────────────────────────────────────────────

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without display)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GTOS_ROOT
from utils.dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--phase",      type=int, default=1, choices=[1, 2])
    p.add_argument("--data",       type=str, default=GTOS_ROOT)
    return p.parse_args()


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix saved → {save_path}")


def compute_per_class_metrics(preds, labels, num_classes):
    metrics = {}
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[c] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    _, test_loader, class_names, num_classes = get_dataloaders(args.data)

    # load model
    if args.phase == 1:
        from models.cnn_scratch import get_model
        model = get_model(num_classes=num_classes).to(device)
    else:
        from models.mobilenet_transfer import get_model
        # use finetune mode so all layers are present when loading fine-tuned checkpoint
        model = get_model(num_classes=num_classes, mode="finetune").to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}  (val_acc={ckpt['val_acc']:.2f}%)")

    preds, labels = get_predictions(model, test_loader, device)

    # ── Overall accuracy ───────────────────────────
    overall_acc = 100.0 * (preds == labels).mean()
    print(f"\n{'─'*50}")
    print(f"  Overall Accuracy: {overall_acc:.2f}%")
    print(f"{'─'*50}")

    # ── Per-class metrics ──────────────────────────
    metrics = compute_per_class_metrics(preds, labels, num_classes)
    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print(f"  {'─'*50}")
    for c, name in enumerate(class_names):
        m = metrics[c]
        print(f"  {name:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}")

    mean_f1 = np.mean([m["f1"] for m in metrics.values()])
    print(f"\n  Mean F1: {mean_f1:.3f}")

    # ── Confusion matrix ───────────────────────────
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    save_path = f"logs/confusion_matrix_phase{args.phase}.png"
    plot_confusion_matrix(cm, class_names, save_path)

    # ── Top confused pairs ─────────────────────────
    print(f"\n  Top confused class pairs:")
    off_diag = [(cm[i][j], class_names[i], class_names[j])
                for i in range(num_classes)
                for j in range(num_classes) if i != j]
    off_diag.sort(reverse=True)
    for count, true_cls, pred_cls in off_diag[:5]:
        print(f"    True='{true_cls}' predicted as '{pred_cls}' → {count} times")


if __name__ == "__main__":
    main()
