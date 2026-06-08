# ─────────────────────────────────────────────
#  utils/plot_results.py  –  Training curve visualisation
# ─────────────────────────────────────────────
#
#  Usage:
#    python utils/plot_results.py               # plots both phases side by side
#    python utils/plot_results.py --phase 1     # only phase 1
# ─────────────────────────────────────────────

import os
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_phase(rows, phase_label, axes):
    """Plot loss and accuracy curves for a single phase."""
    # For Phase 2 the CSV has a 'stage' column — merge epochs sequentially
    epochs      = list(range(1, len(rows) + 1))
    train_loss  = [float(r["train_loss"]) for r in rows]
    val_loss    = [float(r["val_loss"])   for r in rows]
    train_acc   = [float(r["train_acc"])  for r in rows]
    val_acc     = [float(r["val_acc"])    for r in rows]

    ax_loss, ax_acc = axes

    ax_loss.plot(epochs, train_loss, label="Train Loss", color="#4C72B0", linewidth=2)
    ax_loss.plot(epochs, val_loss,   label="Val Loss",   color="#DD8452", linewidth=2, linestyle="--")
    ax_loss.set_title(f"{phase_label} – Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, train_acc, label="Train Acc", color="#4C72B0", linewidth=2)
    ax_acc.plot(epochs, val_acc,   label="Val Acc",   color="#DD8452", linewidth=2, linestyle="--")
    ax_acc.set_title(f"{phase_label} – Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    best_val = max(val_acc)
    ax_acc.axhline(best_val, color="green", linestyle=":", alpha=0.6,
                   label=f"Best: {best_val:.2f}%")
    ax_acc.legend()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                   help="0=both, 1=phase1 only, 2=phase2 only")
    p.add_argument("--log_dir", type=str, default="logs")
    return p.parse_args()


def main():
    args  = parse_args()
    p1log = os.path.join(args.log_dir, "phase1_log.csv")
    p2log = os.path.join(args.log_dir, "phase2_log.csv")

    phases_to_plot = []
    if args.phase in (0, 1) and os.path.exists(p1log):
        phases_to_plot.append(("Phase 1 – CNN from Scratch", load_csv(p1log)))
    if args.phase in (0, 2) and os.path.exists(p2log):
        phases_to_plot.append(("Phase 2 – MobileNetV2 Transfer", load_csv(p2log)))

    if not phases_to_plot:
        print("No log files found. Train first with train_phase1.py or train_phase2.py")
        return

    n     = len(phases_to_plot)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for i, (label, rows) in enumerate(phases_to_plot):
        plot_phase(rows, label, axes[i])

    fig.suptitle("Terrain Classification – Training Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(args.log_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {out_path}")


if __name__ == "__main__":
    main()
