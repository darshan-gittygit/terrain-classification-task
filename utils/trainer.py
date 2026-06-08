# ─────────────────────────────────────────────
#  utils/trainer.py  –  Training + Evaluation engine
# ─────────────────────────────────────────────

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += images.size(0)

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / total,
        "acc":  100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += images.size(0)

    return {
        "loss": total_loss / total,
        "acc":  100.0 * correct / total,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: str,
    log_path: str,
) -> dict:
    """
    Full training loop with:
    - CrossEntropyLoss
    - Adam optimizer
    - CosineAnnealingLR scheduler (smooth LR decay, good for CNNs)
    - Best model checkpointing
    - CSV logging
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc   = 0.0
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # CSV log header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    print(f"\n{'─'*65}")
    print(f"  {'Epoch':>6} │ {'Train Loss':>10} │ {'Train Acc':>9} │ {'Val Loss':>8} │ {'Val Acc':>8}")
    print(f"{'─'*65}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_metrics   = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f"  {epoch:>6} │ {train_metrics['loss']:>10.4f} │ "
            f"{train_metrics['acc']:>8.2f}% │ {val_metrics['loss']:>8.4f} │ "
            f"{val_metrics['acc']:>7.2f}%  ({elapsed:.1f}s)"
        )

        # log to CSV
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_metrics['loss']:.4f},{train_metrics['acc']:.2f},"
                f"{val_metrics['loss']:.4f},{val_metrics['acc']:.2f},{current_lr:.6f}\n"
            )

        # save best model
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ New best saved → {best_acc:.2f}%")

    print(f"{'─'*65}")
    print(f"  Training complete. Best val accuracy: {best_acc:.2f}%")
    return history
