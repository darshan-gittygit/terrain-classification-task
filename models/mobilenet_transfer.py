# ─────────────────────────────────────────────
#  models/mobilenet_transfer.py  –  Phase 2: Transfer Learning
# ─────────────────────────────────────────────
#
#  Why MobileNetV2 for wearable robotics?
#  • Designed for edge/mobile deployment → low latency, low power
#  • Depthwise separable convolutions → ~8-9x fewer ops than standard conv
#  • ~3.4M parameters vs ~11M for ResNet-18
#  • Hits real-time speeds (~30+ FPS) even on embedded hardware (Jetson Nano, RPi)
#  • Used in the original GTOS paper and achieves 80-84% accuracy
#
#  Two training modes:
#  1. FEATURE EXTRACTION: freeze all backbone layers, only train classifier head
#     → fast, good when your dataset is small
#  2. FINE-TUNING: unfreeze last few blocks + classifier, train everything
#     → better accuracy, needs more data/time
#
#  We do feature extraction first, then fine-tune (standard best practice)
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Terrain(nn.Module):
    """
    MobileNetV2 fine-tuned for terrain classification.

    Args:
        num_classes   : number of terrain classes
        mode          : 'feature_extract' or 'finetune'
        pretrained    : use ImageNet pretrained weights
    """
    def __init__(
        self,
        num_classes : int  = 10,
        mode        : str  = "feature_extract",
        pretrained  : bool = True,
    ):
        super().__init__()

        # Load pretrained MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # ── Freeze / unfreeze layers ───────────────
        if mode == "feature_extract":
            # Freeze entire backbone → only train the new classifier head
            for param in backbone.parameters():
                param.requires_grad = False
            print("[MobileNetV2] Mode: feature_extract (backbone frozen)")

        elif mode == "finetune":
            # Unfreeze last 3 InvertedResidual blocks (features[15:]) + classifier
            # Keep early layers frozen → they have generic low-level features
            for param in backbone.parameters():
                param.requires_grad = False
            for param in backbone.features[15:].parameters():
                param.requires_grad = True
            print("[MobileNetV2] Mode: finetune (last 3 blocks + head unfrozen)")

        else:
            raise ValueError(f"mode must be 'feature_extract' or 'finetune', got '{mode}'")

        # ── Replace classifier head ────────────────
        # Original head: Linear(1280 → 1000) for ImageNet
        # New head: Linear(1280 → 512) → ReLU → Dropout → Linear(512 → num_classes)
        in_features = backbone.classifier[1].in_features   # 1280
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def get_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total


def get_model(num_classes: int, mode: str = "feature_extract") -> MobileNetV2Terrain:
    return MobileNetV2Terrain(num_classes=num_classes, mode=mode)


# ── Quick sanity check ──────────────────────────
if __name__ == "__main__":
    for mode in ["feature_extract", "finetune"]:
        model = MobileNetV2Terrain(num_classes=10, mode=mode)
        dummy = torch.randn(4, 3, 224, 224)
        out   = model(dummy)
        t, total = model.get_trainable_params()
        print(f"[{mode}] Output: {out.shape} | Trainable: {t:,} / {total:,}")
