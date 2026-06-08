# ─────────────────────────────────────────────
#  models/cnn_scratch.py  –  Custom CNN (Phase 1)
# ─────────────────────────────────────────────
#
#  Architecture overview:
#
#  Input (3×224×224)
#    │
#    ├─ Block 1: Conv(3→32)   → BN → ReLU → Conv(32→32)  → BN → ReLU → MaxPool → Dropout
#    ├─ Block 2: Conv(32→64)  → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool → Dropout
#    ├─ Block 3: Conv(64→128) → BN → ReLU → Conv(128→128)→ BN → ReLU → MaxPool → Dropout
#    ├─ Block 4: Conv(128→256)→ BN → ReLU → Conv(256→256)→ BN → ReLU → AdaptiveAvgPool
#    │
#    └─ Classifier: FC(256→512) → ReLU → Dropout → FC(512→num_classes)
#
#  Design choices (mention these to DRDO mentor):
#  • Double conv per block (like VGG) → richer feature extraction per spatial level
#  • BatchNorm after every conv → stable training, acts as regularizer
#  • AdaptiveAvgPool at the end → input-size agnostic (important for real-time variable crops)
#  • Dropout(0.5) before final FC → prevents overfitting on terrain textures
# ─────────────────────────────────────────────

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Double conv block: Conv → BN → ReLU → Conv → BN → ReLU → Pool → Dropout
    """
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True, dropout: float = 0.25):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))   # halves spatial dims
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TerrainCNN(nn.Module):
    """
    Custom CNN from scratch for terrain classification.

    Args:
        num_classes: number of terrain classes (10 for subset, 40 for full)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True,  dropout=0.25),   # 224 → 112
            ConvBlock(32,  64,  pool=True,  dropout=0.25),   # 112 → 56
            ConvBlock(64,  128, pool=True,  dropout=0.25),   # 56  → 28
            ConvBlock(128, 256, pool=False, dropout=0.0),    # 28  → 28 (no pool before GAP)
        )

        # Global Average Pooling: collapses spatial dims to 1×1
        # This is key for real-time use — no fixed FC size tied to input resolution
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # Weight initialization (He init for ReLU networks)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def get_model(num_classes: int) -> TerrainCNN:
    return TerrainCNN(num_classes=num_classes)


# ── Quick sanity check ──────────────────────────
if __name__ == "__main__":
    model = TerrainCNN(num_classes=10)
    dummy = torch.randn(4, 3, 224, 224)   # batch of 4
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # should be [4, 10]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
