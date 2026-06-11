# ─────────────────────────────────────────────
#  merge_datasets.py
#  Merges RUGD (soldier categories) + GTOS-Mobile into one unified dataset
#  Output: soldier_dataset/train/<category>/ and soldier_dataset/test/<category>/
# ─────────────────────────────────────────────

import os
import shutil
import random
from pathlib import Path

# ── Paths — update these ──────────────────────
RUGD_CLASSIFIED  = r"C:\Users\Darshan\Downloads\rugd_classified"
GTOS_MOBILE_ROOT = r"C:\Users\Darshan\Downloads\gtos-mobile\gtos-mobile"
OUTPUT_DIR       = r"C:\Users\Darshan\Downloads\soldier_dataset"

TEST_RATIO  = 0.15
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── GTOS-Mobile → Soldier category mapping ────
# Which GTOS classes contribute to which soldier category
GTOS_TO_SOLDIER = {
    # desert
    "sand":          "desert",
    "shale":         "desert",
    # forest
    "grass":         "forest",
    "moss":          "forest",
    "soil":          "forest",
    "root":          "forest",
    "leaf":          "forest",
    "dry_leaf":      "forest",
    # normal
    "asphalt":       "normal",
    "cement":        "normal",
    "stone_asphalt": "normal",
    "stone_brick":   "normal",
    "stone_cement":  "normal",
    # mud — soil does double duty
    # (we'll copy soil to mud too)
    # rough
    "pebble":        "rough",
    "large_limestone":"rough",
    "small_limestone":"rough",
    "brick":         "rough",
}

# GTOS classes that map to multiple soldier categories
GTOS_MULTI = {
    "soil": ["forest", "mud"],   # soil goes to both forest and mud
    "shale": ["desert", "rough"], # shale goes to both desert and rough
}


def copy_images(src_dir, dst_dir, split, max_images=None):
    """
    Copy images from src_dir into dst_dir.
    If split='train', copies directly.
    If split='test', copies directly.
    Optionally cap at max_images to avoid over-dominance.
    Returns number of images copied.
    """
    os.makedirs(dst_dir, exist_ok=True)

    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if max_images and len(images) > max_images:
        images = random.sample(images, max_images)

    for img in images:
        src = os.path.join(src_dir, img)
        # prefix with source to avoid filename collisions
        prefix = os.path.basename(src_dir)
        dst = os.path.join(dst_dir, f"{prefix}_{img}")
        shutil.copy2(src, dst)

    return len(images)


def copy_gtos_class(gtos_class, soldier_cat, split, max_images=None):
    """Copy a GTOS-Mobile class into the soldier dataset."""
    src = os.path.join(GTOS_MOBILE_ROOT, split, gtos_class)
    dst = os.path.join(OUTPUT_DIR, split, soldier_cat)

    if not os.path.exists(src):
        print(f"    [SKIP] GTOS {gtos_class} not found at {src}")
        return 0

    count = copy_images(src, dst, split, max_images)
    return count


def main():
    soldier_cats = ["desert", "forest", "normal", "mud", "rough", "water"]

    # create output dirs
    for split in ["train", "test"]:
        for cat in soldier_cats:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cat), exist_ok=True)

    print("=" * 60)
    print("  STEP 1 — Copy RUGD classified patches")
    print("=" * 60)

    rugd_counts = {"train": {}, "test": {}}
    for split in ["train", "test"]:
        for cat in soldier_cats:
            src = os.path.join(RUGD_CLASSIFIED, split, cat)
            dst = os.path.join(OUTPUT_DIR, split, cat)
            if not os.path.exists(src):
                rugd_counts[split][cat] = 0
                continue
            count = copy_images(src, dst, split)
            rugd_counts[split][cat] = count
            print(f"  [{split}] {cat:<15} {count:>5} from RUGD")

    print()
    print("=" * 60)
    print("  STEP 2 — Copy GTOS-Mobile classes")
    print("=" * 60)

    gtos_counts = {"train": {c: 0 for c in soldier_cats},
                   "test":  {c: 0 for c in soldier_cats}}

    for split in ["train", "test"]:
        print(f"\n  {split}/")
        for gtos_class, soldier_cat in GTOS_TO_SOLDIER.items():
            count = copy_gtos_class(gtos_class, soldier_cat, split)
            gtos_counts[split][soldier_cat] += count
            if count > 0:
                print(f"    {gtos_class:<20} → {soldier_cat:<10} {count:>5} images")

        # handle multi-category GTOS classes
        for gtos_class, cats in GTOS_MULTI.items():
            for soldier_cat in cats:
                count = copy_gtos_class(gtos_class, soldier_cat, split)
                gtos_counts[split][soldier_cat] += count
                if count > 0:
                    print(f"    {gtos_class:<20} → {soldier_cat:<10} {count:>5} images (shared)")

    print()
    print("=" * 60)
    print("  FINAL DATASET SUMMARY")
    print("=" * 60)

    print(f"\n  {'Category':<15} {'Train':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'─'*45}")

    grand_total = 0
    for cat in soldier_cats:
        tr = len(os.listdir(os.path.join(OUTPUT_DIR, "train", cat)))
        te = len(os.listdir(os.path.join(OUTPUT_DIR, "test",  cat)))
        grand_total += tr + te
        print(f"  {cat:<15} {tr:>8} {te:>8} {tr+te:>8}")

    print(f"  {'─'*45}")
    print(f"  {'TOTAL':<15} {grand_total:>25}")
    print(f"\n  Saved to: {OUTPUT_DIR}")
    print("\n  Next step → upload soldier_dataset to Google Drive")
    print("  and update config.py to point to this dataset")


if __name__ == "__main__":
    main()
