# ─────────────────────────────────────────────
#  rugd_to_classification.py
#  Converts RUGD segmentation dataset → classification folder structure
#  that matches GTOS-Mobile and works with PyTorch ImageFolder
# ─────────────────────────────────────────────
#
#  How it works:
#  1. For each frame + its annotation mask
#  2. Find pixels belonging to our target classes
#  3. If a class covers enough of the image (MIN_COVERAGE), crop a patch
#  4. Save patch to output/train/<class>/ or output/test/<class>/
#
#  Output structure:
#  rugd_classified/
#    train/
#      dirt/   grass/   sand/   asphalt/   gravel/ ...
#    test/
#      dirt/   grass/   sand/   asphalt/   gravel/ ...
# ─────────────────────────────────────────────

import os
import numpy as np
from PIL import Image
from pathlib import Path
import random

# ── Paths — update these ──────────────────────
FRAMES_DIR      = r"C:\Users\Darshan\Downloads\RUGD_frames-with-annotations\RUGD_frames-with-annotations"
ANNOTATIONS_DIR = r"C:\Users\Darshan\Downloads\RUGD_annotations\RUGD_annotations"
OUTPUT_DIR      = r"C:\Users\Darshan\Downloads\rugd_classified"

# ── RUGD colormap ─────────────────────────────
# class_name → (R, G, B)
COLORMAP = {
    "dirt":     (108, 64,  20),
    "sand":     (255, 229, 204),
    "grass":    (0,   102, 0),
    "tree":     (0,   255, 0),
    "asphalt":  (64,  64,  64),
    "gravel":   (255, 128, 0),
    "mulch":    (153, 76,  0),
    "rock-bed": (102, 102, 0),
    "log":      (102, 0,   0),
    "bush":     (255, 153, 204),
    "rock":     (153, 204, 255),
    "concrete": (101, 101, 11),
    "water":    (0,   128, 255),
}

# ── Soldier category mapping ──────────────────
# Maps RUGD classes → soldier-relevant categories
SOLDIER_CATEGORIES = {
    "desert":  ["sand", "dirt"],
    "forest":  ["grass", "tree", "mulch", "bush"],
    "normal":  ["asphalt", "gravel", "concrete"],
    "mud":     ["dirt", "mulch"],
    "rough":   ["rock", "rock-bed", "log"],
    "water":   ["water"],
}

# Build reverse map: rugd_class → LIST of soldier categories
# dirt → [desert, mud], mulch → [forest, mud] etc.
RUGD_TO_SOLDIER = {}
for category, classes in SOLDIER_CATEGORIES.items():
    for cls in classes:
        if cls not in RUGD_TO_SOLDIER:
            RUGD_TO_SOLDIER[cls] = []
        RUGD_TO_SOLDIER[cls].append(category)

# ── Settings ───────────────────────────────────
PATCH_SIZE   = 224      # output patch size (matches model input)
MIN_COVERAGE = 0.10     # class must cover at least 10% of patch area
TEST_RATIO   = 0.15     # 15% of images go to test set
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)


def get_class_mask(annotation_np, color):
    """Return binary mask where annotation pixels match the given RGB color."""
    r, g, b = color
    mask = (
        (annotation_np[:, :, 0] == r) &
        (annotation_np[:, :, 1] == g) &
        (annotation_np[:, :, 2] == b)
    )
    return mask


def extract_dominant_patch(frame_np, annotation_np, class_name, color):
    """
    Find the largest connected region of this class in the annotation,
    crop a centered patch, check coverage, return patch or None.
    """
    mask = get_class_mask(annotation_np, color)
    total_pixels = mask.sum()

    if total_pixels < (PATCH_SIZE * PATCH_SIZE * MIN_COVERAGE):
        return None  # not enough pixels of this class

    # find bounding box of all class pixels
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    # center of the class region
    center_r = int((rows[0] + rows[-1]) / 2)
    center_c = int((cols[0] + cols[-1]) / 2)

    h, w = frame_np.shape[:2]
    half  = PATCH_SIZE // 2

    # clamp to image bounds
    r1 = max(0, center_r - half)
    r2 = r1 + PATCH_SIZE
    if r2 > h:
        r2 = h
        r1 = max(0, r2 - PATCH_SIZE)

    c1 = max(0, center_c - half)
    c2 = c1 + PATCH_SIZE
    if c2 > w:
        c2 = w
        c1 = max(0, c2 - PATCH_SIZE)

    patch_frame = frame_np[r1:r2, c1:c2]
    patch_mask  = mask[r1:r2, c1:c2]

    # check coverage in the actual patch
    coverage = patch_mask.sum() / (patch_mask.size + 1e-6)
    if coverage < MIN_COVERAGE:
        return None

    # resize to exact PATCH_SIZE if needed
    patch_img = Image.fromarray(patch_frame)
    patch_img = patch_img.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
    return patch_img


def main():
    # create output dirs
    soldier_classes = list(SOLDIER_CATEGORIES.keys())
    for split in ["train", "test"]:
        for cls in soldier_classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    # get all scenes
    scenes = sorted([
        d for d in os.listdir(ANNOTATIONS_DIR)
        if os.path.isdir(os.path.join(ANNOTATIONS_DIR, d))
    ])

    print(f"Found {len(scenes)} scenes: {scenes}")
    print(f"Target soldier classes: {soldier_classes}")
    print(f"Processing...\n")

    total_saved = {cls: 0 for cls in soldier_classes}
    patch_counter = 0

    for scene in scenes:
        ann_scene_dir   = os.path.join(ANNOTATIONS_DIR, scene)
        frame_scene_dir = os.path.join(FRAMES_DIR, scene)

        if not os.path.exists(frame_scene_dir):
            print(f"  [SKIP] No frames for scene: {scene}")
            continue

        ann_files = sorted([
            f for f in os.listdir(ann_scene_dir)
            if f.endswith('.png')
        ])

        for ann_file in ann_files:
            ann_path   = os.path.join(ann_scene_dir, ann_file)
            frame_path = os.path.join(frame_scene_dir, ann_file)

            if not os.path.exists(frame_path):
                continue

            try:
                annotation = np.array(Image.open(ann_path).convert("RGB"))
                frame      = np.array(Image.open(frame_path).convert("RGB"))
            except Exception as e:
                print(f"  [ERROR] {ann_file}: {e}")
                continue

            # decide train or test
            split = "test" if random.random() < TEST_RATIO else "train"

            # try to extract a patch for each RUGD class
            for rugd_class, color in COLORMAP.items():
                soldier_cats = RUGD_TO_SOLDIER.get(rugd_class, [])
                if not soldier_cats:
                    continue

                patch = extract_dominant_patch(frame, annotation, rugd_class, color)

                if patch is not None:
                    for soldier_cat in soldier_cats:
                        fname = f"{scene}_{ann_file.replace('.png','')}_{rugd_class}_{soldier_cat}_{patch_counter:05d}.jpg"
                        save_path = os.path.join(OUTPUT_DIR, split, soldier_cat, fname)
                        patch.save(save_path, quality=90)
                        total_saved[soldier_cat] += 1
                    patch_counter += 1

        print(f"  [Done] {scene}")

    print(f"\n{'─'*50}")
    print(f"Extraction complete!")
    print(f"\nPatches per soldier category:")
    for cls, count in total_saved.items():
        print(f"  {cls:<15} {count:>5} patches")
    print(f"\nTotal: {patch_counter} patches")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
