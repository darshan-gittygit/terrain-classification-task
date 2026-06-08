# ─────────────────────────────────────────────
#  utils/dataset.py  –  DataLoader builder
# ─────────────────────────────────────────────

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import (
    IMG_SIZE, MEAN, STD, BATCH_SIZE, NUM_WORKERS,
    SUBSET_CLASSES, USE_SUBSET
)


def get_transforms(train: bool) -> transforms.Compose:
    """
    Training: augmentation + normalize
    Val/Test: just resize + normalize (no augmentation — keeps eval fair)
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),  # slightly larger
            transforms.RandomCrop(IMG_SIZE),                     # then random crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def filter_subset(dataset: datasets.ImageFolder, allowed_classes: list):
    """
    Filter an ImageFolder dataset to only include samples from allowed_classes.
    Remaps labels to be contiguous 0..N-1 for the selected classes.
    Returns the modified dataset (in-place) and the list of selected class names.
    """
    allowed_set = set(allowed_classes)
    # only keep classes that actually exist in the folder
    valid_classes = sorted([c for c in dataset.classes if c in allowed_set])

    if not valid_classes:
        raise ValueError(
            f"None of the requested classes found in dataset.\n"
            f"Requested: {allowed_classes}\n"
            f"Found in dataset: {dataset.classes}"
        )

    # old label index → new contiguous index
    old_to_new = {dataset.class_to_idx[c]: new_idx for new_idx, c in enumerate(valid_classes)}
    valid_old_idx = set(old_to_new.keys())

    # filter samples and remap labels in-place
    dataset.samples = [
        (path, old_to_new[label])
        for path, label in dataset.samples
        if label in valid_old_idx
    ]
    dataset.targets = [label for _, label in dataset.samples]
    dataset.classes = valid_classes
    dataset.class_to_idx = {c: i for i, c in enumerate(valid_classes)}

    return dataset, valid_classes


def get_dataloaders(data_root: str):
    """
    Build train and test DataLoaders from data_root.
    Expects:  data_root/train/<class>/  and  data_root/test/<class>/

    Returns: train_loader, test_loader, class_names, num_classes
    """
    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    test_dataset  = datasets.ImageFolder(test_dir,  transform=get_transforms(train=False))

    if USE_SUBSET:
        train_dataset, class_names = filter_subset(train_dataset, SUBSET_CLASSES)
        test_dataset,  _           = filter_subset(test_dataset,  SUBSET_CLASSES)
        print(f"[Dataset] Using subset: {len(class_names)} classes → {class_names}")
    else:
        class_names = train_dataset.classes
        print(f"[Dataset] Using all {len(class_names)} classes")

    # sanity check
    assert len(train_dataset) > 0, "Train dataset is empty — check your dataset path and class names"
    assert len(test_dataset)  > 0, "Test dataset is empty — check your dataset path and class names"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[Dataset] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_loader, test_loader, class_names, len(class_names)
