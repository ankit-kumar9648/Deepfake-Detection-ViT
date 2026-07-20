"""
src/dataset.py
===============
Dataset loading, preprocessing (STEP 4) and augmentation (STEP 5) pipelines,
plus the DataLoader factory (STEP 6) used by train.py / test.py.

Expects the standard torchvision ImageFolder layout:

    dataset/train/real, dataset/train/fake
    dataset/validation/real, dataset/validation/fake
    dataset/test/real, dataset/test/fake
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def _mean_std() -> Tuple[list, list]:
    """Return the normalisation stats matching the configured backbone."""
    if config.MODEL_BACKEND == "torchvision":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return config.IMAGENET_MEAN, config.IMAGENET_STD


def build_train_transforms() -> transforms.Compose:
    """Resize + augment + normalise pipeline used only on the training split."""
    mean, std = _mean_std()
    aug = config.AUGMENTATION
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
            transforms.RandomRotation(degrees=aug["rotation_degrees"]),
            transforms.RandomCrop(
                config.IMAGE_SIZE, padding=aug["random_crop_padding"]
            ),
            transforms.ColorJitter(**aug["color_jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=aug["random_erasing_p"]),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
    """Deterministic resize + normalise pipeline for validation/test/predict."""
    mean, std = _mean_std()
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _verify_split_populated(split_dir: str, split_name: str) -> None:
    real_dir = os.path.join(split_dir, "real")
    fake_dir = os.path.join(split_dir, "fake")
    n_real = len(os.listdir(real_dir)) if os.path.isdir(real_dir) else 0
    n_fake = len(os.listdir(fake_dir)) if os.path.isdir(fake_dir) else 0
    if n_real == 0 or n_fake == 0:
        raise RuntimeError(
            f"'{split_name}' split is empty (real={n_real}, fake={n_fake}). "
            f"Place images inside '{real_dir}' and '{fake_dir}' before running "
            f"this script."
        )


def get_datasets() -> Dict[str, datasets.ImageFolder]:
    """Build train / validation / test ImageFolder datasets."""
    _verify_split_populated(config.TRAIN_DIR, "train")
    _verify_split_populated(config.VAL_DIR, "validation")
    _verify_split_populated(config.TEST_DIR, "test")

    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=build_train_transforms())
    val_ds = datasets.ImageFolder(config.VAL_DIR, transform=build_eval_transforms())
    test_ds = datasets.ImageFolder(config.TEST_DIR, transform=build_eval_transforms())

    expected = config.CLASS_TO_IDX
    if train_ds.class_to_idx != expected:
        raise RuntimeError(
            f"Unexpected class mapping {train_ds.class_to_idx}, expected {expected}. "
            f"Ensure your folders are named exactly 'real' and 'fake'."
        )
    return {"train": train_ds, "validation": val_ds, "test": test_ds}


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Dict[str, DataLoader]:
    """Return train / validation / test DataLoaders (STEP 6)."""
    ds = get_datasets()
    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            ds["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        ),
        "validation": DataLoader(
            ds["validation"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "test": DataLoader(
            ds["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
    }
    return loaders


def dataset_summary(ds: Dict[str, datasets.ImageFolder]) -> Dict[str, Dict[str, int]]:
    """Per-split, per-class image counts -- used by EDA and console printout."""
    summary = {}
    for split_name, dataset in ds.items():
        counts = {cls: 0 for cls in config.CLASS_NAMES}
        for _, label_idx in dataset.samples:
            counts[config.IDX_TO_CLASS[label_idx]] += 1
        summary[split_name] = counts
    return summary
