"""
src/eda.py
==========
Exploratory Data Analysis (STEP 3). Generates and saves:
  - class distribution bar chart + pie chart
  - image resolution distribution
  - random sample grid
  - pixel intensity distribution

Run directly with `python -m src.eda` after the dataset folder is populated.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from utils import get_logger

logger = get_logger(__name__)


def _list_images(split_dir: str) -> Dict[str, List[str]]:
    files = {}
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls)
        if os.path.isdir(cls_dir):
            files[cls] = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ]
        else:
            files[cls] = []
    return files


def print_dataset_overview() -> None:
    """STEP 2: dataset size, class counts, images per class."""
    total = 0
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(config.DATASET_DIR, split)
        files = _list_images(split_dir)
        split_total = sum(len(v) for v in files.values())
        total += split_total
        logger.info(f"[{split.upper()}] total={split_total} | "
                    f"{ {k: len(v) for k, v in files.items()} }")
    logger.info(f"Number of classes: {config.NUM_CLASSES} -> {config.CLASS_NAMES}")
    logger.info(f"Dataset total images (all splits): {total}")


def plot_class_distribution(save_dir: str = config.EDA_REPORT_DIR) -> None:
    """Bar chart + pie chart of class counts across all splits combined."""
    counts = Counter()
    for split in ["train", "validation", "test"]:
        files = _list_images(os.path.join(config.DATASET_DIR, split))
        for cls, flist in files.items():
            counts[cls] += len(flist)

    labels = list(counts.keys())
    values = list(counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, values, color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Class Distribution (Bar Chart)")
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate(values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    axes[1].pie(values, labels=labels, autopct="%1.1f%%",
                colors=["#e74c3c", "#2ecc71"], startangle=90)
    axes[1].set_title("Class Distribution (Pie Chart)")

    fig.tight_layout()
    out_path = os.path.join(save_dir, "class_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved class distribution plot -> {out_path}")


def plot_resolution_distribution(save_dir: str = config.EDA_REPORT_DIR,
                                  sample_per_class: int = 200) -> None:
    """Scatter of (width, height) for a sample of images per class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"real": "#2ecc71", "fake": "#e74c3c"}

    for cls in config.CLASS_NAMES:
        files = _list_images(config.TRAIN_DIR)[cls]
        sample = random.sample(files, min(sample_per_class, len(files))) if files else []
        widths, heights = [], []
        for path in sample:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as exc:
                logger.warning(f"Could not read '{path}': {exc}")
        ax.scatter(widths, heights, alpha=0.5, label=cls, color=colors.get(cls))

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution Distribution (train split sample)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(save_dir, "resolution_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved resolution distribution plot -> {out_path}")


def plot_random_samples(save_dir: str = config.EDA_REPORT_DIR, n_per_class: int = 4) -> None:
    """Grid of randomly sampled real/fake images."""
    files = _list_images(config.TRAIN_DIR)
    n_classes = len(config.CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(3 * n_per_class, 3 * n_classes))
    if n_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, cls in enumerate(config.CLASS_NAMES):
        sample = random.sample(files[cls], min(n_per_class, len(files[cls]))) if files[cls] else []
        for col in range(n_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(sample):
                try:
                    with Image.open(sample[col]) as img:
                        ax.imshow(img.convert("RGB"))
                except Exception as exc:
                    logger.warning(f"Could not read '{sample[col]}': {exc}")
            if col == 0:
                ax.set_ylabel(cls)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
        axes[row][0].set_title(f"class: {cls}", loc="left")

    fig.suptitle("Random Sample Images (train split)")
    fig.tight_layout()
    out_path = os.path.join(save_dir, "random_samples.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved random sample grid -> {out_path}")


def plot_pixel_distribution(save_dir: str = config.EDA_REPORT_DIR, sample_per_class: int = 100) -> None:
    """Histogram of per-channel pixel intensities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"real": "#2ecc71", "fake": "#e74c3c"}

    for cls in config.CLASS_NAMES:
        files = _list_images(config.TRAIN_DIR)[cls]
        sample = random.sample(files, min(sample_per_class, len(files))) if files else []
        pixels = []
        for path in sample:
            try:
                with Image.open(path) as img:
                    arr = np.asarray(img.convert("L"), dtype=np.uint8)
                    pixels.append(arr.flatten())
            except Exception as exc:
                logger.warning(f"Could not read '{path}': {exc}")
        if pixels:
            all_pixels = np.concatenate(pixels)
            ax.hist(all_pixels, bins=50, alpha=0.5, label=cls,
                    color=colors.get(cls), density=True)

    ax.set_xlabel("Pixel Intensity (grayscale)")
    ax.set_ylabel("Density")
    ax.set_title("Pixel Intensity Distribution (train split sample)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(save_dir, "pixel_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved pixel distribution plot -> {out_path}")


def run_full_eda() -> None:
    """Convenience entry point running every EDA step and saving all graphs."""
    logger.info("Running full EDA ...")
    print_dataset_overview()
    plot_class_distribution()
    plot_resolution_distribution()
    plot_random_samples()
    plot_pixel_distribution()
    logger.info(f"EDA complete. Graphs saved under '{config.EDA_REPORT_DIR}'.")


if __name__ == "__main__":
    run_full_eda()
