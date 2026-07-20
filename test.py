"""
test.py
=======
Evaluate the best trained checkpoint on the held-out test split (STEP 9),
saving accuracy/precision/recall/F1/ROC-AUC, confusion matrix, ROC curve,
PR curve and a full classification report.

Usage
-----
    python test.py
    python test.py --checkpoint models/last_model.pth
"""

from __future__ import annotations

import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import config
from src.dataset import get_dataloaders
from src.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_classification_report,
)
from src.model import build_model
from utils import get_logger, load_checkpoint, save_json

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Deepfake Detection ViT model.")
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    pos_idx = config.CLASS_TO_IDX["real"]

    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(config.DEVICE)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, pos_idx].cpu().tolist())

    return y_true, y_pred, y_prob


def main() -> None:
    args = parse_args()

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch')}")

    loaders = get_dataloaders(batch_size=args.batch_size)
    y_true, y_pred, y_prob = evaluate(model, loaders["test"])

    metrics = compute_metrics(y_true, y_pred, y_prob)
    save_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)
    plot_pr_curve(y_true, y_prob)

    save_json(metrics, config.OUTPUTS_DIR + "/test_metrics.json")

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"{k:>12s}: {v}")
    logger.info("=" * 60)
    logger.info(f"All plots saved under '{config.PLOTS_DIR}'")


if __name__ == "__main__":
    main()
