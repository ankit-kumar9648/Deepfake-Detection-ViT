"""
src/metrics.py
===============
Evaluation utilities (STEP 9): accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, classification report, ROC curve and Precision-Recall
curve -- all saved as figures / JSON under outputs/.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from utils import get_logger, save_json

logger = get_logger(__name__)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict:
    """Compute the full evaluation-metric suite for the positive class 'real'."""
    pos_label = config.CLASS_TO_IDX["real"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError as exc:
        logger.warning(f"ROC-AUC could not be computed: {exc}")
        metrics["roc_auc"] = None

    logger.info(f"Computed metrics: {metrics}")
    return metrics


def save_classification_report(y_true, y_pred, path: str = config.CLASSIFICATION_REPORT_PATH) -> Dict:
    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    save_json(report, path)
    logger.info(f"Saved classification report -> {path}")
    return report


def plot_confusion_matrix(y_true, y_pred, save_path: str = config.CONFUSION_MATRIX_PATH) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix -> {save_path}")


def plot_roc_curve(y_true, y_prob, save_path: str = config.ROC_CURVE_PATH) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve -> {save_path}")


def plot_pr_curve(y_true, y_prob, save_path: str = config.PR_CURVE_PATH) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#c0392b", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PR curve -> {save_path}")


def plot_training_curves(history: Dict[str, List[float]], save_path: str = config.TRAINING_CURVES_PATH) -> None:
    """Plot loss and accuracy curves recorded during training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curves -> {save_path}")
