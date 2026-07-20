"""
utils.py
========
Shared, project-wide utility functions: reproducibility helpers, logger
factory, checkpoint I/O and the EarlyStopping controller used by train.py.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import numpy as np
import torch

import config


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.SEED) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Logging (STEP 15)
# --------------------------------------------------------------------------- #
def get_logger(name: str = "deepfake_vit") -> logging.Logger:
    """
    Return a singleton-style logger that writes to both stdout and a
    rotating log file at config.LOG_FILE.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured -> avoid duplicate handlers
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    file_handler = RotatingFileHandler(
        config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
) -> None:
    """Persist model + optimizer state alongside training metadata."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "class_names": config.CLASS_NAMES,
        "image_size": config.IMAGE_SIZE,
        "model_backend": config.MODEL_BACKEND,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint dict saved by save_checkpoint()."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. Train the model first with "
            f"`python train.py`."
        )
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=map_location)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Early stopping (STEP 8)
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopping:
    """
    Stops training when the monitored metric (validation loss, minimised)
    has not improved by at least `min_delta` for `patience` consecutive
    epochs.
    """

    patience: int = config.EARLY_STOPPING_PATIENCE
    min_delta: float = config.EARLY_STOPPING_MIN_DELTA
    best_score: Optional[float] = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def step(self, val_loss: float) -> bool:
        """Call once per epoch with the current validation loss."""
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
