"""
train.py
========
End-to-end training entry point (STEP 8) for the Deepfake Detection ViT
classifier.

Usage
-----
    python train.py
    python train.py --epochs 10 --batch-size 16
    python train.py --unfreeze          # fine-tune last N backbone blocks too

Produces
--------
    models/best_model.pth, models/last_model.pth
    outputs/training_history.json
    outputs/plots/training_curves.png
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.dataset import get_dataloaders
from src.metrics import plot_training_curves
from src.model import build_model, get_param_groups
from utils import EarlyStopping, count_parameters, get_logger, save_checkpoint, save_json, set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Deepfake Detection ViT model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--unfreeze", action="store_true",
                         help="Unfreeze the last transformer blocks for full fine-tuning.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler: torch.cuda.amp.GradScaler,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    phase = "train" if train else "val"
    progress = tqdm(loader, desc=phase, leave=False)

    for images, labels in progress:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if config.USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        progress.set_postfix(loss=loss.item())

    return {"loss": total_loss / total, "acc": correct / total}


def main() -> None:
    args = parse_args()
    set_seed()

    logger.info(f"Device: {config.DEVICE} | AMP: {config.USE_AMP}")
    logger.info("Loading dataloaders ...")
    loaders = get_dataloaders(batch_size=args.batch_size)
    logger.info(f"Train batches: {len(loaders['train'])} | "
                f"Val batches: {len(loaders['validation'])}")

    logger.info(f"Building model (backend={config.MODEL_BACKEND}) ...")
    model = build_model()
    if args.unfreeze:
        logger.info(f"Unfreezing last {config.UNFREEZE_LAST_N_BLOCKS} backbone blocks.")
        model.unfreeze_last_n_blocks()

    param_counts = count_parameters(model)
    logger.info(f"Parameters -> total: {param_counts['total']:,} | "
                f"trainable: {param_counts['trainable']:,} | "
                f"frozen: {param_counts['frozen']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        get_param_groups(model), weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.LR_SCHEDULER_T_MAX, eta_min=config.LR_SCHEDULER_ETA_MIN
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    early_stopping = EarlyStopping()

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, scaler, train=True)
        val_metrics = run_epoch(model, loaders["validation"], criterion, optimizer, scaler, train=False)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        save_checkpoint(model, optimizer, epoch, val_metrics, config.LAST_MODEL_PATH)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, config.BEST_MODEL_PATH)
            logger.info(f"New best model saved (val_loss={best_val_loss:.4f}) -> {config.BEST_MODEL_PATH}")

        if early_stopping.step(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {early_stopping.patience} epochs).")
            break

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time / 60:.1f} minutes.")

    save_json(history, config.TRAINING_HISTORY_PATH)
    plot_training_curves(history)
    logger.info(f"Best model: {config.BEST_MODEL_PATH} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
