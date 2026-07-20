"""
predict.py
==========
Single-image prediction module (STEP 11). Loads the best checkpoint once
and exposes `predict_image()` for reuse by both the CLI and app.py.

CLI usage
---------
    python predict.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

import config
from src.dataset import build_eval_transforms
from src.model import build_model
from utils import get_logger, load_checkpoint

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model(checkpoint_path: str = config.BEST_MODEL_PATH) -> torch.nn.Module:
    """Load (and cache) the trained model for inference."""
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from '{checkpoint_path}' (epoch {checkpoint.get('epoch')}).")
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the deterministic eval transform and add a batch dimension."""
    transform = build_eval_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.to(config.DEVICE)


@torch.no_grad()
def predict_image(image: Image.Image, checkpoint_path: str = config.BEST_MODEL_PATH) -> Dict:
    """
    Run inference on a single PIL image.

    Returns
    -------
    dict with keys: label, confidence, probabilities ({'real': p, 'fake': p})
    """
    model = load_model(checkpoint_path)
    tensor = preprocess_image(image)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu()

    pred_idx = int(torch.argmax(probs).item())
    label = config.IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx].item())

    probabilities = {
        config.IDX_TO_CLASS[i]: float(probs[i].item()) for i in range(config.NUM_CLASSES)
    }

    return {"label": label, "confidence": confidence, "probabilities": probabilities}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict real/fake for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file.")
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image)
    result = predict_image(image, checkpoint_path=args.checkpoint)

    print("\n" + "=" * 40)
    print(f"Image      : {args.image}")
    print(f"Prediction : {result['label'].upper()}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"Probabilities : real={result['probabilities']['real']:.4f} | "
          f"fake={result['probabilities']['fake']:.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
