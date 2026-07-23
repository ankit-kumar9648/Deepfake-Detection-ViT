"""
predict.py
==========

Single-image prediction module.

- Loads the model from the local models folder if available.
- Otherwise downloads it automatically from Hugging Face.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Dict, Union

import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

import config
from src.dataset import build_eval_transforms
from src.model import build_model
from utils import get_logger, load_checkpoint

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model(checkpoint_path: str | None = None):
    """
    Load trained model.

    Priority:
    1. Local models/best_model.pth
    2. Download from Hugging Face
    """

    if checkpoint_path is None:

        if os.path.exists(config.BEST_MODEL_PATH):
            checkpoint_path = config.BEST_MODEL_PATH
            logger.info(f"Loading local model: {checkpoint_path}")

        else:
            logger.info("Downloading model from Hugging Face...")

            checkpoint_path = hf_hub_download(
                repo_id="ankit-321/deepfake-vit-model",
                filename="best_model.pth",
            )

            logger.info(f"Downloaded checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    logger.info("Model loaded successfully.")

    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = build_eval_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.to(config.DEVICE)


def extract_logits(output: Union[torch.Tensor, tuple, object]) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits

    if isinstance(output, (tuple, list)):
        return output[0]

    return output


@torch.no_grad()
def predict_image(image: Image.Image) -> Dict:

    model = load_model()

    tensor = preprocess_image(image)

    output = model(tensor)

    logits = extract_logits(output)

    probs = F.softmax(logits, dim=1)[0].cpu()

    pred_idx = int(torch.argmax(probs))

    label = config.IDX_TO_CLASS[pred_idx]

    confidence = float(probs[pred_idx])

    probabilities = {
        config.IDX_TO_CLASS[i]: float(probs[i])
        for i in range(config.NUM_CLASSES)
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Image path",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    image = Image.open(args.image)

    result = predict_image(image)

    print("=" * 40)
    print("Prediction :", result["label"])
    print("Confidence :", round(result["confidence"] * 100, 2), "%")
    print("Probabilities")
    print(result["probabilities"])
    print("=" * 40)


if __name__ == "__main__":
    main()