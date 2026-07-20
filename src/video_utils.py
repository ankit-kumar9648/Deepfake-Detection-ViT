"""
src/video_utils.py
===================
Optional video-level deepfake detection (STEP 14): extract frames with
OpenCV, run frame-wise prediction, and aggregate into an overall
video verdict.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import cv2
import numpy as np
from PIL import Image

import config
from utils import get_logger

logger = get_logger(__name__)


def extract_frames(
    video_path: str,
    sample_rate: int = config.VIDEO_FRAME_SAMPLE_RATE,
    max_frames: int = config.VIDEO_MAX_FRAMES,
) -> List[Image.Image]:
    """
    Sample every `sample_rate`-th frame from a video, up to `max_frames`
    frames total, and return them as PIL images (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: '{video_path}'")

    frames = []
    frame_idx = 0
    try:
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            frame_idx += 1
    finally:
        cap.release()

    logger.info(f"Extracted {len(frames)} frames from '{video_path}' "
                f"(sample_rate={sample_rate}, scanned {frame_idx} raw frames)")
    return frames


def predict_video(
    video_path: str,
    predict_fn: Callable[[Image.Image], Dict],
    sample_rate: int = config.VIDEO_FRAME_SAMPLE_RATE,
    max_frames: int = config.VIDEO_MAX_FRAMES,
) -> Dict:
    """
    Run frame-wise prediction over a sampled set of video frames and
    aggregate into an overall verdict.

    `predict_fn` must accept a PIL.Image and return a dict with keys
    'label' and 'confidence' (as produced by predict.predict_image).
    """
    frames = extract_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)
    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    frame_results = []
    for i, frame in enumerate(frames):
        result = predict_fn(frame)
        result["frame_index"] = i
        frame_results.append(result)

    fake_scores = [r["probabilities"]["fake"] for r in frame_results]
    real_scores = [r["probabilities"]["real"] for r in frame_results]

    mean_fake = float(np.mean(fake_scores))
    mean_real = float(np.mean(real_scores))
    overall_label = "fake" if mean_fake > mean_real else "real"
    fake_frame_ratio = float(np.mean([1 if r["label"] == "fake" else 0 for r in frame_results]))

    return {
        "overall_label": overall_label,
        "overall_confidence": max(mean_fake, mean_real),
        "mean_fake_probability": mean_fake,
        "mean_real_probability": mean_real,
        "fake_frame_ratio": fake_frame_ratio,
        "num_frames_analyzed": len(frame_results),
        "frame_results": frame_results,
    }
