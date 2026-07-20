"""
config.py
=========
Single source of truth for every configurable value used across the
Deepfake Detection (ViT) project. Nothing in src/, train.py, test.py,
predict.py or app.py should hard-code a path, hyper-parameter or class
name -- everything is imported from here.
"""

import os
import torch


# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Create every directory the project needs on first import (STEP 1).
for _dir in [
    DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    os.path.join(TRAIN_DIR, "real"), os.path.join(TRAIN_DIR, "fake"),
    os.path.join(VAL_DIR, "real"), os.path.join(VAL_DIR, "fake"),
    os.path.join(TEST_DIR, "real"), os.path.join(TEST_DIR, "fake"),
    MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, LOGS_DIR,
    CHECKPOINTS_DIR, PREDICTIONS_DIR, REPORTS_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# Class configuration
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["fake", "real"]           # alphabetical -> matches ImageFolder
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# --------------------------------------------------------------------------- #
# Image / preprocessing configuration
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]
# NOTE: google/vit-base-patch16-224 was pretrained with mean/std = 0.5 (see
# HF preprocessor_config.json). If TORCHVISION_BACKBONE is used instead,
# ImageNet stats [0.485,0.456,0.406]/[0.229,0.224,0.225] are substituted
# automatically inside src/dataset.py.

# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
# "huggingface" -> transformers ViTModel google/vit-base-patch16-224
# "torchvision" -> torchvision.models.vit_b_16 with IMAGENET1K_V1 weights
MODEL_BACKEND = "huggingface"
HF_MODEL_NAME = "google/vit-base-patch16-224"
FREEZE_BACKBONE = True          # freeze pretrained backbone initially (STEP 7)
UNFREEZE_LAST_N_BLOCKS = 2      # blocks to unfreeze during fine-tuning stage
DROPOUT = 0.3

# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # mixed precision only if CUDA available

BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 12
LEARNING_RATE = 3e-4
HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

LR_SCHEDULER_T_MAX = EPOCHS
LR_SCHEDULER_ETA_MIN = 1e-6

SEED = 42

# --------------------------------------------------------------------------- #
# Augmentation configuration (STEP 5)
# --------------------------------------------------------------------------- #
AUGMENTATION = {
    "horizontal_flip_p": 0.5,
    "rotation_degrees": 15,
    "random_crop_padding": 8,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.05,
    },
    "random_erasing_p": 0.25,
}

# --------------------------------------------------------------------------- #
# Checkpoint / artifact filenames
# --------------------------------------------------------------------------- #
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "last_model.pth")
TRAINING_HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")
TRAINING_CURVES_PATH = os.path.join(PLOTS_DIR, "training_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(PLOTS_DIR, "roc_curve.png")
PR_CURVE_PATH = os.path.join(PLOTS_DIR, "pr_curve.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.json")
EDA_REPORT_DIR = os.path.join(PLOTS_DIR, "eda")
os.makedirs(EDA_REPORT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Logging configuration (STEP 15)
# --------------------------------------------------------------------------- #
LOG_FILE = os.path.join(LOGS_DIR, "project.log")
LOG_LEVEL = "INFO"

# --------------------------------------------------------------------------- #
# Grad-CAM / Explainability (STEP 13)
# --------------------------------------------------------------------------- #
GRADCAM_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "gradcam")
os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Video inference configuration (STEP 14)
# --------------------------------------------------------------------------- #
VIDEO_FRAME_SAMPLE_RATE = 15     # analyse every Nth frame
VIDEO_MAX_FRAMES = 60            # hard cap per video to bound runtime
