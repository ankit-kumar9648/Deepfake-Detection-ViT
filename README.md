# Deepfake Detection using Vision Transformer (ViT)

An end-to-end MCA major project that fine-tunes a **pretrained Vision
Transformer (ViT-B/16)** to classify face images (and optionally video
frames) as **Real** or **Fake**, complete with an explainable-AI
Streamlit web application.

---

## 1. Project Overview

Deepfakes — synthetically generated or manipulated media — pose a
growing threat to digital trust, journalism and personal security. This
project builds a complete deepfake **image classifier** using **transfer
learning** on top of a Vision Transformer pretrained on ImageNet-21k
(`google/vit-base-patch16-224`), rather than training a transformer from
scratch. The trained model is deployed through an interactive Streamlit
app with Grad-CAM / attention-rollout explainability and optional
video-level inference.

## 2. Objectives

- Fine-tune a pretrained ViT-B/16 for binary Real vs. Fake image classification.
- Build a clean, reproducible, modular training/evaluation pipeline.
- Provide full exploratory data analysis (EDA) and evaluation reporting.
- Ship an interactive, explainable web application.
- Support optional frame-wise video analysis.
- Package the project as a professional, GitHub-ready repository.

## 3. Features

- 🤖 **Transfer learning** on a pretrained ViT-B/16 (backbone frozen, custom classifier head, optional partial unfreezing for fine-tuning).
- 📊 **Full EDA**: class distribution, resolution scatter, pixel histograms, random sample grids.
- 🖼️ **Augmentation pipeline**: flip, rotation, crop, color jitter, random erasing.
- ⚡ **Efficient training**: AdamW + CosineAnnealingLR + mixed precision (AMP) on CUDA + early stopping + best-checkpointing.
- 📈 **Rich evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC & PR curves, classification report.
- 🌐 **Streamlit app**: drag-and-drop image/video upload, confidence bars, loading states, error handling.
- 🔍 **Explainable AI**: Grad-CAM heatmaps and attention-rollout overlays.
- 🎥 **Optional video support**: frame extraction + frame-wise + overall prediction.
- 🪵 **Logging** to console + rotating log file.
- ⚙️ **Single config file** for every hyperparameter and path.

## 4. Folder Structure

```
Deepfake-Detection-ViT/
├── dataset/
│   ├── train/{real,fake}/
│   ├── validation/{real,fake}/
│   └── test/{real,fake}/
├── notebooks/
│   └── Deepfake_Detection_ViT.ipynb
├── src/
│   ├── dataset.py        # transforms, augmentation, DataLoaders
│   ├── model.py           # pretrained ViT + custom classifier head
│   ├── eda.py              # exploratory data analysis
│   ├── metrics.py         # evaluation metrics & plots
│   ├── gradcam.py         # Grad-CAM + attention rollout (XAI)
│   └── video_utils.py     # frame extraction & video inference
├── models/                # saved checkpoints (best_model.pth, last_model.pth)
├── outputs/
│   ├── plots/              # EDA graphs, training curves, confusion matrix, ROC/PR
│   ├── logs/                # rotating log files
│   ├── checkpoints/
│   └── predictions/
├── reports/                # project report & slides
├── streamlit_app/
│   └── assets/style.css
├── config.py               # all hyperparameters & paths
├── train.py                 # training entry point
├── test.py                   # evaluation entry point
├── predict.py               # single-image inference module
├── utils.py                  # logging, checkpoints, early stopping
├── app.py                     # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

```bash
git clone <your-repo-url>
cd Deepfake-Detection-ViT

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> GPU users: install the CUDA build of PyTorch that matches your driver
> from https://pytorch.org/get-started/locally/ *before* installing the
> rest of `requirements.txt`, so `torch.cuda.is_available()` returns
> `True`.

## 6. Dataset Structure

Place your dataset inside `dataset/` following the ImageFolder layout
below (any common public deepfake dataset — e.g. FaceForensics++,
Celeb-DF, DFDC, 140k Real and Fake Faces — can be reorganised into this
structure):

```
dataset/
├── train/
│   ├── real/   *.jpg
│   └── fake/   *.jpg
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

A recommended split ratio is 70% train / 15% validation / 15% test,
with a comparable number of real and fake images per split to avoid
class imbalance.

## 7. Training

```bash
python train.py
python train.py --epochs 15 --batch-size 16 --lr 3e-4
python train.py --unfreeze          # also fine-tunes the last backbone blocks
```

Outputs: `models/best_model.pth`, `models/last_model.pth`,
`outputs/training_history.json`, `outputs/plots/training_curves.png`.

All hyperparameters (learning rate, batch size, epochs, freeze
strategy, augmentation strengths, etc.) live in `config.py`.

## 8. Testing / Evaluation

```bash
python test.py
python test.py --checkpoint models/last_model.pth
```

Outputs (all under `outputs/`): `test_metrics.json`,
`classification_report.json`, `plots/confusion_matrix.png`,
`plots/roc_curve.png`, `plots/pr_curve.png`.

## 9. Prediction (single image)

```bash
python predict.py --image path/to/image.jpg
```

Prints the predicted label, confidence and full class probabilities.
The same `predict_image()` function is reused by the Streamlit app.

## 10. Exploratory Data Analysis

```bash
python -m src.eda
```

Saves class distribution (bar + pie), resolution scatter, random
sample grid and pixel-intensity histograms under
`outputs/plots/eda/`.

## 11. Running the Web App

```bash
streamlit run app.py
```

Open the printed local URL in your browser. Features:

- Image or video upload
- Real/Fake prediction with confidence score and probability bar chart
- Grad-CAM and attention-rollout heatmap overlays
- Loading spinners and graceful error handling for unreadable files

> The app requires `models/best_model.pth` to exist — train the model
> first with `python train.py`.

## 12. Deployment

The app is a standard Streamlit application and can be deployed to
Streamlit Community Cloud, a Docker container, or any VM/server with
Python 3.10+:

```bash
docker build -t deepfake-vit .          # if you add a Dockerfile
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Make sure `models/best_model.pth` is included in the deployment image
or mounted as a volume, since it is excluded from version control by
`.gitignore`.

## 13. Results

This repository ships **complete, runnable code** but **no bundled
dataset or pretrained weights** (the actual accuracy/precision/recall/
F1/ROC-AUC figures depend entirely on the dataset you place in
`dataset/`). After running `python train.py` followed by
`python test.py`, your own results will be written to:

- `outputs/test_metrics.json` — accuracy, precision, recall, F1, ROC-AUC
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/roc_curve.png`, `outputs/plots/pr_curve.png`
- `outputs/plots/training_curves.png`

Paste those numbers/plots into `reports/Project_Report.docx` and
`reports/Project_Presentation.pptx` (both included as ready-to-fill
templates in this repository) to finalise your submission.

## 14. Future Scope

- Extend to multi-frame temporal models (3D-CNN / Video ViT) for native video deepfake detection.
- Add adversarial-robustness evaluation against compression/noise attacks.
- Explore ensemble of ViT + CNN (e.g. EfficientNet) backbones.
- Add face-detection/alignment preprocessing (MTCNN) for in-the-wild images.
- Deploy as a REST API (FastAPI) alongside the Streamlit demo.
- Active-learning loop to continuously incorporate newly labelled hard examples.

## 15. Tech Stack

Python · PyTorch · Torchvision · Transformers (HuggingFace) · OpenCV ·
Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit · Pillow · tqdm

## 16. License

This project is provided for academic (MCA major project) use.
