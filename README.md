# 🧠 Deepfake Detection using Vision Transformer (ViT)

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch">
<img src="https://img.shields.io/badge/Model-Vision%20Transformer-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Deployment-Streamlit-green?style=for-the-badge">

</p>


<p align="center">
A complete Deep Learning pipeline for detecting AI-generated and manipulated faces using a fine-tuned Vision Transformer (ViT-B/16).
</p>


---

# 📌 Project Overview

Deepfake technology has become one of the biggest challenges in digital media authenticity. 
This project focuses on developing an intelligent deepfake detection system capable of classifying facial images into:

- ✅ Real Images
- ❌ Fake / AI Generated Images


The system uses **Transfer Learning with Vision Transformer (ViT-B/16)** pretrained on ImageNet-21k and fine-tunes it for binary image classification.

The trained model is deployed using an interactive **Streamlit Web Application** that provides:

- Real/Fake prediction
- Confidence score
- Explainable AI visualization
- Image & Video level analysis


---

# 🎯 Objectives

✔ Build an accurate Real vs Fake image classifier using Transformer architecture.

✔ Implement a complete ML pipeline from data preprocessing to deployment.

✔ Apply Explainable AI techniques for model interpretation.

✔ Create a production-ready modular deep learning project.

✔ Provide an interactive user interface for end users.


---

# ✨ Key Features


## 🤖 Deep Learning Model

- Vision Transformer (ViT-B/16)
- Transfer Learning approach
- Custom classification head
- Frozen backbone + optional fine tuning
- AdamW optimizer
- Cosine learning rate scheduling


## 📊 Data Analysis & Processing

- Dataset visualization
- Class distribution analysis
- Image resolution analysis
- Pixel intensity analysis
- Data augmentation pipeline


## 📈 Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score


Visualization:

- Confusion Matrix
- ROC Curve
- Precision Recall Curve
- Training & Validation Loss


## 🔍 Explainable AI

To understand model decisions:

- Grad-CAM Heatmaps
- Attention Rollout Visualization


## 🌐 Streamlit Application

Features:

- Drag & Drop image upload
- Real/Fake prediction
- Confidence probability
- Visual explanation
- Video frame analysis


---

# 🏗️ System Architecture

