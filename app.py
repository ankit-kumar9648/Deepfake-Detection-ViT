"""
app.py
======
Streamlit web application (STEP 12 + 13 + 14) for the Deepfake Detection
ViT project.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time

import pandas as pd
import streamlit as st
import torch
from PIL import Image

import config
from predict import load_model, predict_image, preprocess_image
from src.gradcam import generate_attention_rollout, generate_gradcam, overlay_heatmap_on_image
from src.video_utils import predict_video
from utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Deepfake Detection | ViT",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def load_css() -> None:
    css_path = os.path.join(config.BASE_DIR, "streamlit_app", "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# --------------------------------------------------------------------------- #
# Cached model loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_cached_model():
    if not os.path.exists(config.BEST_MODEL_PATH):
        return None
    return load_model(config.BEST_MODEL_PATH)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🕵️ Deepfake Detection")
    st.markdown("Vision Transformer (ViT-B/16) · Transfer Learning")
    st.divider()
    mode = st.radio("Input type", ["Image", "Video (experimental)"], index=0)
    st.divider()
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_attention = st.checkbox("Show attention rollout", value=True)
    st.divider()
    st.caption(f"Device: `{config.DEVICE}`")
    st.caption(f"Backend: `{config.MODEL_BACKEND}`")
    st.caption(f"Checkpoint: `{'found' if os.path.exists(config.BEST_MODEL_PATH) else 'NOT FOUND'}`")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Deepfake Detection using Vision Transformer")
st.markdown(
    "Upload a face image (or short video) and the fine-tuned **ViT-B/16** "
    "model will classify it as **Real** or **Fake**, along with a "
    "confidence score and an explainable-AI heatmap."
)

model = get_cached_model()
if model is None:
    st.warning(
        f"⚠️ No trained checkpoint found at `{config.BEST_MODEL_PATH}`. "
        f"Populate `dataset/` and run `python train.py` first, then restart this app."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def render_prediction_card(result: dict) -> None:
    label = result["label"]
    confidence = result["confidence"]
    color = "#e74c3c" if label == "fake" else "#2ecc71"

    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:0.75rem;background:{color}20;
                    border:1px solid {color};">
            <h2 style="color:{color};margin:0;">Prediction: {label.upper()}</h2>
            <p style="margin:0.3rem 0 0 0;font-size:1.1rem;">
                Confidence: <b>{confidence * 100:.2f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence)

    probs_df = pd.DataFrame(
        {"class": list(result["probabilities"].keys()),
         "probability": list(result["probabilities"].values())}
    )
    st.bar_chart(probs_df.set_index("class"))


def render_explainability(image: Image.Image, target_class: int) -> None:
    tensor = preprocess_image(image)

    cols = st.columns(2 if (show_gradcam and show_attention) else 1)
    col_idx = 0

    if show_gradcam:
        with st.spinner("Computing Grad-CAM..."):
            try:
                heatmap, _ = generate_gradcam(model, tensor, target_class=target_class)
                overlay = overlay_heatmap_on_image(image, heatmap)
                with cols[col_idx]:
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
                col_idx += 1
            except Exception as exc:
                st.info(f"Grad-CAM unavailable for this backend/image: {exc}")

    if show_attention and config.MODEL_BACKEND == "huggingface":
        with st.spinner("Computing attention rollout..."):
            try:
                with torch.no_grad():
                    _, attentions = model(tensor, output_attentions=True)
                heatmap = generate_attention_rollout(attentions)
                overlay = overlay_heatmap_on_image(image, heatmap)
                with cols[col_idx if col_idx < len(cols) else 0]:
                    st.image(overlay, caption="Attention Rollout Overlay", use_container_width=True)
            except Exception as exc:
                st.info(f"Attention rollout unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Image mode
# --------------------------------------------------------------------------- #
if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read this image file: {exc}")
            st.stop()

        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Running inference..."):
            start = time.time()
            result = predict_image(image)
            elapsed = time.time() - start

        with right:
            render_prediction_card(result)
            st.caption(f"Inference time: {elapsed * 1000:.0f} ms")

        st.divider()
        st.subheader("🔍 Explainable AI")
        render_explainability(image, config.CLASS_TO_IDX[result["label"]])
    else:
        st.info("👆 Upload an image to get started.")


# --------------------------------------------------------------------------- #
# Video mode (STEP 14, optional)
# --------------------------------------------------------------------------- #
else:
    st.info(
        "Experimental: the app samples frames from the video and averages "
        "frame-wise predictions into an overall verdict."
    )
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.video(uploaded_video)

        try:
            with st.spinner("Extracting frames and running inference..."):
                video_result = predict_video(tmp_path, predict_fn=predict_image)
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
            os.unlink(tmp_path)
            st.stop()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        render_prediction_card(
            {
                "label": video_result["overall_label"],
                "confidence": video_result["overall_confidence"],
                "probabilities": {
                    "real": video_result["mean_real_probability"],
                    "fake": video_result["mean_fake_probability"],
                },
            }
        )
        st.caption(
            f"Analyzed {video_result['num_frames_analyzed']} frames · "
            f"{video_result['fake_frame_ratio'] * 100:.1f}% classified as fake"
        )

        with st.expander("Frame-by-frame breakdown"):
            frame_df = pd.DataFrame(
                [
                    {"frame": r["frame_index"], "label": r["label"],
                     "confidence": r["confidence"]}
                    for r in video_result["frame_results"]
                ]
            )
            st.dataframe(frame_df, use_container_width=True)
    else:
        st.info("👆 Upload a short video to get started.")


st.divider()
st.caption(
    "Deepfake Detection using Vision Transformer (ViT-B/16) · MCA Major Project · "
    "Built with PyTorch, Transformers & Streamlit"
)