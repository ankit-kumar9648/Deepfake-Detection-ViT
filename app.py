"""
app.py
======
Streamlit web application for the Deepfake Detection ViT project.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import glob
import os
import tempfile
import time

import pandas as pd
import streamlit as st
import torch
from PIL import Image

import config
from predict import load_model, predict_image, preprocess_image
from src.gradcam import (
    generate_attention_rollout,
    generate_gradcam,
    overlay_heatmap_on_image,
)
from src.video_utils import predict_video
from utils import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Page Setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Deepfake Detection | ViT-B/16 Dashboard",
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
    st.markdown("**Vision Transformer (ViT-B/16)**")
    st.divider()

    mode = st.radio("🎯 Input Mode", ["Image", "Video (experimental)"], index=0)
    st.divider()

    with st.expander("⚙️ Explainable AI (XAI) Settings", expanded=True):
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
        show_attention = st.checkbox("Show Attention Rollout", value=True)

    with st.expander("🖥️ System & Hardware", expanded=False):
        st.caption(f"**Device:** `{config.DEVICE}`")
        st.caption(f"**Backend:** `{config.MODEL_BACKEND}`")
        st.caption(
            f"**Checkpoint:** `{'found' if os.path.exists(config.BEST_MODEL_PATH) else 'NOT FOUND'}`"
        )

    with st.expander("📊 Model Information", expanded=False):
        st.markdown("- **Architecture:** ViT-B/16")
        st.markdown("- **Pretraining:** ImageNet-21k")
        st.markdown("- **Task:** Binary Face Classification")

    st.divider()
    st.caption("MCA Major Project · PyTorch & Transformers")


# --------------------------------------------------------------------------- #
# Header & Key Metrics
# --------------------------------------------------------------------------- #
st.title("🛡️ Deepfake Detection using Vision Transformer")
st.markdown(
    "Upload a face image or video to analyze real vs. synthetic manipulation patterns using fine-tuned **ViT-B/16** and **Explainable-AI (Grad-CAM & Attention Rollout)**."
)

# Key KPI Metrics Top Bar
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Model Architecture", value="ViT-B/16")
kpi2.metric(label="Inference Device", value=str(config.DEVICE).upper())
kpi3.metric(label="Backend", value=str(config.MODEL_BACKEND).capitalize())
kpi4.metric(
    label="Checkpoint Status",
    value="Ready" if os.path.exists(config.BEST_MODEL_PATH) else "Missing",
)

st.divider()

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
    badge_icon = "🚨 FAKE DETECTED" if label == "fake" else "✅ REAL IMAGE"

    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:0.75rem;background:{color}20;
                    border:1px solid {color};margin-bottom:1rem;">
            <span style="color:{color};font-weight:bold;font-size:0.9rem;letter-spacing:1px;">{badge_icon}</span>
            <h2 style="color:{color};margin:0.2rem 0 0 0;">Prediction: {label.upper()}</h2>
            <p style="margin:0.3rem 0 0 0;font-size:1.1rem;">
                Confidence Score: <b>{confidence * 100:.2f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence)

    probs_df = pd.DataFrame(
        {
            "class": list(result["probabilities"].keys()),
            "probability": list(result["probabilities"].values()),
        }
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
                    st.image(
                        overlay,
                        caption="Grad-CAM Focus Overlay",
                        use_container_width=True,
                    )
                    st.info(
                        "Grad-CAM highlights regional spatial activations driving the classification."
                    )
                col_idx += 1
            except Exception as exc:
                st.info(f"Grad-CAM unavailable for this backend/image: {exc}")

    if show_attention and config.MODEL_BACKEND == "huggingface":
        with st.spinner("Computing attention rollout..."):
            try:
                with torch.no_grad():
                    # Unpack the exact (logits, attentions) tuple returned by HuggingFaceViTClassifier
                    logits, attentions = model(tensor, output_attentions=True)
                
                if attentions is not None:
                    heatmap = generate_attention_rollout(attentions)
                    overlay = overlay_heatmap_on_image(image, heatmap)
                    with cols[col_idx if col_idx < len(cols) else 0]:
                        st.image(
                            overlay,
                            caption="Attention Rollout Overlay",
                            use_container_width=True,
                        )
                        st.info(
                            "Attention Rollout tracks self-attention flow across ViT patches."
                        )
            except Exception as exc:
                st.info(f"Attention rollout unavailable: {exc}")
                
# --------------------------------------------------------------------------- #
# Image mode
# --------------------------------------------------------------------------- #
if mode == "Image":
    col_upload, col_samples = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Target Image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

    with col_samples:
        st.markdown("**Or test with existing dataset samples:**")
        sample_real_btn = st.button("🖼️ Sample Real Face", use_container_width=True)
        sample_fake_btn = st.button("🤖 Sample Fake Face", use_container_width=True)

    image_to_process = None

    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read uploaded image: {exc}")
            st.stop()

    elif sample_real_btn:
        real_files = glob.glob(
            os.path.join(config.BASE_DIR, "dataset", "test", "real", "*.*")
        )
        if not real_files:
            real_files = glob.glob(
                os.path.join(config.BASE_DIR, "dataset", "validation", "real", "*.*")
            )

        if real_files:
            image_to_process = Image.open(real_files[0]).convert("RGB")
            st.toast(f"Loaded dataset sample: {os.path.basename(real_files[0])}")
        else:
            st.warning("No real sample image found in dataset/test/real/")

    elif sample_fake_btn:
        fake_files = glob.glob(
            os.path.join(config.BASE_DIR, "dataset", "test", "fake", "*.*")
        )
        if not fake_files:
            fake_files = glob.glob(
                os.path.join(config.BASE_DIR, "dataset", "validation", "fake", "*.*")
            )

        if fake_files:
            image_to_process = Image.open(fake_files[0]).convert("RGB")
            st.toast(f"Loaded dataset sample: {os.path.basename(fake_files[0])}")
        else:
            st.warning("No fake sample image found in dataset/test/fake/")

    if image_to_process is not None:
        left, right = st.columns([1, 1])
        with left:
            st.image(
                image_to_process, caption="Input Target Face", use_container_width=True
            )

        with st.spinner("Running Vision Transformer inference..."):
            start = time.time()
            result = predict_image(image_to_process)
            elapsed = time.time() - start

        with right:
            render_prediction_card(result)
            st.caption(f"⚡ Inference processing time: **{elapsed * 1000:.0f} ms**")

        st.divider()
        st.subheader("🔍 Explainable AI (XAI) Visualizations")
        render_explainability(image_to_process, config.CLASS_TO_IDX[result["label"]])
    else:
        st.info("👆 Upload an image or click a sample face button to run detection.")


# --------------------------------------------------------------------------- #
# Video mode
# --------------------------------------------------------------------------- #
else:
    st.info(
        "⚡ **Experimental:** Video mode extracts individual frames, processes each frame through the Vision Transformer, and computes an ensemble verdict."
    )
    uploaded_video = st.file_uploader(
        "Upload Video Clip", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_video.name)[1]
        ) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        v_left, v_right = st.columns([1, 1])
        with v_left:
            st.video(uploaded_video)

        try:
            with st.spinner("Extracting frames & running batch ViT inference..."):
                video_result = predict_video(tmp_path, predict_fn=predict_image)
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            st.stop()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        with v_right:
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
                f"Analyzed **{video_result['num_frames_analyzed']} frames** · "
                f"**{video_result['fake_frame_ratio'] * 100:.1f}%** frames flagged as fake"
            )

        st.divider()
        with st.expander("📹 Frame-by-Frame Temporal Breakdown", expanded=True):
            frame_df = pd.DataFrame(
                [
                    {
                        "Frame #": r["frame_index"],
                        "Verdict": r["label"].upper(),
                        "Confidence": f"{r['confidence'] * 100:.2f}%",
                    }
                    for r in video_result["frame_results"]
                ]
            )
            st.dataframe(frame_df, use_container_width=True)
    else:
        st.info("👆 Upload a short video clip to begin.")


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    "Deepfake Detection using Vision Transformer (ViT-B/16) · MCA Major Project · "
    "Built with PyTorch, Transformers & Streamlit"
)