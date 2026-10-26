"""Plant Disease Detection - Streamlit app (MobileNetV2 transfer learning)."""

from __future__ import annotations

import html
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# A missing package must produce a readable message, not a stack trace.
# (requirements.txt is the only file Streamlit Community Cloud installs.)
try:
    import numpy as np

    from utils import (
        CLASS_NAMES_PATH,
        METRICS_PATH,
        MODEL_DIR,
        MODEL_PATH,
        InvalidImageError,
        center_square_crop,
        load_and_validate_image,
        parse_class_name,
    )
except ImportError as exc:  # numpy / Pillow (PIL) not installed
    st.error(
        f"**Missing Python package** `{getattr(exc, 'name', None) or exc}`.\n\n"
        "Install the app dependencies and restart the app:\n\n"
        "```\npython -m pip install -r requirements.txt\n```"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; max-width: 1150px; }
      .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #66bb6a 100%);
        color: #fff; padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(27,94,32,.25);
      }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 2.1rem; color: #fff; }
      .hero p  { margin: 0; font-size: 1.05rem; opacity: .95; color: #fff; }
      .card {
        border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);
        border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
      }
      .result { border-left: 8px solid #2e7d32; }
      .result.disease { border-left-color: #d84315; }
      .result.neutral { border-left-color: #757575; }
      .badge {
        display: inline-block; padding: .2rem .75rem; border-radius: 999px; font-weight: 600;
        font-size: .82rem; letter-spacing: .02em; color: #fff; background: #2e7d32;
      }
      .badge.disease { background: #d84315; }
      .badge.neutral { background: #757575; }
      .result h2 { margin: .6rem 0 .1rem 0; font-size: 1.7rem; }
      .muted { opacity: .72; }
      .bar-wrap { background: rgba(128,128,128,.22); border-radius: 999px; height: 12px; overflow: hidden; }
      .bar { height: 12px; border-radius: 999px; background: linear-gradient(90deg,#43a047,#9ccc65); }
      .bar.low { background: linear-gradient(90deg,#ef6c00,#ffb300); }
      .row { display: flex; justify-content: space-between; font-size: .93rem; margin: .55rem 0 .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model loading (cached once per server process)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    from tensorflow import keras  # imported lazily to keep first paint fast

    class_names = json.loads(CLASS_NAMES_PATH.read_text())
    model = keras.models.load_model(MODEL_PATH, compile=False)
    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model has {model.output_shape[-1]} outputs but class_names.json lists {len(class_names)} classes."
        )
    return model, class_names


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def predict(model, img) -> np.ndarray:
    """Return a probability vector. Uses the same resize op as the training pipeline.

    The side length comes from the loaded model itself, so an app never has to
    match ``--img_size`` used at training time by hand.
    """
    import tensorflow as tf

    height, width = model.input_shape[1], model.input_shape[2]
    arr = np.asarray(center_square_crop(img), dtype="float32")[None, ...]
    x = tf.image.resize(arr, (height, width), method="bilinear").numpy()
    probs = model(x, training=False).numpy()[0]
    if not np.all(np.isfinite(probs)):
        raise RuntimeError("The model returned invalid values for this image.")
    return probs


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def confidence_bar(label: str, prob: float, low: bool = False) -> str:
    pct = prob * 100
    cls = "bar low" if low else "bar"
    return (
        f'<div class="row"><span>{html.escape(label)}</span><b>{pct:.1f}%</b></div>'
        f'<div class="bar-wrap"><div class="{cls}" style="width:{max(pct, 1):.1f}%"></div></div>'
    )


def render_result(info: dict, prob: float, threshold: float) -> None:
    if info["is_background"]:
        kind, badge = "neutral", "No leaf detected"
    elif info["healthy"]:
        kind, badge = "", "Healthy"
    else:
        kind, badge = "disease", "Disease detected"
    condition = info["condition"]
    title = (
        "Background / no leaf"
        if info["is_background"]
        else condition[:1].upper() + condition[1:]  # "healthy" -> "Healthy"
    )
    st.markdown(
        f"""
        <div class="card result {kind}">
          <span class="badge {kind}">{html.escape(badge)}</span>
          <h2>{html.escape(title)}</h2>
          <div class="muted">{html.escape(info["plant"]) if not info["is_background"] else "Try a clear photo of a single leaf"}</div>
          <div style="margin-top:1rem">{confidence_bar("Model confidence", prob, prob < threshold)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_setup_instructions() -> None:
    st.warning("No trained model was found in the `models/` folder yet.")
    st.markdown(
        """
        **Train it once, then commit the result:**
        1. Ensure the Blackgram dataset is in `data/plantvillage/` (five class folders: anthracnose, healthy, leaf_crinkle, powdery_mildew, yellow_mosaic).
        2. Run `python train.py --data_dir data/plantvillage`.
        3. Commit `models/plant_disease_mobilenetv2.keras` and `models/class_names.json`, then redeploy.
        """
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Confidence warning threshold", 0.30, 0.95, 0.60, 0.05,
                          help="Predictions below this confidence are flagged as uncertain.")
    top_k = st.slider("Predictions to show", 1, 5, 3)
    st.divider()
    st.subheader("📸 Photo tips")
    st.markdown(
        "- One leaf, filling most of the frame\n"
        "- Even daylight, no heavy shadows\n"
        "- In focus, plain background works best\n"
        "- JPG / PNG / WEBP / BMP, up to 10 MB"
    )
    metrics_sidebar = load_metrics()
    if metrics_sidebar:
        st.divider()
        st.caption(
            f"Model: MobileNetV2 · {metrics_sidebar.get('num_classes', '?')} classes · "
            f"held-out test accuracy {metrics_sidebar.get('test_accuracy', 0) * 100:.2f}%"
        )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌿 Plant Disease Detector</h1>
      <p>Upload a leaf photo and get an instant prediction from a MobileNetV2 model
         fine-tuned on the Blackgram Plant Leaf Disease dataset using transfer learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_detect, tab_perf, tab_about = st.tabs(["🔍 Diagnose", "📊 Model performance", "ℹ️ About"])

# ----------------------------------------------------------------------------
# Diagnose tab
# ----------------------------------------------------------------------------
with tab_detect:
    if not (MODEL_PATH.exists() and CLASS_NAMES_PATH.exists()):
        show_setup_instructions()
    else:
        try:
            model, class_names = load_model_and_classes()
        except ImportError as exc:  # TensorFlow (or one of its deps) is not installed
            st.error(
                f"**Missing Python package** `{getattr(exc, 'name', None) or exc}` - "
                "TensorFlow is required to load the model.\n\n"
                "Install the app dependencies and restart the app:\n\n"
                "```\npython -m pip install -r requirements.txt\n```"
            )
            st.stop()
        except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a stack trace
            st.error("The model could not be loaded.")
            with st.expander("Technical details"):
                st.code(str(exc))
            st.stop()

        source = st.radio("Image source", ["Upload a file", "Use camera"], horizontal=True,
                          label_visibility="collapsed")
        if source == "Upload a file":
            picked = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                      label_visibility="collapsed")
        else:
            picked = st.camera_input("Take a photo of a leaf", label_visibility="collapsed")

        if picked is None:
            st.info("👆 Add a leaf photo to get a prediction.")
        else:
            try:
                img = load_and_validate_image(picked.getvalue())
            except InvalidImageError as exc:
                st.error(f"⚠️ {exc}")
                st.stop()

            col_img, col_res = st.columns([1, 1.15], gap="large")
            with col_img:
                st.image(img, caption=f"Your image ({img.size[0]}×{img.size[1]} px)", width="stretch")
                st.caption("The central square of the photo is analysed.")

            with col_res:
                try:
                    with st.spinner("Analysing leaf..."):
                        probs = predict(model, img)
                except Exception as exc:  # noqa: BLE001
                    st.error("Something went wrong while analysing this image. Please try another photo.")
                    with st.expander("Technical details"):
                        st.code(str(exc))
                    st.stop()

                order = np.argsort(probs)[::-1]
                best = int(order[0])
                info = parse_class_name(class_names[best])
                best_prob = float(probs[best])
                render_result(info, best_prob, threshold)

                if best_prob < threshold:
                    st.warning(
                        "Low confidence. This may not be a clear leaf photo, or the condition may not be one "
                        "the model knows. Treat the result as a guess and try a closer, sharper photo."
                    )
                elif len(order) > 1 and best_prob - float(probs[order[1]]) < 0.15:
                    st.info("This was a close call between the top two predictions - see the list below.")

                st.markdown("##### Top predictions")
                bars = "".join(
                    confidence_bar(parse_class_name(class_names[int(i)])["label"], float(probs[int(i)]),
                                   float(probs[int(i)]) < threshold)
                    for i in order[:top_k]
                )
                st.markdown(f'<div class="card">{bars}</div>', unsafe_allow_html=True)
                st.caption("This tool supports, but does not replace, advice from an agricultural expert.")

# ----------------------------------------------------------------------------
# Performance tab
# ----------------------------------------------------------------------------
with tab_perf:
    metrics = load_metrics()
    if not metrics:
        st.info("Metrics appear here after you run `train.py` and commit `models/metrics.json`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
        c2.metric("Macro F1", f"{metrics['macro_f1'] * 100:.2f}%")
        c3.metric("Classes", metrics["num_classes"])
        sizes = metrics.get("split_sizes", {})
        c4.metric("Test images", f"{sizes.get('test', 0):,}")
        st.caption(
            f"Train {sizes.get('train', 0):,} · validation {sizes.get('val', 0):,} · test {sizes.get('test', 0):,} "
            f"images. Splits were made before augmentation; the test set was used once, for this report. "
            f"Selected stage: {metrics.get('selected_phase', 'n/a').replace('_', ' ')}. "
            f"Trained {metrics.get('trained_at', '')}."
        )
    curves, cm_path = MODEL_DIR / "training_curves.png", MODEL_DIR / "confusion_matrix.png"
    if curves.exists():
        st.image(str(curves), caption="Training and validation curves", width="stretch")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix on the held-out test set", width="stretch")

# ----------------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### How it works
        This app uses **transfer learning**: a MobileNetV2 network pre-trained on ImageNet already knows
        edges, textures and shapes, so it is reused as the starting point for plant-disease classification.

        1. **Feature extraction** - the pre-trained layers are frozen and a new classifier head is trained.
        2. **Fine-tuning** - the top MobileNetV2 layers are unfrozen and trained with a very small learning rate.
        3. **Evaluation** - accuracy is reported on a test set that was never used for training or model selection.

        ### Limitations
        - Blackgram Plant Leaf Disease photos are taken on plain backgrounds; real field photos can score lower than the test set.
        - The model only knows the classes it was trained on, and it always returns *some* answer.
          Use the confidence value, and treat low-confidence results as unreliable.
        - Results are decision support, not a professional diagnosis.
        """
    )
