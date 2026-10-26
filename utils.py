"""Shared helpers for training and the Streamlit app.

This module intentionally imports neither TensorFlow nor Streamlit so it can be
used (and tested) cheaply from both ``train.py`` and ``app.py``.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
IMG_SIZE = 224  # MobileNetV2 default input resolution
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

MAX_UPLOAD_MB = 10
MIN_SIDE_PX = 64
MAX_PIXELS = 50_000_000  # guards against decompression bombs
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be used as model input."""


# ----------------------------------------------------------------------------
# Class-name handling
# ----------------------------------------------------------------------------
def parse_class_name(raw: str) -> dict:
    """Turn a folder name into display-friendly parts.

    Supports two formats:
    - PlantVillage: ``"Tomato___Late_blight"`` -> plant ``"Tomato"``, condition ``"Late blight"``
    - Blackgram BPLD: plain names like ``"healthy"``, ``"anthracnose"``

    Only string cleanup is performed; no information is added.
    """
    raw = str(raw)
    is_background = raw.lower().startswith("background")
    if "___" in raw:
        plant_raw, condition_raw = raw.split("___", 1)
    else:
        plant_raw, condition_raw = raw, ""

    def clean(text: str) -> str:
        return " ".join(text.replace("_", " ").split()).strip()

    plant = clean(plant_raw)
    condition = clean(condition_raw)
    # Support both PlantVillage (with "___") and BPLD (plain names like "healthy")
    if condition.lower() == "healthy" or plant.lower() == "healthy":
        healthy = True
    else:
        healthy = False
    if is_background:
        plant, condition, healthy = "No leaf", "Background", False
    # Build a display label: for PlantVillage use "Plant - Condition", for BPLD just use the plant name
    if "___" in raw:
        label = f"{plant} - {condition}" if condition else plant
    else:
        label = plant
    return {
        "raw": raw,
        "plant": plant,
        "condition": condition or plant,
        "healthy": healthy,
        "is_background": is_background,
        "label": label,
    }


# ----------------------------------------------------------------------------
# Image validation
# ----------------------------------------------------------------------------
def load_and_validate_image(data: bytes) -> Image.Image:
    """Decode bytes into a clean RGB PIL image or raise ``InvalidImageError``."""
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidImageError(f"The file is larger than {MAX_UPLOAD_MB} MB.")

    try:
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or "").upper()
        probe.verify()  # detects truncated / corrupted files
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImageError(
            "This file is not a readable image. Please upload a JPG, PNG, WEBP or BMP photo."
        ) from None

    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format '{fmt or 'unknown'}'. Use JPG, PNG, WEBP or BMP."
        )

    try:
        img = Image.open(io.BytesIO(data))  # verify() invalidates the first handle
        img = ImageOps.exif_transpose(img)  # respect phone-camera orientation
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImageError("The image could not be decoded (it may be corrupted).") from None

    width, height = img.size
    if min(width, height) < MIN_SIDE_PX:
        raise InvalidImageError(
            f"The image is too small ({width}x{height}px). Minimum side is {MIN_SIDE_PX}px."
        )
    return img


def center_square_crop(img: Image.Image) -> Image.Image:
    """Crop the largest centered square (PlantVillage training images are square)."""
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))
