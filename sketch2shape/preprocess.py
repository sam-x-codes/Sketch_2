"""Stage 1: turn a canvas drawing or photo into a ControlNet scribble map."""

from __future__ import annotations

from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image

from .config import INPUT_TARGET_SIZE

ArrayLike = Union[np.ndarray, Image.Image, Dict[str, Any], str, None]


class SketchError(ValueError):
    """Raised when the input has no usable strokes."""


def to_rgb_uint8(image: ArrayLike) -> np.ndarray:
    """Normalise Gradio / PIL / path / numpy inputs to HxWx3 RGB uint8."""
    if image is None:
        raise SketchError("No image provided.")
    if isinstance(image, dict):
        image = (
            image.get("composite")
            or image.get("image")
            or image.get("value")
            or image.get("background")
        )
        if image is None:
            raise SketchError("Empty sketch canvas.")
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGBA"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        arr = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise SketchError(f"Unsupported image shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def preprocess_sketch(
    image: ArrayLike,
    *,
    target_size: int = INPUT_TARGET_SIZE,
) -> Dict[str, np.ndarray]:
    """Denoise, binarise, crop, square and resize a sketch.

    Returns:
        processed: RGB 512² clean sketch (dark strokes on white)
        control_map: RGB 512² ControlNet scribble (white strokes on black)
        sketch: grayscale control map
    """
    rgb = to_rgb_uint8(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    white_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    corrected = np.clip(gray.astype(np.int16) - white_hat + black_hat, 0, 255).astype(np.uint8)
    if hasattr(cv2, "fastNlMeansDenoise"):
        denoised = cv2.fastNlMeansDenoise(corrected, h=10)
    else:
        denoised = cv2.GaussianBlur(corrected, (5, 5), 0)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Light-background sketches: Otsu yields white paper / black ink. Invert so strokes are white.
    if float(np.mean(denoised)) >= 127:
        strokes = cv2.bitwise_not(otsu)
    else:
        strokes = otsu

    strokes = _remove_specks(strokes, min_area=8)
    ink_ratio = float(np.count_nonzero(strokes)) / float(strokes.size)
    if ink_ratio < 0.0008:
        raise SketchError(
            "no visible strokes / too faint — use dark, bold strokes on a light background."
        )

    cropped = _tight_crop(strokes)
    squared = _pad_square(cropped, fill=0)
    control = cv2.resize(squared, (target_size, target_size), interpolation=cv2.INTER_AREA)
    _, control = cv2.threshold(control, 32, 255, cv2.THRESH_BINARY)

    sketch_on_white = cv2.bitwise_not(control)
    processed = cv2.cvtColor(sketch_on_white, cv2.COLOR_GRAY2RGB)
    control_map = cv2.cvtColor(control, cv2.COLOR_GRAY2RGB)
    return {
        "processed": processed,
        "control_map": control_map,
        "sketch": control,
    }


def _remove_specks(binary: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == i] = 255
    return kept


def _tight_crop(binary: np.ndarray, margin: int = 8) -> np.ndarray:
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    y0 = max(0, y - margin)
    x0 = max(0, x - margin)
    y1 = min(binary.shape[0], y + h + margin)
    x1 = min(binary.shape[1], x + w + margin)
    return binary[y0:y1, x0:x1]


def _pad_square(image: np.ndarray, fill: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side), fill, dtype=image.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas
