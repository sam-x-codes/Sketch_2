"""Stage 3: rembg foreground isolation, then center/scale on white."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .config import FOREGROUND_RATIO, PREPARE_SIZE


class PrepareError(ValueError):
    """Raised when rembg cannot isolate a foreground object."""


_SESSION = None


def _session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    from rembg import new_session

    for name in ("birefnet-general", "u2net"):
        try:
            _SESSION = new_session(name)
            print(f"rembg session: {name}", flush=True)
            return _SESSION
        except Exception as exc:
            print(f"rembg {name} unavailable ({exc})", flush=True)
    from rembg import new_session as _ns

    _SESSION = _ns("u2net")
    return _SESSION


def remove_background(image: np.ndarray) -> np.ndarray:
    from rembg import remove

    rgb = image
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgba = remove(rgb, session=_session())
    return np.asarray(rgba)


def center_and_scale_foreground(
    rgba: np.ndarray,
    *,
    foreground_ratio: float = FOREGROUND_RATIO,
    target_size: int = PREPARE_SIZE,
) -> Dict[str, np.ndarray]:
    if rgba.shape[2] != 4:
        raise PrepareError("Expected an RGBA image from rembg.")
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    mask = alpha > 12
    if int(np.count_nonzero(mask)) < 80:
        raise PrepareError(
            "Could not isolate a foreground object — try another seed or add a short prompt."
        )

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_a = alpha[y0:y1, x0:x1]
    h, w = crop_rgb.shape[:2]
    scale = (target_size * foreground_ratio) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_a = cv2.resize(crop_a, (new_w, new_h), interpolation=cv2.INTER_AREA)

    prepared = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    rgba_out = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    oy = (target_size - new_h) // 2
    ox = (target_size - new_w) // 2
    a = (scaled_a.astype(np.float32) / 255.0)[..., None]
    region = prepared[oy : oy + new_h, ox : ox + new_w].astype(np.float32)
    blended = scaled_rgb.astype(np.float32) * a + region * (1.0 - a)
    prepared[oy : oy + new_h, ox : ox + new_w] = np.clip(blended, 0, 255).astype(np.uint8)
    rgba_out[oy : oy + new_h, ox : ox + new_w, :3] = scaled_rgb
    rgba_out[oy : oy + new_h, ox : ox + new_w, 3] = scaled_a
    return {
        "prepared_rgb": prepared,
        "prepared_rgba": rgba_out,
        "foreground_mask": scaled_a,
    }


def prepare_image(image: np.ndarray) -> Dict[str, np.ndarray]:
    rgba = remove_background(image)
    return center_and_scale_foreground(rgba)
