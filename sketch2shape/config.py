"""Paths, model ids, and environment-overridable settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXAMPLES_INPUTS = PROJECT_ROOT / "examples" / "inputs"
DEFAULT_EXAMPLES_OUTPUTS = PROJECT_ROOT / "examples" / "outputs"

# ---------------------------------------------------------------------------
# Image generation: SDXL-Lightning (8-step) + ControlNet Scribble SDXL
# ---------------------------------------------------------------------------
SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_CKPT = "sdxl_lightning_8step_unet.safetensors"
CONTROLNET_SCRIBBLE_NAME = "xinsir/controlnet-scribble-sdxl-1.0"

DEFAULT_PROMPT = "a single everyday object, studio product photo, rich colour, sharp material detail"
PROMPT_SUFFIX = (
    ", single object, pure white background, centered, photorealistic product render, "
    "highly detailed materials, accurate colours, 8k, studio lighting"
)

INPUT_TARGET_SIZE = 512
PREPARE_SIZE = 512
SD_IMAGE_SIZE = 1024
SD_STEPS = 8
SD_GUIDANCE = 1.2  # Lightning + ControlNet: 0 is too empty, 1.0–1.5 is sharp

# ---------------------------------------------------------------------------
# 3D reconstruction
# Hunyuan3D-2 mini turbo: modern image-to-mesh (geometry)
# TripoSR: lightweight coloured fallback (native vertex RGB)
# ---------------------------------------------------------------------------
HUNYUAN_GIT = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
HUNYUAN_MODEL = "tencent/Hunyuan3D-2mini"
HUNYUAN_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"
HUNYUAN_PAINT_MODEL = "tencent/Hunyuan3D-2"
HUNYUAN_STEPS = 5
HUNYUAN_OCTREE = 384

TRIPOSR_GIT = "https://github.com/stabilityai/TripoSR.git"
TRIPOSR_MODEL = "stabilityai/TripoSR"
TRIPOSR_CHUNK_SIZE = 4096
TRIPOSR_MC_RESOLUTION = 384

TEXTURE_SIZE = 2048
FOREGROUND_RATIO = 0.82
STAGES = ("preprocess", "generate", "prepare", "reconstruct", "export")


def apply_kaggle_defaults() -> None:
    """Fill paths when running a Kaggle kernel so setup.sh + Python cells agree."""
    if not Path("/kaggle").exists():
        return
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf_home")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("S2S_OUTPUT_DIR", "/kaggle/working/outputs")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("S2S_BACKEND", "auto")
    os.environ.setdefault("S2S_TEXTURE_PAINT", "1")
    os.environ.setdefault("S2S_OCTREE", "384")
    os.environ.setdefault("S2S_MC_RESOLUTION", "384")
    os.environ.setdefault("S2S_TEXTURE_SIZE", "2048")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["S2S_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)


apply_kaggle_defaults()


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def get_settings() -> Dict[str, Any]:
    output = Path(os.getenv("S2S_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).expanduser()
    hf_home: Optional[str] = os.getenv("HF_HOME")
    return {
        "low_vram": env_flag("S2S_LOW_VRAM"),
        "output_dir": output,
        "hf_home": hf_home,
        "backend": env_str("S2S_BACKEND", "auto"),
        "texture_paint": env_flag("S2S_TEXTURE_PAINT", "1"),
        "mc_resolution": env_int("S2S_MC_RESOLUTION", TRIPOSR_MC_RESOLUTION),
        "chunk_size": env_int("S2S_CHUNK_SIZE", TRIPOSR_CHUNK_SIZE),
        "octree_resolution": env_int("S2S_OCTREE", HUNYUAN_OCTREE),
        "texture_size": env_int("S2S_TEXTURE_SIZE", TEXTURE_SIZE),
        "sd_steps": env_int("S2S_SD_STEPS", SD_STEPS),
    }
