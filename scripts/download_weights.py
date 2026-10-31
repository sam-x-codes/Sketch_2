#!/usr/bin/env python3
"""Pre-cache SDXL-Lightning, ControlNet, rembg, TripoSR, Hunyuan3D-2 mini."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sketch2shape.config import (  # noqa: E402
    CONTROLNET_SCRIBBLE_NAME,
    HUNYUAN_MODEL,
    HUNYUAN_SUBFOLDER,
    SDXL_BASE_MODEL,
    SDXL_LIGHTNING_CKPT,
    SDXL_LIGHTNING_REPO,
    TRIPOSR_MODEL,
    get_settings,
)


def main() -> int:
    settings = get_settings()
    if settings["hf_home"]:
        os.environ["HF_HOME"] = settings["hf_home"]

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    print(">>> SDXL-Lightning UNet")
    path = hf_hub_download(SDXL_LIGHTNING_REPO, SDXL_LIGHTNING_CKPT)
    print(f"    {path}")

    print(">>> SDXL base + ControlNet scribble")
    from diffusers import ControlNetModel

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ControlNetModel.from_pretrained(CONTROLNET_SCRIBBLE_NAME, torch_dtype=dtype)
    snapshot_download(SDXL_BASE_MODEL, allow_patterns=["*.json", "*.txt", "*.safetensors", "unet/*", "vae/*", "text_encoder*/*", "tokenizer*/*", "scheduler/*", "model_index.json"])
    print("    cached")

    print(">>> rembg (BiRefNet, then U2-Net fallback)")
    try:
        from rembg import new_session

        try:
            new_session("birefnet-general")
            print("    birefnet-general cached")
        except Exception as exc:
            print(f"    birefnet skipped: {exc}")
            new_session("u2net")
            print("    u2net cached")
    except Exception as exc:
        print(f"    rembg skipped: {exc}")

    print(">>> TripoSR")
    try:
        snapshot_download(TRIPOSR_MODEL)
        print("    cached")
    except Exception as exc:
        print(f"    TripoSR skipped: {exc}")

    print(">>> Hunyuan3D-2 mini turbo")
    try:
        snapshot_download(HUNYUAN_MODEL, allow_patterns=[f"{HUNYUAN_SUBFOLDER}/*", f"{HUNYUAN_SUBFOLDER}/**"])
        print("    cached")
    except Exception as exc:
        print(f"    Hunyuan skipped: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
