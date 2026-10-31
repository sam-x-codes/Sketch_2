"""Stage 2: SDXL-Lightning + ControlNet scribble (1024², 8 steps)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .config import (
    CONTROLNET_SCRIBBLE_NAME,
    DEFAULT_PROMPT,
    PROMPT_SUFFIX,
    SD_GUIDANCE,
    SD_IMAGE_SIZE,
    SD_STEPS,
    SDXL_BASE_MODEL,
    SDXL_LIGHTNING_CKPT,
    SDXL_LIGHTNING_REPO,
    get_settings,
)
from .device import dtype_for, empty_cuda, enable_fast_cuda, generation_device, vram_gb

_PIPE = None


def _load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import (
        ControlNetModel,
        EulerDiscreteScheduler,
        StableDiffusionXLControlNetPipeline,
        UNet2DConditionModel,
    )
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    settings = get_settings()
    enable_fast_cuda()
    device = generation_device()
    dtype = dtype_for(device)

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_SCRIBBLE_NAME,
        torch_dtype=dtype,
    )
    unet_config = UNet2DConditionModel.load_config(SDXL_BASE_MODEL, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config)
    ckpt = hf_hub_download(SDXL_LIGHTNING_REPO, SDXL_LIGHTNING_CKPT)
    unet.load_state_dict(load_file(ckpt), strict=True)
    unet.to(dtype=dtype)

    load_kwargs = dict(
        unet=unet,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if dtype == torch.float16:
        load_kwargs["variant"] = "fp16"
    try:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_MODEL, add_watermarker=False, **load_kwargs
        )
    except TypeError:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_MODEL, **load_kwargs
        )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=False)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pass

    tight_vram = device.type == "cuda" and vram_gb(device.index or 0) <= 17
    if (settings["low_vram"] or tight_vram) and device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    _PIPE = pipe
    return pipe


def unload_generator() -> None:
    """Free SDXL VRAM before reconstruction starts."""
    global _PIPE
    if _PIPE is None:
        return
    try:
        _PIPE.to("cpu")
    except Exception:
        pass
    del _PIPE
    _PIPE = None
    empty_cuda()


def generate_image(
    control_map: np.ndarray,
    prompt: str = "",
    *,
    num_inference_steps: Optional[int] = None,
    guidance_scale: float = SD_GUIDANCE,
    seed: int = -1,
    control_scale: float = 0.9,
) -> Dict[str, Any]:
    import torch

    settings = get_settings()
    steps = int(num_inference_steps or settings["sd_steps"] or SD_STEPS)
    user = (prompt or "").strip() or DEFAULT_PROMPT
    effective = f"{user}{PROMPT_SUFFIX}"
    negative = (
        "multiple objects, cropped, out of frame, blurry, watermark, text, logo, "
        "deformed, cluttered background, low quality, overexposed, underexposed, "
        "cartoon, sketch, line art, grayscale, black and white, dull colours"
    )

    control = control_map
    if control.ndim == 2:
        control = np.stack([control] * 3, axis=-1)
    control_pil = (
        Image.fromarray(control.astype(np.uint8))
        .convert("RGB")
        .resize((SD_IMAGE_SIZE, SD_IMAGE_SIZE), Image.LANCZOS)
    )

    pipe = _load_pipeline()
    generator: Optional[torch.Generator] = None
    if seed is not None and int(seed) >= 0:
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    call = dict(
        prompt=effective,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(control_scale),
        generator=generator,
        width=SD_IMAGE_SIZE,
        height=SD_IMAGE_SIZE,
    )
    try:
        result = pipe(image=control_pil, **call)
    except TypeError:
        result = pipe(control_image=control_pil, **call)
    image = result.images[0].convert("RGB")
    return {
        "image": np.asarray(image, dtype=np.uint8),
        "prompt": effective,
        "seed": int(seed),
        "control_scale": float(control_scale),
    }
