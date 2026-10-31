"""GPU selection and VRAM hygiene for dual-T4 / single-GPU runs."""

from __future__ import annotations

import gc
from typing import Optional, Tuple

import torch


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def gpu_count() -> int:
    return int(torch.cuda.device_count()) if cuda_available() else 0


def vram_gb(index: int = 0) -> float:
    if not cuda_available():
        return 0.0
    props = torch.cuda.get_device_properties(index)
    return round(props.total_memory / (1024**3), 2)


def dtype_for(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def generation_device() -> torch.device:
    if not cuda_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def reconstruction_device() -> torch.device:
    """Prefer the second T4 so SDXL fragmentation on GPU0 cannot OOM 3D."""
    if gpu_count() > 1:
        return torch.device("cuda:1")
    if cuda_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def empty_cuda(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if not cuda_available():
        return
    if device is not None and device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return
    for idx in range(gpu_count()):
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def enable_fast_cuda() -> None:
    if not cuda_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def describe() -> Tuple[str, int, float]:
    if not cuda_available():
        return "cpu", 0, 0.0
    names = [torch.cuda.get_device_name(i) for i in range(gpu_count())]
    return " + ".join(names), gpu_count(), vram_gb(0)
