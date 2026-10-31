"""Orchestrate the five stages, write artefacts, and emit report.json."""

from __future__ import annotations

import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from .config import STAGES, get_settings
from .export import export_assets
from .generate import generate_image, unload_generator
from .prepare import prepare_image
from .preprocess import preprocess_sketch, to_rgb_uint8
from .reconstruct import reconstruct_mesh


def _now_timer():
    import time

    return time.perf_counter()


def hardware_report() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "gpu": "none",
        "vram_gb": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            info["gpu"] = " + ".join(torch.cuda.get_device_name(i) for i in range(n))
            info["gpu_count"] = n
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def new_run_dir(base: Optional[Path] = None) -> Path:
    settings = get_settings()
    root = Path(base) if base else Path(settings["output_dir"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = root / f"{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def _save_rgb(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    cv2.imwrite(str(path), bgr)


def run_pipeline(
    sketch_image: Any,
    *,
    prompt: str = "",
    seed: int = -1,
    control_scale: float = 0.9,
    output_dir: str = "",
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def log(message: str) -> None:
        print(message, flush=True)
        if progress is not None:
            progress(message)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = new_run_dir()

    timings: Dict[str, float] = {}
    hw = hardware_report()
    raw = to_rgb_uint8(sketch_image)

    log("1/5 Preprocess")
    t0 = _now_timer()
    pre = preprocess_sketch(raw)
    timings["preprocess"] = round(_now_timer() - t0, 3)
    _save_rgb(out / "01_sketch_clean.png", pre["processed"])
    _save_rgb(out / "02_control_map.png", pre["control_map"])

    log("2/5 SDXL-Lightning + ControlNet")
    t0 = _now_timer()
    gen = generate_image(
        pre["control_map"],
        prompt=prompt,
        seed=seed,
        control_scale=control_scale,
    )
    timings["generate"] = round(_now_timer() - t0, 3)
    _save_rgb(out / "03_generated.png", gen["image"])
    unload_generator()

    log("3/5 Background removal")
    t0 = _now_timer()
    prepared = prepare_image(gen["image"])
    timings["prepare"] = round(_now_timer() - t0, 3)
    _save_rgb(out / "04_prepared.png", prepared["prepared_rgb"])
    cv2.imwrite(str(out / "04_prepared_rgba.png"), cv2.cvtColor(prepared["prepared_rgba"], cv2.COLOR_RGBA2BGRA))

    log("4/5 Image-to-3D (Hunyuan3D / TripoSR)")
    t0 = _now_timer()
    recon = reconstruct_mesh(prepared["prepared_rgb"], output_dir=out, seed=seed)
    timings["reconstruct"] = round(_now_timer() - t0, 3)

    log("5/5 Colour bake + export")
    t0 = _now_timer()
    exported = export_assets(
        recon.get("mesh") or recon["mesh_path"],
        out,
        texture_path=recon.get("texture_path"),
    )
    timings["export"] = round(_now_timer() - t0, 3)

    if recon.get("texture_path"):
        src = Path(recon["texture_path"])
        if src.is_file() and src.parent.resolve() != out.resolve():
            shutil.copy2(src, out / src.name)

    report = {
        "prompt": prompt or None,
        "effective_prompt": gen["prompt"],
        "seed": seed,
        "control_scale": control_scale,
        "hardware": hw,
        "timings_seconds": timings,
        "total_seconds": round(sum(timings.values()), 3),
        "backend": recon.get("backend"),
        "mesh": exported["stats"],
        "checks": exported["checks"],
        "valid": exported["valid"],
        "roundtrip_valid": exported["roundtrip_valid"],
        "paths": {
            "run_dir": str(out),
            "glb": exported["glb_path"],
            "obj": exported["obj_path"],
            "sketch": str(out / "01_sketch_clean.png"),
            "control_map": str(out / "02_control_map.png"),
            "generated": str(out / "03_generated.png"),
            "prepared": str(out / "04_prepared.png"),
        },
        "stages": list(STAGES),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "output_dir": str(out),
        "report_json_path": str(report_path),
        "glb_path": exported["glb_path"],
        "obj_path": exported["obj_path"],
        "valid": exported["valid"],
        "roundtrip_valid": exported["roundtrip_valid"],
        "checks": exported["checks"],
        "stages": timings,
        "report": report,
        "sketch_path": str(out / "01_sketch_clean.png"),
        "control_map_path": str(out / "02_control_map.png"),
        "generated_path": str(out / "03_generated.png"),
        "prepared_path": str(out / "04_prepared.png"),
        "hardware": hw,
        "backend": recon.get("backend"),
    }
