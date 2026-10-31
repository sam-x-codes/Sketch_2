"""Stage 4: Hunyuan3D-2 mini (geometry) + colour bake, TripoSR fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh
from PIL import Image

from .colorize import colorize_mesh
from .config import (
    HUNYUAN_MODEL,
    HUNYUAN_PAINT_MODEL,
    HUNYUAN_STEPS,
    HUNYUAN_SUBFOLDER,
    TRIPOSR_MODEL,
    get_settings,
)
from .device import (
    dtype_for,
    empty_cuda,
    enable_fast_cuda,
    reconstruction_device,
)
from .repos import ensure_hunyuan, ensure_triposr


class ReconstructError(RuntimeError):
    """Image-to-3D failed or produced no mesh."""


def _save_input(prepared_image: np.ndarray, output_dir: Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "recon_input.png"
    Image.fromarray(prepared_image).convert("RGB").save(path)
    return path


def _postprocess(mesh: trimesh.Trimesh, image: np.ndarray, texture_size: int) -> trimesh.Trimesh:
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return colorize_mesh(mesh, image, texture_size=texture_size)


def _reconstruct_hunyuan(
    prepared_image: np.ndarray,
    *,
    seed: int,
    octree_resolution: int,
    paint: bool,
) -> trimesh.Trimesh:
    import torch

    if ensure_hunyuan() is None:
        raise ReconstructError("Hunyuan3D-2 is not available")

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    enable_fast_cuda()
    device = reconstruction_device()
    if device.type != "cuda":
        raise ReconstructError("Hunyuan3D-2 needs CUDA")

    dtype = dtype_for(device)
    last_exc: Optional[Exception] = None
    pipeline = None
    for safetensors in (True, False):
        kwargs = dict(
            subfolder=HUNYUAN_SUBFOLDER,
            use_safetensors=safetensors,
            device=str(device),
        )
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN_MODEL, torch_dtype=dtype, **kwargs
            )
            break
        except TypeError:
            try:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    HUNYUAN_MODEL, **kwargs
                )
                break
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc
    if pipeline is None:
        raise ReconstructError(f"Could not load Hunyuan3D-2 mini: {last_exc}")

    try:
        pipeline.enable_flashvdm(topk_mode="merge")
    except Exception:
        try:
            pipeline.enable_flashvdm()
        except Exception:
            pass

    image = Image.fromarray(prepared_image).convert("RGBA")
    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed) if seed is not None and int(seed) >= 0 else 42)

    try:
        out = pipeline(
            image=image,
            num_inference_steps=HUNYUAN_STEPS,
            octree_resolution=int(octree_resolution),
            num_chunks=20000,
            generator=generator,
            output_type="trimesh",
        )
    except torch.cuda.OutOfMemoryError:
        empty_cuda()
        out = pipeline(
            image=image,
            num_inference_steps=HUNYUAN_STEPS,
            octree_resolution=min(256, int(octree_resolution)),
            num_chunks=8000,
            generator=generator,
            output_type="trimesh",
        )
    mesh = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(mesh, trimesh.Trimesh):
        raise ReconstructError(f"Hunyuan3D returned {type(mesh)}")

    del pipeline
    empty_cuda(device)

    if paint:
        mesh = _try_hunyuan_paint(mesh, image, device)
    return mesh


def _try_hunyuan_paint(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    device,
) -> trimesh.Trimesh:
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
    except Exception as exc:
        print(f"Hunyuan paint import skipped: {exc}", flush=True)
        return mesh
    try:
        paint = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_PAINT_MODEL)
        if hasattr(paint, "enable_model_cpu_offload"):
            try:
                paint.enable_model_cpu_offload()
            except Exception:
                pass
        painted = paint(mesh, image=image)
        if isinstance(painted, (list, tuple)):
            painted = painted[0]
        del paint
        empty_cuda(device)
        if isinstance(painted, trimesh.Trimesh):
            print("Hunyuan paint: textured mesh", flush=True)
            return painted
    except Exception as exc:
        print(f"Hunyuan paint skipped ({exc})", flush=True)
        empty_cuda(device)
    return mesh


def _reconstruct_triposr(
    prepared_image: np.ndarray,
    *,
    mc_resolution: int,
    chunk_size: int,
) -> trimesh.Trimesh:
    import torch

    if ensure_triposr() is None:
        raise ReconstructError("TripoSR is not available")

    from tsr.system import TSR

    enable_fast_cuda()
    device = reconstruction_device()
    if device.type != "cuda":
        raise ReconstructError("TripoSR needs CUDA")

    model = TSR.from_pretrained(
        TRIPOSR_MODEL,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(int(chunk_size))
    model.to(device)

    image = Image.fromarray(prepared_image).convert("RGB")
    with torch.no_grad():
        scene_codes = model([image], device=str(device))
        try:
            meshes = model.extract_mesh(
                scene_codes,
                resolution=int(mc_resolution),
                has_vertex_color=True,
            )
        except TypeError:
            meshes = model.extract_mesh(scene_codes, int(mc_resolution))
    mesh = meshes[0]
    del model, scene_codes
    empty_cuda(device)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ReconstructError("TripoSR produced no triangle mesh")
    return mesh


def reconstruct_mesh(
    prepared_image: np.ndarray,
    *,
    output_dir: str | Path,
    seed: int = 42,
    timeout_s: int = 1200,  # kept for call-site compatibility
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    del timeout_s, config_path
    settings = get_settings()
    out = Path(output_dir)
    input_path = _save_input(prepared_image, out)
    backend_pref = (settings["backend"] or "auto").lower()
    texture_size = int(settings["texture_size"])
    errors = []
    mesh: Optional[trimesh.Trimesh] = None
    backend_used = ""

    tried = []
    if backend_pref in {"auto", "hunyuan"}:
        tried.append("hunyuan")
        try:
            print("Reconstruct: Hunyuan3D-2 mini turbo", flush=True)
            mesh = _reconstruct_hunyuan(
                prepared_image,
                seed=seed,
                octree_resolution=int(settings["octree_resolution"]),
                paint=bool(settings["texture_paint"]),
            )
            backend_used = "hunyuan3d-2mini-turbo"
        except Exception as exc:
            errors.append(f"hunyuan: {exc}")
            print(f"Hunyuan3D failed, falling back ({exc})", flush=True)
            empty_cuda()

    if mesh is None and backend_pref in {"auto", "triposr", "hunyuan"}:
        if backend_pref == "hunyuan" and mesh is None:
            pass  # still try TripoSR as last resort
        tried.append("triposr")
        try:
            print("Reconstruct: TripoSR (coloured LRM)", flush=True)
            mesh = _reconstruct_triposr(
                prepared_image,
                mc_resolution=int(settings["mc_resolution"]),
                chunk_size=int(settings["chunk_size"]),
            )
            backend_used = "triposr"
        except Exception as exc:
            errors.append(f"triposr: {exc}")
            empty_cuda()

    if mesh is None:
        raise ReconstructError(
            "Image-to-3D failed. Tried: "
            + "; ".join(errors or tried)
            + ". On Kaggle run the setup cell first (Hunyuan3D-2 + TripoSR)."
        )

    mesh = _postprocess(mesh, prepared_image, texture_size=texture_size)
    raw_path = out / "recon_raw.glb"
    mesh.export(str(raw_path))
    texture_path = out / "albedo.png"
    image = getattr(getattr(mesh, "visual", None), "image", None)
    if image is None:
        material = getattr(getattr(mesh, "visual", None), "material", None)
        image = getattr(material, "baseColorTexture", None) if material is not None else None
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image.convert("RGB").save(texture_path)
    else:
        texture_path = None

    return {
        "mesh": mesh,
        "mesh_path": str(raw_path),
        "texture_path": str(texture_path) if texture_path else None,
        "backend": backend_used,
        "input_path": str(input_path),
    }
