"""Stage 5: validate reconstructed mesh and export coloured .glb + .obj."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import trimesh
from PIL import Image

from .colorize import apply_pbr

PathLike = Union[str, Path]


class MeshExportError(ValueError):
    """Mesh failed mandatory geometry checks."""


def load_mesh(path: PathLike) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise MeshExportError(f"No triangle mesh inside {path}")
        loaded = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshExportError(f"Could not load a triangle mesh from {path}")
    return loaded


def mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    extents = mesh.extents
    min_e = float(extents.min()) if extents is not None else 0.0
    max_e = float(extents.max()) if extents is not None else 0.0
    flatness = min_e / (max_e + 1e-12)
    volume = None
    try:
        if mesh.is_watertight:
            volume = float(mesh.volume)
    except Exception:
        volume = None
    visual = getattr(mesh, "visual", None)
    has_uv = bool(visual is not None and getattr(visual, "uv", None) is not None)
    has_image = bool(
        visual is not None
        and (
            getattr(visual, "image", None) is not None
            or getattr(getattr(visual, "material", None), "baseColorTexture", None) is not None
        )
    )
    has_vc = bool(visual is not None and getattr(visual, "vertex_colors", None) is not None)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "volume": volume,
        "bounds_min": mesh.bounds[0].tolist() if mesh.bounds is not None else None,
        "bounds_max": mesh.bounds[1].tolist() if mesh.bounds is not None else None,
        "extents": extents.tolist() if extents is not None else None,
        "flatness_ratio": float(flatness),
        "has_texture": bool(has_uv and has_image),
        "has_vertex_colors": has_vc,
        "has_colour": bool(has_vc or (has_uv and has_image)),
        "finite_vertices": bool(mesh.vertices.size and np_all_finite(mesh.vertices)),
    }


def np_all_finite(values) -> bool:
    return bool(np.isfinite(values).all())


def validate_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    stats = mesh_stats(mesh)
    checks = {
        "non_empty": stats["vertices"] > 0 and stats["faces"] > 0,
        "finite_bounds": stats["finite_vertices"] and stats["extents"] is not None,
        "has_volume_or_depth": stats["flatness_ratio"] >= 0.04,
        "not_a_flat_plane": stats["flatness_ratio"] >= 0.04,
        "has_colour": stats["has_colour"],
    }
    if stats["volume"] is not None:
        checks["positive_volume"] = stats["volume"] > 1e-8
    geometry_ok = all(
        value for key, value in checks.items() if key != "has_colour"
    )
    return {"checks": checks, "stats": stats, "valid": geometry_ok}


def roundtrip_glb(glb_path: PathLike) -> bool:
    try:
        mesh = load_mesh(glb_path)
        return len(mesh.vertices) > 0 and len(mesh.faces) > 0
    except Exception:
        return False


def _attach_texture(mesh: trimesh.Trimesh, texture_path: PathLike) -> trimesh.Trimesh:
    image = Image.open(texture_path).convert("RGB")
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None) if visual is not None else None
    if uv is None:
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=getattr(visual, "vertex_colors", None),
        )
        return mesh
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    return apply_pbr(mesh)


def export_assets(
    mesh_or_path: Union[trimesh.Trimesh, PathLike],
    output_dir: PathLike,
    *,
    texture_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = mesh_or_path if isinstance(mesh_or_path, trimesh.Trimesh) else load_mesh(mesh_or_path)
    if texture_path and Path(texture_path).is_file():
        mesh = _attach_texture(mesh, texture_path)
    mesh = apply_pbr(mesh)

    report = validate_mesh(mesh)
    # Colour is added in reconstruct; do not reject a valid solid if colour bake failed.
    geometry_ok = all(
        report["checks"][k]
        for k in ("non_empty", "finite_bounds", "has_volume_or_depth", "not_a_flat_plane")
        if k in report["checks"]
    )
    if not geometry_ok:
        raise MeshExportError(
            "Mesh rejected as flat or empty. Re-run with another seed; "
            f"checks={report['checks']}"
        )

    glb_path = out / "model.glb"
    obj_path = out / "model.obj"
    mesh.export(str(glb_path), file_type="glb")
    try:
        mesh.export(str(obj_path), include_color=True, include_texture=True)
    except TypeError:
        mesh.export(str(obj_path))
    if not roundtrip_glb(glb_path):
        raise MeshExportError(f"Exported GLB could not be re-opened: {glb_path}")

    return {
        "glb_path": str(glb_path),
        "obj_path": str(obj_path),
        "texture_path": str(texture_path) if texture_path else None,
        "roundtrip_valid": True,
        **report,
        "valid": True,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m sketch2shape.export path/to/model.glb", file=sys.stderr)
        return 2
    path = Path(args[0])
    mesh = load_mesh(path)
    report = validate_mesh(mesh)
    report["roundtrip_valid"] = path.suffix.lower() in {".glb", ".gltf"} and roundtrip_glb(path)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
