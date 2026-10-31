"""Bake full-object colour onto a mesh: vertex colours + UV albedo texture."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        image = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)


def _palette(image: np.ndarray, k: int = 6) -> np.ndarray:
    pixels = image.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 4000:
        rng = np.random.default_rng(0)
        pixels = pixels[rng.choice(len(pixels), 4000, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _ok, _labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    return np.clip(centers, 0, 255).astype(np.float32)


def _nearest_palette(colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
    d = ((colors[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    return palette[d.argmin(axis=1)]


def project_vertex_colors(
    mesh: trimesh.Trimesh,
    image: np.ndarray,
) -> np.ndarray:
    """Front-project the product photo, fill the back from the image palette."""
    image = _as_rgb(image)
    h, w = image.shape[:2]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = (verts.max(axis=0) + verts.min(axis=0)) * 0.5
    local = verts - center
    extent = np.abs(local).max() + 1e-8
    local = local / extent

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception:
        mesh.fix_normals()
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms

    # Auto-pick the viewing axis that most vertices face (Hunyuan/TripoSR vary).
    axes = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
    scores = [float(np.mean(np.clip(normals @ axis, 0, 1))) for axis in axes]
    view = axes[int(np.argmax(scores))]

    # Build a camera plane orthonormal to `view`.
    helper = np.array([0.0, 1.0, 0.0]) if abs(view[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, view)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(view, right)

    u = local @ right
    v = local @ up
    facing = np.clip(normals @ view, 0.0, 1.0)

    xs = np.clip(((u + 1.05) / 2.1 * (w - 1)).astype(np.int32), 0, w - 1)
    ys = np.clip(((1.05 - v) / 2.1 * (h - 1)).astype(np.int32), 0, h - 1)
    sampled = image[ys, xs].astype(np.float32)

    palette = _palette(image)
    mean_color = image.reshape(-1, 3).mean(axis=0)
    back = _nearest_palette(sampled, palette)
    # Mix sampled photo (front) with palette-matched fill (sides/back).
    weight = np.power(facing, 1.35)[:, None]
    colors = sampled * weight + (0.55 * back + 0.45 * mean_color) * (1.0 - weight)
    rgba = np.concatenate(
        [np.clip(colors, 0, 255), np.full((len(colors), 1), 255.0)],
        axis=1,
    ).astype(np.uint8)
    return rgba


def _scatter_uv_texture(
    faces: np.ndarray,
    uvs: np.ndarray,
    vertex_colors: np.ndarray,
    size: int,
) -> np.ndarray:
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    pts = np.empty((3, 2), dtype=np.int32)
    for face in faces:
        uv = uvs[face]
        pts[:, 0] = np.clip(uv[:, 0] * (size - 1), 0, size - 1)
        pts[:, 1] = np.clip((1.0 - uv[:, 1]) * (size - 1), 0, size - 1)
        color = tuple(int(c) for c in vertex_colors[face].mean(axis=0)[:3])
        cv2.fillConvexPoly(tex, pts, color, lineType=cv2.LINE_AA)
    return tex


def _inpaint_holes(tex: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY)
    hole = (gray == 0).astype(np.uint8) * 255
    if int(hole.sum()) == 0:
        return tex
    hole = cv2.dilate(hole, np.ones((3, 3), np.uint8), iterations=1)
    filled = cv2.inpaint(tex, hole, 4, cv2.INPAINT_TELEA)
    filled = cv2.bilateralFilter(filled, 5, 40, 40)
    return filled


def bake_uv_texture(
    mesh: trimesh.Trimesh,
    vertex_colors: np.ndarray,
    size: int = 2048,
) -> Optional[Tuple[trimesh.Trimesh, Image.Image]]:
    try:
        import xatlas
    except Exception:
        return None
    try:
        vmapping, indices, uvs = xatlas.parametrize(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.uint32),
        )
    except Exception:
        return None
    new_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[vmapping],
        faces=np.asarray(indices, dtype=np.int64),
        process=False,
    )
    colors = vertex_colors[vmapping]
    uv = np.asarray(uvs, dtype=np.float64)
    tex = _scatter_uv_texture(new_mesh.faces, uv, colors, size=size)
    tex = _inpaint_holes(tex)
    image = Image.fromarray(tex, mode="RGB")
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    return new_mesh, image


def apply_pbr(mesh: trimesh.Trimesh, roughness: float = 0.42, metallic: float = 0.04) -> trimesh.Trimesh:
    visual = getattr(mesh, "visual", None)
    image = None
    uv = None
    if visual is not None:
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        image = getattr(visual, "image", None)
        if image is None and material is not None:
            image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is not None and uv is not None:
        pbr = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[255, 255, 255, 255],
            baseColorTexture=image,
            metallicFactor=float(metallic),
            roughnessFactor=float(roughness),
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr, image=image)
    elif visual is not None and getattr(visual, "vertex_colors", None) is not None:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=visual.vertex_colors)
    return mesh


def colorize_mesh(
    mesh: trimesh.Trimesh,
    image: np.ndarray,
    *,
    texture_size: int = 2048,
    force: bool = False,
) -> trimesh.Trimesh:
    """Ensure the mesh has colour. Prefer existing textures; otherwise bake them."""
    visual = getattr(mesh, "visual", None)
    has_uv = (
        visual is not None
        and getattr(visual, "uv", None) is not None
        and (
            getattr(visual, "image", None) is not None
            or getattr(getattr(visual, "material", None), "baseColorTexture", None) is not None
        )
    )
    has_vc = visual is not None and getattr(visual, "vertex_colors", None) is not None
    if has_uv and not force:
        return apply_pbr(mesh)
    if has_vc and not force:
        colors = np.asarray(visual.vertex_colors)
    else:
        colors = project_vertex_colors(mesh, image)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

    baked = bake_uv_texture(mesh, colors, size=texture_size)
    if baked is None:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        return mesh
    mesh, _tex = baked
    return apply_pbr(mesh)
