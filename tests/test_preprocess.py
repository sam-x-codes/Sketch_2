"""CPU-only checks for sketch conditioning and mesh validation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sketch2shape.export import validate_mesh  # noqa: E402
from sketch2shape.preprocess import SketchError, preprocess_sketch  # noqa: E402


def test_blank_sketch_rejected() -> None:
    blank = np.full((256, 256, 3), 250, dtype=np.uint8)
    try:
        preprocess_sketch(blank)
    except SketchError:
        return
    raise SystemExit("expected SketchError for a blank canvas")


def test_chair_sketch() -> None:
    import cv2

    path = ROOT / "examples" / "inputs" / "chair.png"
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    out = preprocess_sketch(image)
    assert out["control_map"].shape == (512, 512, 3)
    assert int(out["sketch"].max()) == 255
    assert int((out["sketch"] > 0).sum()) > 500


def test_flat_mesh_rejected() -> None:
    import trimesh

    mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.001])
    report = validate_mesh(mesh)
    assert report["valid"] is False
    assert report["checks"]["not_a_flat_plane"] is False


def test_vertex_colors_on_box() -> None:
    import trimesh
    from sketch2shape.colorize import project_vertex_colors

    mesh = trimesh.creation.box(extents=[1.0, 1.2, 0.8])
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = (180, 40, 40)
    image[:, 32:] = (40, 80, 200)
    colors = project_vertex_colors(mesh, image)
    assert colors.shape == (len(mesh.vertices), 4)
    assert colors[:, 3].min() == 255
    assert colors[:, :3].max() > 0


if __name__ == "__main__":
    test_blank_sketch_rejected()
    test_chair_sketch()
    test_flat_mesh_rejected()
    test_vertex_colors_on_box()
    print("ok")
