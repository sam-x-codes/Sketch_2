# Sketch2Shape AI — From a 2D Sketch to a coloured 3D asset

Sketch2Shape turns a rough 2D sketch (drawn in-app or uploaded) plus an optional text prompt into a **coloured, exportable 3D model** (`.glb` + `.obj`).

The previous InstantMesh + SD 1.5 stack is gone. It was slow, needed a second CUDA venv / nvdiffrast, and often shipped grey meshes. This pipeline is built for **Kaggle 2× T4 (16 GB each)** and for higher-quality colour.

---

## Pipeline

```
sketch → clean + ControlNet scribble
      → SDXL-Lightning (8 steps, 1024²) + SDXL scribble ControlNet
      → BiRefNet / U²-Net cutout, centered on white
      → Hunyuan3D-2 mini turbo (geometry)  [fallback: TripoSR]
      → Hunyuan paint if VRAM allows, else 2048² UV / vertex colour bake
      → validate → model.glb + model.obj
```

| Stage | What changed | Why |
|---|---|---|
| Image | SD 1.5 (25 steps, 512²) → **SDXL-Lightning 8-step, 1024²** | Sharper, coloured product shot; 3D can only be as good as this frame |
| Cutout | U²-Net → **BiRefNet** (U²-Net fallback) | Cleaner silhouette, fewer floating blobs |
| 3D | InstantMesh (Zero123++ + LRM, two venvs) → **Hunyuan3D-2 mini turbo** | Modern image-to-mesh, ~6 GB, no nvdiffrast required for shape |
| Colour | Texture often dropped on export → **PBR albedo + vertex colours** | Gradio / Blender show a coloured asset, not a grey clay render |
| Fallback | — | **TripoSR** native RGB vertices if Hunyuan is missing |

Dual GPU: SDXL on `cuda:0`, reconstruction on `cuda:1` when a second T4 is present. Models are unloaded between stages so each T4 stays near 16 GB.

---

## Kaggle (2× T4, ~30 GB RAM)

One cell, GPU notebook, **Internet ON**, Accelerator **GPU T4 x2**:

```bash
# repo added as a Kaggle Dataset (read-only) — the script copies it to /kaggle/working
!bash /kaggle/input/<dataset-name>/scripts/setup.sh

# or, if the repo is already under /kaggle/working:
!bash scripts/setup.sh
```

Add a Kaggle secret named `HF_TOKEN` (Hugging Face token) so gated SDXL weights can download. Re-run the script if the download drops; it is idempotent.

Then:

```python
import os, sys
os.chdir("/kaggle/working/Sketch-main")  # printed at the end of setup
sys.path.insert(0, os.getcwd())

# !python run_pipeline.py --input examples/inputs/chair.png --prompt "oak chair"
from app import build_app
build_app().launch(share=True)
```

```text
GPU    2× Tesla T4, 16 GB each
RAM    ~30 GB  → sequential load (do not keep SDXL and Hunyuan in VRAM together)
Disk   cache HF weights under /kaggle/working/hf_home
```

Quality knobs (already set in the notebook):

| Env | Default | Notes |
|---|---|---|
| `S2S_BACKEND` | `auto` | `hunyuan` or `triposr` to force |
| `S2S_OCTREE` | `384` | `256` on OOM, `512` if it fits |
| `S2S_TEXTURE_PAINT` | `1` | Official Hunyuan paint; set `0` if OOM — colour bake still runs |
| `S2S_TEXTURE_SIZE` | `2048` | Albedo atlas |
| `S2S_LOW_VRAM` | auto on ≤17 GB GPUs | CPU offload for SDXL |
| `S2S_MC_RESOLUTION` | `384` | TripoSR marching cubes (fallback only) |

If Hunyuan paint fails to compile its CUDA rasterizer, ignore it. The colour baker still projects the SDXL product photo onto the mesh and fills the back from the image palette so the GLB is never grey.

---

## Local install (Linux + NVIDIA)

```bash
bash scripts/setup.sh
source .venv/bin/activate
python app.py
# or
python run_pipeline.py --input examples/inputs/chair.png --prompt "oak armchair, rust fabric"
```

One virtualenv. InstantMesh and `.venv-instantmesh` are not used.

---

## Usage

1. Draw or upload a sketch (dark strokes on light).
2. Optional prompt, e.g. `red leather armchair, matte finish`.
3. **Generate 3D** — rotate the coloured preview, download `.glb` / `.obj`.

---

## Hardware

| | Recommended | Kaggle minimum |
|---|---|---|
| GPU | 1× 16 GB+ NVIDIA, or 2× T4 | 2× T4 16 GB |
| RAM | 32 GB | 30 GB (sequential) |
| Disk | ~20 GB weights | `/kaggle/working` + HF cache |

CPU-only and Apple Silicon cannot run stage 4.

---

## Models (local after download)

| Component | Id | Role |
|---|---|---|
| SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | Product image |
| Lightning UNet | `ByteDance/SDXL-Lightning` 8-step | Fast high-res denoising |
| ControlNet | `xinsir/controlnet-scribble-sdxl-1.0` | Sketch contours |
| rembg | BiRefNet / U²-Net | Foreground |
| Hunyuan3D-2 mini turbo | `tencent/Hunyuan3D-2mini` | Mesh |
| Hunyuan paint (optional) | `tencent/Hunyuan3D-2` | UV texture |
| TripoSR (fallback) | `stabilityai/TripoSR` | Coloured mesh |

No paid APIs at inference time.
