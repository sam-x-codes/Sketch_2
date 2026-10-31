"""Gradio UI: draw or upload a sketch, preview and download a 3D asset."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import gradio as gr

from sketch2shape.pipeline import run_pipeline
from sketch2shape.preprocess import SketchError, to_rgb_uint8

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT_DIR / "examples" / "inputs"

EXAMPLE_PRESETS = {
    "chair": {
        "image": str(EXAMPLES_DIR / "chair.png"),
        "prompt": "modern ergonomic office chair, mesh back, chrome base, studio product photo",
        "seed": 42,
        "scale": 1.0,
    },
    "mug": {
        "image": str(EXAMPLES_DIR / "mug.png"),
        "prompt": "glazed ceramic coffee mug, teal finish, soft studio shadows",
        "seed": 123,
        "scale": 1.1,
    },
    "car": {
        "image": str(EXAMPLES_DIR / "car.png"),
        "prompt": "sleek concept sports car, metallic red, aerodynamic, studio render",
        "seed": 999,
        "scale": 1.0,
    },
}

STYLE_PROMPTS = {
    "Armchair": "vintage brown leather armchair, walnut frame, brass rivets, studio lighting",
    "Ceramic Mug": "minimalist ceramic coffee mug, glazed teal, soft studio shadows",
    "Sports Car": "futuristic concept car, metallic crimson finish, studio render",
    "Sneaker": "modern running shoe, knit mesh upper, cushioned sole, product photo",
    "Drone": "compact quadcopter drone, carbon fiber, glowing blue LED accents",
}

# ---------------------------------------------------------------------------
# CSS — clean warm theme inspired by Meshy.ai
# ---------------------------------------------------------------------------
CSS = """
/* ── Global ── */
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 40px 24px 32px;
    margin-bottom: 8px;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 8px 0;
    color: #1e293b;
    line-height: 1.15;
}
.dark .hero h1 { color: #f1f5f9; }

.hero p {
    font-size: 1.05rem;
    color: #64748b;
    margin: 0 auto 24px;
    max-width: 580px;
    line-height: 1.6;
}
.dark .hero p { color: #94a3b8; }

.hero-tags {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.tag {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}
.dark .tag {
    background: rgba(255,255,255,0.06);
    color: #cbd5e1;
    border-color: rgba(255,255,255,0.1);
}
.tag-accent {
    background: #b45309;
    color: #fff;
    border-color: transparent;
}
.dark .tag-accent {
    background: #d97706;
    color: #fff;
}

.hero-steps {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.88rem;
    color: #64748b;
}
.dark .hero-steps { color: #94a3b8; }

.step-dot {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #b45309;
    color: #fff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 4px;
}
.dark .step-dot { background: #d97706; }

.step-arrow { color: #cbd5e1; font-weight: bold; }
.dark .step-arrow { color: #475569; }

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
    padding-bottom: 0;
}
.dark .section-label { color: #64748b; }

/* ── Generate button ── */
.gen-btn {
    background: #b45309 !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 12px rgba(180, 83, 9, 0.25) !important;
    transition: all 0.15s ease !important;
}
.gen-btn:hover {
    background: #92400e !important;
    box-shadow: 0 4px 20px rgba(180, 83, 9, 0.4) !important;
    transform: translateY(-1px) !important;
}
.dark .gen-btn {
    background: #d97706 !important;
}
.dark .gen-btn:hover {
    background: #b45309 !important;
}

/* ── Style chips ── */
.style-chip {
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    border-radius: 999px !important;
    background: transparent !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    min-width: auto !important;
}
.dark .style-chip {
    border-color: rgba(255,255,255,0.12) !important;
    color: #94a3b8 !important;
}
.style-chip:hover {
    background: #b45309 !important;
    color: #fff !important;
    border-color: #b45309 !important;
}
.dark .style-chip:hover {
    background: #d97706 !important;
    border-color: #d97706 !important;
}

/* ── Preset card buttons ── */
.preset-btn {
    border: 1px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    background: transparent !important;
    color: #475569 !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
.dark .preset-btn {
    border-color: rgba(255,255,255,0.1) !important;
    color: #94a3b8 !important;
}
.preset-btn:hover {
    border-color: #b45309 !important;
    color: #b45309 !important;
    border-style: solid !important;
}
.dark .preset-btn:hover {
    border-color: #d97706 !important;
    color: #d97706 !important;
}

/* ── Viewport frame ── */
.viewport-bar {
    padding: 8px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8fafc;
}
.dark .viewport-bar {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.06);
    color: #94a3b8;
}
.viewport-bar span { font-weight: 400; opacity: 0.7; font-size: 0.72rem; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 28px 16px 14px;
    margin-top: 16px;
    border-top: 1px solid #e2e8f0;
    color: #94a3b8;
    font-size: 0.82rem;
}
.dark .app-footer { border-color: rgba(255,255,255,0.06); }
.app-footer-sub { font-size: 0.75rem; opacity: 0.7; margin-top: 4px; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_pixels(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(value.get(k) is not None
                   for k in ("composite", "image", "value", "background"))
    return True


def _load_preset(name: str):
    p = EXAMPLE_PRESETS.get(name, {})
    return p.get("image"), p.get("prompt", ""), p.get("seed", 42), p.get("scale", 1.0)


def _status_ready():
    return "**Ready.** Draw a sketch or upload an image, then hit Generate."


def _reset_all():
    return None, None, "", 42, 0.9, _status_ready(), None, None, None


# ---------------------------------------------------------------------------
# Core generation (unchanged pipeline logic)
# ---------------------------------------------------------------------------
def generate_3d(
    sketch: Any,
    uploaded: Any,
    prompt: str,
    seed: float,
    control_scale: float,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        if _has_pixels(uploaded):
            image = to_rgb_uint8(uploaded)
        elif _has_pixels(sketch):
            image = to_rgb_uint8(sketch)
        else:
            raise SketchError("Draw a sketch or upload an image first.")

        seed_i = int(seed)
        msgs: list[str] = []

        def on_progress(message: str) -> None:
            msgs.append(message)
            try:
                progress(len(msgs) / 5.0, desc=message)
            except Exception:
                pass

        result = run_pipeline(
            image,
            prompt=prompt or "",
            seed=seed_i,
            control_scale=float(control_scale),
            progress=on_progress,
        )

        stages = result["stages"]
        total = sum(stages.values())
        hw = result.get("hardware") or {}

        status = (
            f"**Done in {total:.1f}s** — "
            f"Preprocess {stages.get('preprocess',0):.1f}s · "
            f"SD+ControlNet {stages.get('generate',0):.1f}s · "
            f"BG Removal {stages.get('prepare',0):.1f}s · "
            f"3D {stages.get('reconstruct',0):.1f}s · "
            f"Export {stages.get('export',0):.1f}s  \n"
            f"GPU: {hw.get('gpu','—')} · VRAM: {hw.get('vram_gb','—')} GB · "
            f"backend: {result.get('backend','—')}"
        )

        return (
            status,
            result["sketch_path"],
            result["control_map_path"],
            result["generated_path"],
            result["prepared_path"],
            result["glb_path"],
            result["glb_path"],
            result["obj_path"],
            json.dumps(result.get("report", {}), indent=2),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )

    with gr.Blocks(title="Sketch2Shape AI", theme=theme, css=CSS) as demo:

        # ── Hero ──
        gr.HTML("""
        <div class="hero">
            <div class="hero-tags">
                <span class="tag tag-accent">TechZephyr 4.0</span>
                <span class="tag">ControlNet SDXL</span>
                <span class="tag">SDXL-Lightning</span>
                <span class="tag">Hunyuan3D-2</span>
                <span class="tag">Coloured GLB + OBJ</span>
            </div>
            <h1>Generate production-ready 3D assets<br>from a rough sketch.</h1>
            <p>
                Draw or upload a 2D sketch and get a textured, exportable 3D model.
                Drop it into your engine, renderer, or DCC tool.
            </p>
            <div class="hero-steps">
                <span><span class="step-dot">1</span> Sketch</span>
                <span class="step-arrow">→</span>
                <span><span class="step-dot">2</span> Style &amp; Tune</span>
                <span class="step-arrow">→</span>
                <span><span class="step-dot">3</span> 3D Export</span>
            </div>
        </div>
        """)

        # ── Main 3-column layout ──
        with gr.Row(equal_height=False):

            # ─── Left: Input ───
            with gr.Column(scale=5):
                gr.HTML('<div class="section-label">INPUT</div>')

                with gr.Tabs():
                    with gr.Tab("Draw"):
                        sketch = gr.Sketchpad(
                            label="Sketchpad",
                            type="numpy",
                            height=400,
                        )
                    with gr.Tab("Upload"):
                        uploaded = gr.Image(
                            label="Upload image",
                            sources=["upload", "clipboard"],
                            type="numpy",
                            height=400,
                        )

                gr.HTML('<div class="section-label" style="margin-top:12px">QUICK START — sample sketches</div>')
                with gr.Row():
                    btn_chair = gr.Button("Chair", size="sm", elem_classes=["preset-btn"])
                    btn_mug   = gr.Button("Mug",   size="sm", elem_classes=["preset-btn"])
                    btn_car   = gr.Button("Car",   size="sm", elem_classes=["preset-btn"])

            # ─── Center: Settings ───
            with gr.Column(scale=3, min_width=260):
                gr.HTML('<div class="section-label">STYLE</div>')

                prompt = gr.Textbox(
                    label="Prompt (optional)",
                    placeholder="red leather armchair, matte finish …",
                    lines=2,
                    max_lines=3,
                )

                with gr.Row():
                    chip_btns = []
                    for name in STYLE_PROMPTS:
                        b = gr.Button(name, size="sm", elem_classes=["style-chip"])
                        chip_btns.append((name, b))

                gr.HTML('<div class="section-label" style="margin-top:16px">PARAMETERS</div>')

                control_scale = gr.Slider(
                    label="Sketch adherence",
                    minimum=0.4, maximum=1.5, value=0.9, step=0.05,
                    info="Low = creative · High = strict contours",
                )
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0, scale=3)
                    btn_rand = gr.Button("🎲", size="sm", scale=1, min_width=48)

                generate_btn = gr.Button(
                    "Generate 3D →",
                    variant="primary",
                    size="lg",
                    elem_classes=["gen-btn"],
                )
                reset_btn = gr.Button("Reset", variant="secondary", size="sm")

            # ─── Right: Output ───
            with gr.Column(scale=5):
                gr.HTML('<div class="section-label">OUTPUT</div>')

                status = gr.Markdown(_status_ready())

                gr.HTML("""
                <div class="viewport-bar">
                    3D Preview
                    <span>Drag to rotate · Scroll to zoom</span>
                </div>
                """)
                model = gr.Model3D(label="3D Preview", height=400)

                with gr.Row():
                    glb = gr.File(label="Download .glb")
                    obj = gr.File(label="Download .obj")

        # ── Pipeline stages (collapsed) ──
        with gr.Accordion("Pipeline stages", open=False):
            with gr.Row():
                img_clean   = gr.Image(label="1 — Cleaned sketch",   interactive=False)
                img_control = gr.Image(label="2 — Control map",      interactive=False)
                img_gen     = gr.Image(label="3 — Generated image",  interactive=False)
                img_prep    = gr.Image(label="4 — Prepared (no bg)", interactive=False)
            with gr.Accordion("report.json", open=False):
                report = gr.Code(label="Run report", language="json")

        # ── Footer ──
        gr.HTML("""
        <div class="app-footer">
            <strong>TechZephyr 4.0</strong> · IIT Bhubaneswar · Computer Vision &amp; Generative AI
            <div class="app-footer-sub">
                SDXL-Lightning + ControlNet · Hunyuan3D-2 mini / TripoSR · Coloured GLB export
            </div>
        </div>
        """)

        # ── Wiring ──

        # Preset sketch buttons
        btn_chair.click(fn=lambda: _load_preset("chair"), outputs=[uploaded, prompt, seed, control_scale])
        btn_mug.click(fn=lambda: _load_preset("mug"),     outputs=[uploaded, prompt, seed, control_scale])
        btn_car.click(fn=lambda: _load_preset("car"),     outputs=[uploaded, prompt, seed, control_scale])

        # Style chips → fill prompt
        for name, btn in chip_btns:
            text = STYLE_PROMPTS[name]
            btn.click(fn=lambda t=text: t, outputs=[prompt])

        # Randomize seed
        btn_rand.click(fn=lambda: random.randint(1, 999_999), outputs=[seed])

        # Reset
        reset_btn.click(
            fn=_reset_all,
            outputs=[sketch, uploaded, prompt, seed, control_scale, status, model, glb, obj],
        )

        # Generate
        generate_btn.click(
            fn=generate_3d,
            inputs=[sketch, uploaded, prompt, seed, control_scale],
            outputs=[status, img_clean, img_control, img_gen, img_prep, model, glb, obj, report],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    is_colab = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
    share = is_colab or "--share" in sys.argv
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=share)
