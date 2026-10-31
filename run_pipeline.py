"""Command-line entry point for Sketch2Shape AI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from sketch2shape.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sketch2Shape AI: 2D sketch → textured 3D model"
    )
    parser.add_argument("--input", required=True, help="Sketch image (PNG/JPG/WEBP)")
    parser.add_argument("--prompt", default="", help="Optional appearance prompt")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed (-1 = random)")
    parser.add_argument("--control-scale", type=float, default=1.0, help="ControlNet strength")
    parser.add_argument("--output", default="", help="Run directory (default: outputs/<timestamp>)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: input not found: {path}", file=sys.stderr)
        return 1
    image = Image.open(path)
    result = run_pipeline(
        image,
        prompt=args.prompt,
        seed=args.seed,
        control_scale=args.control_scale,
        output_dir=args.output,
    )
    print("\nSketch2Shape complete")
    print(f"  dir:  {result['output_dir']}")
    print(f"  glb:  {result['glb_path']}")
    print(f"  obj:  {result['obj_path']}")
    print(f"  report: {result['report_json_path']}")
    print(f"  backend: {result.get('backend')}")
    print(f"  valid: {result['valid']}  round-trip: {result['roundtrip_valid']}")
    print("\nTimings (s):")
    for name, seconds in result["stages"].items():
        print(f"  {name}: {seconds:.1f}")
    print(f"  total: {sum(result['stages'].values()):.1f}")
    print("\n" + json.dumps(result.get("report", {}).get("hardware", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
