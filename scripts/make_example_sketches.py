#!/usr/bin/env python3
"""Draw simple dark-on-light sketches used as sample inputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "inputs"


def canvas() -> np.ndarray:
    img = np.full((512, 512, 3), 250, dtype=np.uint8)
    return img


def stroke(img, pts, thickness=7):
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, (20, 20, 20), thickness, cv2.LINE_AA)


def chair() -> np.ndarray:
    img = canvas()
    stroke(img, [(140, 220), (140, 140), (370, 140), (370, 220)], 8)  # back
    stroke(img, [(120, 220), (390, 220)], 9)  # seat
    stroke(img, [(150, 220), (140, 420)], 8)
    stroke(img, [(360, 220), (370, 420)], 8)
    stroke(img, [(150, 220), (200, 420)], 6)
    stroke(img, [(360, 220), (310, 420)], 6)
    cv2.ellipse(img, (255, 160), (70, 28), 0, 200, 340, (20, 20, 20), 6)
    return img


def mug() -> np.ndarray:
    img = canvas()
    cv2.ellipse(img, (240, 150), (90, 28), 0, 0, 360, (20, 20, 20), 7)
    stroke(img, [(150, 150), (145, 380)], 7)
    stroke(img, [(330, 150), (335, 380)], 7)
    cv2.ellipse(img, (240, 380), (95, 30), 0, 0, 180, (20, 20, 20), 7)
    cv2.ellipse(img, (350, 250), (55, 70), 0, 280, 80, (20, 20, 20), 7)
    return img


def car() -> np.ndarray:
    img = canvas()
    stroke(img, [(60, 300), (110, 300), (150, 220), (330, 220), (390, 300), (460, 300)], 8)
    stroke(img, [(60, 300), (60, 360), (460, 360), (460, 300)], 8)
    cv2.circle(img, (150, 370), 38, (20, 20, 20), 7)
    cv2.circle(img, (370, 370), 38, (20, 20, 20), 7)
    stroke(img, [(170, 230), (170, 300)], 5)
    stroke(img, [(300, 230), (300, 300)], 5)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "examples" / "outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "chair.png"), chair())
    cv2.imwrite(str(OUT / "mug.png"), mug())
    cv2.imwrite(str(OUT / "car.png"), car())
    print(f"Wrote sketches in {OUT}")


if __name__ == "__main__":
    main()
