"""
Scene thumbnails for Figure 1: rectilinear views cut from the 360-degree panoramas,
anonymised, and saved WITHOUT any EXIF/XMP/GPS metadata.

Inputs : photos/<n>.jpg (equirectangular, tracked originals carry GPS in EXIF)
         analysis/scene_info.csv  (edit this file; the figure regenerates)
Outputs: <out>/scene_<n>.jpg (900 x 600, metadata-free)

scene_info.csv columns
  scene        1..10
  short_name   label under the thumbnail
  yaw, pitch   view direction in degrees (yaw 0 = centre of the panorama, +right;
               pitch +up)
  fov          horizontal field of view in degrees
  auto_detect  yes/no: also blur Haar face detections (no where it gave false positives)
  blur_boxes   extra boxes to blur, "x0,y0,x1,y1;x0,y0,x1,y1" in OUTPUT pixels
               (faces the detector misses, licence plates, ...)

Usage: python analysis/make_scene_thumbnails.py [--out DIR] [--no-detect] [--preview]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "photos"
INFO = Path(__file__).resolve().parent / "scene_info.csv"
OUT_W, OUT_H = 900, 600

# Draft values (yaw/pitch/fov chosen from the panoramas; edit scene_info.csv).
DEFAULT_INFO = [
    (1, "Street kiosk", 0, 4, 95, "yes", ""),
    (2, "Palm plaza", 43, 4, 100, "yes", ""),
    (3, "Overgrown yard", 5, 0, 100, "yes", ""),
    (4, "Parked street", -54, 0, 100, "yes", ""),
    (5, "Arcade", 18, 0, 100, "yes", ""),
    (6, "Tree path", 0, 0, 100, "yes", ""),
    (7, "Beach", -18, 6, 100, "yes", ""),
    (8, "Cafe", 36, 0, 90, "yes", ""),
    (9, "Service car park", -29, 0, 100, "yes", ""),
    (10, "Parking hall", 0, -3, 100, "yes", ""),
]


def ensure_info() -> pd.DataFrame:
    if not INFO.exists():
        pd.DataFrame(DEFAULT_INFO, columns=["scene", "short_name", "yaw", "pitch", "fov", "auto_detect", "blur_boxes"]) \
            .to_csv(INFO, index=False)
    return pd.read_csv(INFO, keep_default_na=False)


def equirect_to_rect(pano: np.ndarray, yaw: float, pitch: float, fov: float,
                     w: int = OUT_W, h: int = OUT_H) -> np.ndarray:
    """Rectilinear (pinhole) view from an equirectangular panorama."""
    H, W = pano.shape[:2]
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    xs, ys = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    d = np.stack([xs / f, -ys / f, np.ones_like(xs)], axis=-1)           # x right, y up, z forward
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, y = np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    d = d @ Rx.T @ Ry.T
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * H).astype(np.float32)
    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def parse_boxes(s: str) -> list[tuple[int, int, int, int]]:
    boxes = []
    for part in str(s).split(";"):
        part = part.strip()
        if part:
            x0, y0, x1, y1 = (int(float(v)) for v in part.split(","))
            boxes.append((x0, y0, x1, y1))
    return boxes


def blur_box(img: np.ndarray, box, strength: int = 0) -> None:
    x0, y0, x1, y1 = box
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    k = strength or max(15, (min(x1 - x0, y1 - y0) // 2) | 1)
    roi[:] = cv2.GaussianBlur(roi, (k | 1, k | 1), 0)
    roi[:] = cv2.GaussianBlur(roi, (k | 1, k | 1), 0)


def detect_faces(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Haar frontal + profile detections (many small panoramic faces will be missed;
    manual boxes in scene_info.csv cover those)."""
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    boxes = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        for (x, y, w, h) in casc.detectMultiScale(gray, 1.1, 5, minSize=(14, 14), maxSize=(110, 110)):
            pad = int(0.25 * w)
            boxes.append((x - pad, y - pad, x + w + pad, y + h + pad))
    return boxes


def make_thumbnail(scene: int, row, detect: bool = True) -> tuple[Image.Image, int]:
    pano = np.array(Image.open(PHOTOS / f"{scene}.jpg").convert("RGB"))
    view = equirect_to_rect(pano, float(row["yaw"]), float(row["pitch"]), float(row["fov"]))
    n = 0
    if detect and str(row.get("auto_detect", "yes")).lower() != "no":
        for b in detect_faces(view):
            blur_box(view, b); n += 1
    for b in parse_boxes(row["blur_boxes"]):
        blur_box(view, b); n += 1
    return Image.fromarray(view), n          # a new PIL image carries no EXIF/XMP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "scene_thumbnails"))
    ap.add_argument("--no-detect", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = ensure_info()
    for _, row in info.iterrows():
        im, n = make_thumbnail(int(row["scene"]), row, detect=not args.no_detect)
        im.save(out / f"scene_{int(row['scene'])}.jpg", quality=92)     # no exif argument -> no metadata
        print(f"scene {int(row['scene']):2d} {row['short_name']:<18s} blurred regions: {n}")


if __name__ == "__main__":
    main()
