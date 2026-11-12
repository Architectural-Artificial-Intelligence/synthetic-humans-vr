"""
NOTE: release/ was post-edited after this script ran (human.csv headers renamed with tools/anglicise_human.py;
panas_lib.py/tests/README edited by hand; gender outputs regenerated). Re-running this script would discard
those edits and reproduce the earlier layout.

Build the self-contained Zenodo release folder  release/  from this repository.

Non-destructive: nothing in the repository is modified or removed; everything is
COPIED (or generated) into release/.  Re-running rebuilds release/ from scratch.

    python tools/build_release.py [--out release] [--skip-photos]

What it does
  1. copies the final-batch data, code, tests, results (see KEEP_* below)
  2. writes a de-identified data/human.csv (same schema as the original, so every
     analysis script runs unchanged): timestamp -> participant id P001.., occupation
     coarsened; birth year, sex, education and the PANAS responses are kept
  3. writes photos/ = full equirectangular panoramas, faces/plates blurred where
     detected or marked in analysis/scene_info.csv (projected back from the
     figure view), EXIF/GPS removed; original SHA-256 hashes in photos/SHA256_originals.txt
  4. writes MANIFEST_SHA256.txt for every file in the release
README/CODEBOOK/LICENSE/CITATION/.zenodo.json/reproduce.sh live in tools/release_templates/
and are copied in.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import make_scene_thumbnails as T  # noqa: E402

BATCH = "20261101_103808"
GENDER = "gender_20261108_103819"

ANALYSIS_KEEP = [
    "panas_lib.py", "robust_inference.py", "stability.py", "paired_baseline.py",
    "demographics_table.py", "make_paper_outputs.py", "figure_stats.py",
    "make_main_figures.py", "make_scene_thumbnails.py", "gender_analysis.py",
    "gender_supplement.py", "scene_info.csv",
]
TOP_FILES = ["vlm_prompt.py", "requirements.txt"]
ARCH = {"אדריכלות", "אדריכלית", "ארכיטקטורה", "אדריכל"}
ENGINEERING = {"הנדסת חשמל", "הנדסה אזרחית"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path, ignore=("__pycache__", "*.pyc", ".DS_Store")):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore))


def deidentify_human(src: Path, dst: Path) -> None:
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    cols = df.columns
    df[cols[0]] = [f"P{i + 1:03d}" for i in range(len(df))]          # timestamp -> id
    def occ(v):
        v = v.strip()
        if v in ARCH:
            return "אדריכלות"          # architecture
        if v in ENGINEERING:
            return "הנדסה"             # engineering
        return "אחר" if v else ""      # other / not reported
    df[cols[5]] = df[cols[5]].map(occ)
    df = df.drop(columns=[c for c in cols if c.startswith("עמודה")])   # empty export column
    df = df.rename(columns={cols[0]: "participant_id"})
    df.to_csv(dst, index=False, encoding="utf-8")


def _maps(shape, yaw, pitch, fov, w=T.OUT_W, h=T.OUT_H):
    H, W = shape[:2]
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    xs, ys = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    d = np.stack([xs / f, -ys / f, np.ones_like(xs)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, y = np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    d = d @ Rx.T @ Ry.T
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    return ((lon / (2 * np.pi) + 0.5) * W), ((0.5 - lat / np.pi) * H)


def anonymise_panorama(scene: int, row) -> tuple[Image.Image, int]:
    pano = np.array(Image.open(ROOT / "photos" / f"{scene}.jpg").convert("RGB"))
    H, W = pano.shape[:2]
    view = T.equirect_to_rect(pano, float(row["yaw"]), float(row["pitch"]), float(row["fov"]))
    boxes = []
    if str(row.get("auto_detect", "yes")).lower() != "no":
        boxes += T.detect_faces(view)
    boxes += T.parse_boxes(row["blur_boxes"])
    mask_view = np.zeros(view.shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask_view[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 255
    mx, my = _maps(pano.shape, float(row["yaw"]), float(row["pitch"]), float(row["fov"]))
    sel = mask_view > 0
    mask = np.zeros((H, W), np.uint8)
    mask[my[sel].astype(int) % H, mx[sel].astype(int) % W] = 255
    k = max(9, int(W / 360 * 100 / T.OUT_W * 2) | 1)              # fill sampling gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    blurred = cv2.GaussianBlur(pano, (0, 0), sigmaX=max(8, W / 400))
    out = np.where(mask[..., None] > 0, blurred, pano)
    return Image.fromarray(out), len(boxes)


def build_photos(out: Path) -> None:
    info = T.ensure_info()
    d = out / "photos"
    d.mkdir()
    lines = []
    for _, row in info.iterrows():
        s = int(row["scene"])
        im, n = anonymise_panorama(s, row)
        im.save(d / f"{s}.jpg", quality=95)            # no exif= argument -> no metadata
        lines.append(f"{s}.jpg  original_sha256={sha256(ROOT / 'photos' / f'{s}.jpg')}")
        print(f"photo {s}: {n} blurred regions")
    (d / "SHA256_originals.txt").write_text(
        "# SHA-256 of the ORIGINAL (unblurred, with metadata) panoramas shown to the VLMs;\n"
        "# the same hashes are logged per call in data/runs/*.jsonl.\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release"))
    ap.add_argument("--skip-photos", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # code
    (out / "analysis").mkdir()
    for f in ANALYSIS_KEEP:
        shutil.copy2(ROOT / "analysis" / f, out / "analysis" / f)
    for f in TOP_FILES:
        shutil.copy2(ROOT / f, out / f)
    copy_tree(ROOT / "tests", out / "tests")
    (out / "manuscript").mkdir()
    for f in ("methods_stimuli_procedure.md", "supplementary_gender.md"):
        shutil.copy2(ROOT / "manuscript" / f, out / "manuscript" / f)

    # data
    d = out / "data"
    copy_tree(ROOT / "data" / "runs", d / "runs")
    shutil.copy2(ROOT / "data" / "runs_long.csv", d / "runs_long.csv")
    deidentify_human(ROOT / "data" / "human.csv", d / "human.csv")
    for p in sorted((ROOT / "data").glob("*.csv")):             # per-model 20x10 run means
        if p.name not in ("human.csv", "runs_long.csv"):
            shutil.copy2(p, d / p.name)

    # results (final batch + gender analysis only)
    (out / "results").mkdir()
    copy_tree(ROOT / "results" / BATCH, out / "results" / BATCH)
    copy_tree(ROOT / "results" / GENDER, out / "results" / GENDER)
    shutil.copy2(ROOT / "results" / "demographics_table.csv", out / "results" / "demographics_table.csv")

    # stimuli
    if not a.skip_photos:
        build_photos(out)

    # templates: README, CODEBOOK, licences, CITATION.cff, .zenodo.json, reproduce.sh
    tpl = ROOT / "tools" / "release_templates"
    if tpl.exists():
        for p in tpl.iterdir():
            shutil.copy2(p, out / p.name)

    # demographics_table.py: add English labels for the coarsened occupation categories
    dt = out / "analysis" / "demographics_table.py"
    t = dt.read_text(encoding="utf-8")
    t = t.replace('"הנדסה אזרחית":          "Civil engineering",',
                  '"הנדסה אזרחית":          "Civil engineering",\n    "הנדסה": "Engineering (coarsened)",\n    "אחר": "Other (coarsened)",\n    "אדריכלות": "Architecture",')
    assert "coarsened" in t
    dt.write_text(t, encoding="utf-8")

    # Hebrew item mapping: fix the mapping used for all shipped results (static, no switch).
    pl = out / "analysis" / "panas_lib.py"
    t = pl.read_text(encoding="utf-8")
    a0 = t.index("# Hebrew form keyword -> PANAS item, aligned")
    b0 = t.index("# Model keys that the provider silently")
    block = ("# Hebrew form keyword -> PANAS item (mapping used for all results in this record).\n"
             "HEBREW_ITEM_MAP = {\n"
             '    "להתעניין": "interested", "לחוץ": "nervous", "נרגש": "excited",\n'
             '    "מוטרד": "upset", "חזק": "strong", "אשם": "guilty", "בהלה": "scared",\n'
             '    "עוין": "hostile", "התלהבות": "enthusiastic", "גאה": "proud",\n'
             '    "רוגז": "irritable", "עירני": "alert", "מבוייש": "ashamed",\n'
             '    "שראה": "inspired", "עצבני": "jittery", "נחוש": "determined",\n'
             '    "קשוב": "attentive", "מתח": "distressed", "פעיל": "active",\n'
             '    "פחד": "afraid",\n}\n\n')
    t = t[:a0] + block + t[b0:]
    t = t.replace("import os\n", "", 1)
    pl.write_text(t, encoding="utf-8")
    tp = out / "tests" / "test_panas_lib.py"
    tt = tp.read_text(encoding="utf-8")
    import re as _re
    tt = _re.sub(r"def test_item_mapping_\w+", "def test_item_mapping", tt)
    tt = tt.replace('assert L.HEBREW_ITEM_MAP["עצבני"] == "nervous"', 'assert L.HEBREW_ITEM_MAP["עצבני"] == "jittery"')
    tt = tt.replace('assert L.HEBREW_ITEM_MAP["מתח"] == "jittery"', 'assert L.HEBREW_ITEM_MAP["מתח"] == "distressed"')
    tt = tt.replace('assert L.HEBREW_ITEM_MAP["לחוץ"] == "distressed"', 'assert L.HEBREW_ITEM_MAP["לחוץ"] == "nervous"')
    tp.write_text(tt, encoding="utf-8")
    ts = out / "tests" / "test_scene_thumbnails.py"
    t2 = ts.read_text(encoding="utf-8")
    old2 = "def test_originals_do_carry_gps_so_stripping_matters():\n    assert len(Image.open(T.PHOTOS / \"1.jpg\").getexif().get_ifd(0x8825)) > 0"
    assert old2 in t2
    t2 = t2.replace(old2, "def test_release_photos_carry_no_gps():\n    assert len(Image.open(T.PHOTOS / \"1.jpg\").getexif().get_ifd(0x8825)) == 0")
    ts.write_text(t2, encoding="utf-8")

    # manifest
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.txt")
    (out / "MANIFEST_SHA256.txt").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(out).as_posix()}\n" for p in files))
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"release built: {len(files)} files, {size:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
