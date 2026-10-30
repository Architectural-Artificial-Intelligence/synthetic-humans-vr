"""
Shared loading + metric helpers for the PANAS human-vs-VLM analyses.

Units of analysis (state these in the paper):
  cell   = one (image, item) pair: 10 images x 20 items = 200 cells.
           Human value = mean over participants; model value = mean over runs.
  scale  = PA / NA subscale mean per image (10 images x 2 subscales).
All metrics below take arrays shaped (n_images, n_items) in PANAS_ITEMS order.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
N_IMAGES = 10

PANAS_ITEMS = [
    "interested", "excited", "strong", "enthusiastic", "proud",
    "alert", "inspired", "determined", "attentive", "active",
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid",
]
PA_ITEMS = PANAS_ITEMS[:10]
NA_ITEMS = PANAS_ITEMS[10:]
PA_IDX = np.arange(0, 10)
NA_IDX = np.arange(10, 20)

# Hebrew form keyword -> PANAS item, aligned with the validated Hebrew PANAS
# (Ben-Zur, 2002): עצבני/ת = nervous, מתוח/ה (here "תחושת מתח") = jittery.
# The form's "לחוץ/ה" is not a Ben-Zur item; by elimination it stands for
# distressed (Ben-Zur: נתון/ה במצוקה). The earlier analysis mapped
# לחוץ->nervous, עצבני->jittery, מתח->distressed (a permutation within NA);
# set PANAS_ITEM_MAP=legacy to reproduce it.
HEBREW_ITEM_MAP = {
    "להתעניין": "interested", "לחוץ": "distressed", "נרגש": "excited",
    "מוטרד": "upset", "חזק": "strong", "אשם": "guilty", "בהלה": "scared",
    "עוין": "hostile", "התלהבות": "enthusiastic", "גאה": "proud",
    "רוגז": "irritable", "עירני": "alert", "מבוייש": "ashamed",
    "שראה": "inspired", "עצבני": "nervous", "נחוש": "determined",
    "קשוב": "attentive", "מתח": "jittery", "פעיל": "active",
    "פחד": "afraid",
}
if os.getenv("PANAS_ITEM_MAP") == "legacy":
    HEBREW_ITEM_MAP.update({"לחוץ": "nervous", "עצבני": "jittery", "מתח": "distressed"})

# Model keys that the provider silently served with another model (the
# returned model ID differs and the outputs are identical to that model's).
# They are not distinct models and are excluded from all analyses.
ALIAS_DUPLICATES = {
    "gemini-3.5-flash": "gemini-3.6-flash",
    "gemini-3.7-flash": "gemini-3.8-flash",
}
LIKERT_MAP = {"כלל לא": 1, "מעט": 2, "במידה בינונית": 3,
              "במידה רבה": 4, "רבה מאוד": 5}
SCALE_RANGE = 4.0     # 1..5 Likert -> range used to normalise MAE / RMSE


# ── Human data ─────────────────────────────────────────────────────────────
def _keyword(col: str):
    m = re.search(r"\[עד כמה החלל [^[]+\]", col)
    if not m:
        return None
    return next((kw for kw in HEBREW_ITEM_MAP if kw in m.group(0)), None)


def _image_index(col: str) -> int:
    m = re.search(r"\.(\d+)$", col)
    return int(m.group(1)) if m else 0


def load_human_raw(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (wide raw frame after dropping blank PANAS rows, accounting dict)."""
    df = pd.read_csv(path or DATA_DIR / "human.csv")
    panas_cols = df.columns[6:].tolist()
    acc = {"rows_in_csv": len(df)}
    blank = df[panas_cols].isna().all(axis=1)
    acc["rows_all_panas_blank"] = int(blank.sum())
    df = df[~blank].reset_index(drop=True)
    acc["rows_after_blank_drop"] = len(df)
    ts = df.iloc[:, 0].astype(str)
    acc["duplicate_timestamps"] = int(ts.duplicated().sum())
    acc["final_N"] = len(df)
    return df, acc


def human_array(df: pd.DataFrame) -> np.ndarray:
    """(participants, images, items) array in PANAS_ITEMS order; NaN = missing.
    NOT imputed: callers choose how to treat NaN."""
    panas_cols = df.columns[6:].tolist()
    idx = {it: k for k, it in enumerate(PANAS_ITEMS)}
    arr = np.full((len(df), N_IMAGES, len(PANAS_ITEMS)), np.nan)
    for col in panas_cols:
        kw = _keyword(col)
        if kw is None:
            continue
        img, item = _image_index(col), idx[HEBREW_ITEM_MAP[kw]]
        vals = df[col].map(lambda v: LIKERT_MAP.get(str(v).strip(), np.nan)
                           if pd.notna(v) else np.nan)
        arr[:, img, item] = vals.values
    return arr


def impute_cell_mean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN by the image x item mean over participants (original method)."""
    out = arr.copy()
    cell_mean = np.nanmean(arr, axis=0)
    miss = np.isnan(out)
    out[miss] = np.broadcast_to(cell_mean, out.shape)[miss]
    return out


# ── Model data ─────────────────────────────────────────────────────────────
def drop_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Remove model keys listed in ALIAS_DUPLICATES (column 'model')."""
    return df[~df["model"].isin(ALIAS_DUPLICATES)].copy()


def load_runs_long(path: Path | None = None, tag: str | None = "main") -> pd.DataFrame:
    df = drop_aliases(pd.read_csv(path or DATA_DIR / "runs_long.csv"))
    return df if tag is None else df[df["tag"] == tag].copy()


def runs_array(df_model: pd.DataFrame) -> np.ndarray:
    """(runs, images, items) array for one model (runs numbered 1..R)."""
    runs = sorted(df_model["run"].unique())
    arr = np.full((len(runs), N_IMAGES, len(PANAS_ITEMS)), np.nan)
    ri = {r: k for k, r in enumerate(runs)}
    ii = {it: k for k, it in enumerate(PANAS_ITEMS)}
    for r, im, it, sc in zip(df_model["run"], df_model["image"],
                             df_model["item"], df_model["score"]):
        arr[ri[r], im - 1, ii[it]] = sc
    return arr


def load_legacy_means(path: Path) -> np.ndarray:
    """Legacy CSV (items x 'image n') -> (images, items) in PANAS_ITEMS order."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.strip().str.lower()
    df.columns = [int(c.strip().split()[-1]) for c in df.columns]
    return df.loc[PANAS_ITEMS].T.sort_index().values


# ── Metrics ────────────────────────────────────────────────────────────────
def icc2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """ICC(2,1) and ICC(2,k=2) (two-way random, absolute agreement) for two raters.
    x, y: 1-D arrays of the n targets (cells)."""
    d = np.column_stack([x, y])
    n, k = d.shape
    gm = d.mean()
    msr = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    i21 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    i2k = (msr - mse) / (msr + (msc - mse) / n)
    return float(i21), float(i2k)


def rdm_vec(mat: np.ndarray, metric: str) -> np.ndarray:
    """Upper-triangle RDM entries between rows (images) of mat."""
    return pdist(mat, metric=metric)


def rsa(H: np.ndarray, G: np.ndarray, metric: str = "correlation") -> float:
    rho = stats.spearmanr(rdm_vec(H, metric), rdm_vec(G, metric))[0]
    return float(rho)


def agreement_metrics(H: np.ndarray, G: np.ndarray, rsa_metrics=("correlation", "euclidean", "cosine"),
                      human_sd: float | None = None) -> dict:
    """H, G: (images, items). Primary unit = the 200 cells."""
    h, g = H.ravel(), G.ravel()
    i21, i2k = icc2(h, g)
    diff = h - g
    out = {
        "ICC2_1": i21, "ICC2_k": i2k,
        "pearson": float(np.corrcoef(h, g)[0, 1]),
        "spearman": float(stats.spearmanr(h, g)[0]),
        "MAE": float(np.abs(diff).mean()),
        "RMSE": float(np.sqrt((diff ** 2).mean())),
        "nMAE_range": float(np.abs(diff).mean() / SCALE_RANGE),
        "nRMSE_range": float(np.sqrt((diff ** 2).mean()) / SCALE_RANGE),
        "BA_bias": float(diff.mean()),
        "BA_sd": float(diff.std(ddof=1)),
    }
    sd = human_sd if human_sd is not None else float(h.std(ddof=1))
    out["nMAE_sd"] = out["MAE"] / sd
    out["nRMSE_sd"] = out["RMSE"] / sd
    out["BA_loa_lo"] = out["BA_bias"] - 1.96 * out["BA_sd"]
    out["BA_loa_hi"] = out["BA_bias"] + 1.96 * out["BA_sd"]
    # Image-specific agreement: remove each rater's own item means (over images)
    # so that item-level base rates (e.g. NA items are rare) cannot carry the
    # correlation. This is the part of agreement that requires seeing the image.
    hc, gc = (H - H.mean(0)).ravel(), (G - G.mean(0)).ravel()
    out["r_within_item"] = float(np.corrcoef(hc, gc)[0, 1])
    for m in rsa_metrics:
        out[f"RSA_{m}"] = rsa(H, G, m)
    # subscale (PA / NA) level: 10 images each
    for name, idx in (("PA", PA_IDX), ("NA", NA_IDX)):
        hs, gs = H[:, idx].mean(1), G[:, idx].mean(1)
        out[f"{name}_bias"] = float((hs - gs).mean())
        out[f"{name}_spearman"] = float(stats.spearmanr(hs, gs)[0])
    return out


def item_mean_baseline(H: np.ndarray) -> np.ndarray:
    """No-image 'model': every image gets the human item means over images.
    Its agreement is the floor any model must beat to show image sensitivity."""
    return np.broadcast_to(H.mean(0), H.shape).copy()


def bland_altman_checks(h: np.ndarray, g: np.ndarray) -> dict:
    """Assumption checks: normality of differences, proportional bias."""
    diff, mean = h - g, (h + g) / 2
    sw = stats.shapiro(diff)
    slope = stats.linregress(mean, diff)
    return {"shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue),
            "prop_bias_slope": float(slope.slope), "prop_bias_p": float(slope.pvalue)}
