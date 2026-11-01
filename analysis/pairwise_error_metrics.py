"""
Pairwise PANAS error metrics by model.

Computes direct error summaries over the 200 image x item pairs already
exported as merged_image_item.csv for each model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH = ROOT / "results" / "20260325_234158"
SCORE_MIN = 1.0
SCORE_MAX = 5.0
SCORE_RANGE = SCORE_MAX - SCORE_MIN


def normalize_score(series: pd.Series) -> pd.Series:
    """Normalize PANAS scores from the 1-5 scale to the 0-1 range."""
    return (series - SCORE_MIN) / SCORE_RANGE


def summarize_model(model_dir: Path) -> dict[str, float | int | str]:
    merged_path = model_dir / "merged_image_item.csv"
    df = pd.read_csv(merged_path)

    required = {"human_mean", "gpt_mean", "diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{merged_path} missing columns: {sorted(missing)}")

    raw_err = df["gpt_mean"] - df["human_mean"]
    raw_abs_err = raw_err.abs()

    err = normalize_score(df["gpt_mean"]) - normalize_score(df["human_mean"])
    abs_err = err.abs()

    return {
        "model": model_dir.name,
        "n_pairs": int(len(df)),
        "mean_error_0_1_model_minus_human": float(err.mean()),
        "mae_0_1": float(abs_err.mean()),
        "median_ae_0_1": float(abs_err.median()),
        "rmse_0_1": float(np.sqrt(np.mean(np.square(err)))),
        "max_ae_0_1": float(abs_err.max()),
        "within_0_05": float((abs_err <= 0.05).mean()),
        "within_0_10": float((abs_err <= 0.10).mean()),
        "within_0_25": float((abs_err <= 0.25).mean()),
        "raw_mean_error_1_5_model_minus_human": float(raw_err.mean()),
        "raw_mae_1_5": float(raw_abs_err.mean()),
        "raw_rmse_1_5": float(np.sqrt(np.mean(np.square(raw_err)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=DEFAULT_BATCH,
        help="Batch result directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <batch-dir>/pairwise_error_metrics.csv.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include model folders without agreement.json diagnostics.",
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir
    output = args.output or batch_dir / "pairwise_error_metrics.csv"

    model_dirs = []
    for path in sorted(batch_dir.iterdir()):
        if not path.is_dir() or not (path / "merged_image_item.csv").exists():
            continue
        if not args.include_incomplete and not (path / "agreement.json").exists():
            continue
        model_dirs.append(path)
    rows = [summarize_model(model_dir) for model_dir in model_dirs]
    out = pd.DataFrame(rows).sort_values(["mae_0_1", "rmse_0_1", "model"])
    out.to_csv(output, index=False, float_format="%.6f")

    display_cols = [
        "model",
        "mae_0_1",
        "rmse_0_1",
        "mean_error_0_1_model_minus_human",
        "within_0_10",
        "within_0_25",
        "raw_mae_1_5",
    ]
    print(out[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
