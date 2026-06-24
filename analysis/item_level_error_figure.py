#!/usr/bin/env python3
"""Build an item-level PANAS error figure across VLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH = ROOT / "results" / "20260325_234158"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "20260325_234158" / "manuscript"

PA_ITEMS = {
    "interested",
    "excited",
    "strong",
    "enthusiastic",
    "proud",
    "alert",
    "inspired",
    "determined",
    "attentive",
    "active",
}
NA_ITEMS = {
    "nervous",
    "upset",
    "guilty",
    "scared",
    "hostile",
    "irritable",
    "ashamed",
    "jittery",
    "distressed",
    "afraid",
}
ACTIVATION_ITEMS = {"active", "excited"}
SELF_CONSCIOUS_ITEMS = {"ashamed", "guilty"}

PA_COLOR = "#2f6fbb"
NA_COLOR = "#c7433d"
NEUTRAL = "#4b5563"
GRID = "#e5e7eb"
TEXT = "#111827"


def select_model_dirs(batch_dir: Path, top_n: int | None) -> list[Path]:
    model_dirs = []
    for model_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        if model_dir.name == "llama-3.2-11b-vision":
            continue
        merged_path = model_dir / "merged_image_item.csv"
        agreement_path = model_dir / "agreement.json"
        if not merged_path.exists() or not agreement_path.exists():
            continue
        with open(agreement_path, "r", encoding="utf-8") as f:
            agreement = json.load(f)
        model_dirs.append((model_dir, float(agreement["ICC2"])))

    model_dirs = sorted(model_dirs, key=lambda x: (x[1], x[0].name), reverse=True)
    if top_n is not None:
        model_dirs = model_dirs[:top_n]
    return [model_dir for model_dir, _ in model_dirs]


def load_errors(batch_dir: Path, top_n: int | None = None) -> pd.DataFrame:
    rows = []
    for model_dir in select_model_dirs(batch_dir, top_n):
        merged_path = model_dir / "merged_image_item.csv"

        df = pd.read_csv(merged_path)
        required = {"image", "item", "human_mean", "gpt_mean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{merged_path} missing columns: {sorted(missing)}")

        df["model"] = model_dir.name
        df["subscale"] = np.where(df["item"].isin(PA_ITEMS), "PA", "NA")
        df["contrast_group"] = np.select(
            [
                df["item"].isin(ACTIVATION_ITEMS),
                df["item"].isin(SELF_CONSCIOUS_ITEMS),
            ],
            ["Activation", "Self-conscious"],
            default="Other",
        )
        df["error"] = df["gpt_mean"] - df["human_mean"]
        df["abs_error"] = df["error"].abs()
        df["sq_error"] = df["error"] ** 2
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No usable model outputs found under {batch_dir}")
    out = pd.concat(rows, ignore_index=True)
    out.attrs["model_names"] = [p.name for p in select_model_dirs(batch_dir, top_n)]
    return out


def summarize_items(df: pd.DataFrame) -> pd.DataFrame:
    item_stats = (
        df.groupby(["item", "subscale"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            bias=("error", "mean"),
            human_mean=("human_mean", "mean"),
            model_mean=("gpt_mean", "mean"),
        )
        .sort_values("mae", ascending=True)
        .reset_index(drop=True)
    )
    item_stats["rank_easiest"] = np.arange(1, len(item_stats) + 1)
    return item_stats


def summarize_contrasts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subscale = (
        df.groupby(["model", "subscale"], as_index=False)
        .agg(mae=("abs_error", "mean"), bias=("error", "mean"))
        .pivot(index="model", columns="subscale", values=["mae", "bias"])
    )
    pa_na = pd.DataFrame(
        {
            "contrast": ["PA", "NA"],
            "mae": [
                float(subscale[("mae", "PA")].mean()),
                float(subscale[("mae", "NA")].mean()),
            ],
            "bias": [
                float(subscale[("bias", "PA")].mean()),
                float(subscale[("bias", "NA")].mean()),
            ],
            "color": [PA_COLOR, NA_COLOR],
        }
    )

    group_df = df[df["contrast_group"].isin(["Activation", "Self-conscious"])]
    groups = (
        group_df.groupby(["model", "contrast_group"], as_index=False)
        .agg(mae=("abs_error", "mean"), bias=("error", "mean"))
        .pivot(index="model", columns="contrast_group", values=["mae", "bias"])
    )
    activation_self = pd.DataFrame(
        {
            "contrast": ["Activation", "Self-conscious"],
            "mae": [
                float(groups[("mae", "Activation")].mean()),
                float(groups[("mae", "Self-conscious")].mean()),
            ],
            "bias": [
                float(groups[("bias", "Activation")].mean()),
                float(groups[("bias", "Self-conscious")].mean()),
            ],
            "color": [NEUTRAL, "#8a5a44"],
        }
    )
    return pa_na, activation_self


def write_source_data(
    item_stats: pd.DataFrame,
    pa_na: pd.DataFrame,
    activation_self: pd.DataFrame,
    output_dir: Path,
    suffix: str,
) -> None:
    item_stats.to_csv(output_dir / f"figure5_item_error_structure{suffix}_data.csv", index=False)
    contrasts = pd.concat(
        [
            pa_na.assign(panel="PA_vs_NA"),
            activation_self.assign(panel="activation_vs_self_conscious"),
        ],
        ignore_index=True,
    )
    contrasts.drop(columns=["color"]).to_csv(
        output_dir / f"figure5_item_error_contrasts{suffix}.csv", index=False
    )


def build_figure(
    item_stats: pd.DataFrame,
    pa_na: pd.DataFrame,
    activation_self: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    model_count: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.8,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#9ca3af",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )

    colors = item_stats["subscale"].map({"PA": PA_COLOR, "NA": NA_COLOR}).to_numpy()
    y = np.arange(len(item_stats))

    fig = plt.figure(figsize=(11.2, 7.45), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[4.8, 1.35],
        width_ratios=[1.08, 1.0],
        hspace=0.32,
        wspace=0.24,
    )
    ax_mae = fig.add_subplot(gs[0, 0])
    ax_bias = fig.add_subplot(gs[0, 1], sharey=ax_mae)
    ax_contrast = fig.add_subplot(gs[1, :])

    ax_mae.barh(y, item_stats["mae"], color=colors, height=0.72)
    ax_mae.set_yticks(y)
    ax_mae.set_yticklabels(item_stats["item"])
    ax_mae.invert_yaxis()
    ax_mae.set_xlabel("Mean absolute error on 1-5 PANAS scale")
    ax_mae.set_title("A. Item difficulty")
    ax_mae.axvline(item_stats["mae"].mean(), color="#6b7280", lw=1.1, ls="--")
    ax_mae.text(
        item_stats["mae"].mean() + 0.008,
        len(item_stats) - 0.1,
        "grand mean",
        color="#6b7280",
        fontsize=8.2,
        va="bottom",
    )
    ax_mae.grid(axis="x", color=GRID, lw=0.8)
    ax_mae.set_axisbelow(True)
    ax_mae.set_xlim(0, max(0.82, float(item_stats["mae"].max()) * 1.08))

    ax_bias.axvline(0, color="#111827", lw=1.0)
    ax_bias.hlines(y, 0, item_stats["bias"], color="#cbd5e1", lw=1.7)
    ax_bias.scatter(item_stats["bias"], y, s=46, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax_bias.set_xlabel("Signed bias: model - human")
    ax_bias.set_title("B. Calibration by item")
    ax_bias.tick_params(axis="y", left=False, labelleft=False)
    ax_bias.grid(axis="x", color=GRID, lw=0.8)
    ax_bias.set_axisbelow(True)
    bias_lim = max(0.8, float(np.abs(item_stats["bias"]).max()) * 1.12)
    ax_bias.set_xlim(-bias_lim, bias_lim)
    ax_bias.text(-bias_lim, -0.9, "underestimation", fontsize=8.2, color="#6b7280", ha="left")
    ax_bias.text(bias_lim, -0.9, "overestimation", fontsize=8.2, color="#6b7280", ha="right")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=PA_COLOR, markersize=8, label="PA"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=NA_COLOR, markersize=8, label="NA"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.962),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    contrast_labels = ["PA", "NA", "Activation", "Self-conscious"]
    contrast_mae = pd.concat([pa_na, activation_self], ignore_index=True)
    x = np.array([0, 0.8, 2.05, 2.85])
    ax_contrast.bar(
        x,
        contrast_mae["mae"],
        color=contrast_mae["color"],
        width=0.56,
        alpha=0.95,
    )
    for xpos, val in zip(x, contrast_mae["mae"]):
        ax_contrast.text(xpos, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax_contrast.set_xticks(x)
    ax_contrast.set_xticklabels(contrast_labels)
    ax_contrast.set_ylabel("Mean absolute error")
    ax_contrast.set_title("C. Planned item-group contrasts")
    ax_contrast.set_ylim(0, max(0.55, float(contrast_mae["mae"].max()) * 1.24))
    ax_contrast.grid(axis="y", color=GRID, lw=0.8)
    ax_contrast.set_axisbelow(True)
    ax_contrast.axvline(1.42, color="#d1d5db", lw=1.0)
    ax_contrast.text(0.4, 0.93, "Subscale", ha="center", transform=ax_contrast.get_xaxis_transform(), fontsize=8.5, color="#6b7280")
    ax_contrast.text(2.45, 0.93, "Item content", ha="center", transform=ax_contrast.get_xaxis_transform(), fontsize=8.5, color="#6b7280")

    for ax in [ax_mae, ax_bias, ax_contrast]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Item-level PANAS error structure across {model_count} VLMs",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.947,
        "Errors are computed over model x image pairs for each item; signed bias uses model minus human means.",
        ha="left",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.105, top=0.895)

    png_path = output_dir / f"figure5_item_error_structure{suffix}.png"
    pdf_path = output_dir / f"figure5_item_error_structure{suffix}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_separate_panels(
    item_stats: pd.DataFrame,
    pa_na: pd.DataFrame,
    activation_self: pd.DataFrame,
    output_dir: Path,
    suffix: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#9ca3af",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )

    colors = item_stats["subscale"].map({"PA": PA_COLOR, "NA": NA_COLOR}).to_numpy()
    y = np.arange(len(item_stats))
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=PA_COLOR, markersize=8, label="PA"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=NA_COLOR, markersize=8, label="NA"),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 6.8))
    ax.barh(y, item_stats["mae"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(item_stats["item"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error on 1-5 PANAS scale")
    ax.axvline(item_stats["mae"].mean(), color="#6b7280", lw=1.1, ls="--")
    ax.text(
        item_stats["mae"].mean() + 0.008,
        len(item_stats) - 0.1,
        "grand mean",
        color="#6b7280",
        fontsize=8.5,
        va="bottom",
    )
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(0.82, float(item_stats["mae"].max()) * 1.08))
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", ncol=2, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure5{suffix}_panel_a_item_difficulty.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 6.8))
    ax.axvline(0, color="#111827", lw=1.0)
    ax.hlines(y, 0, item_stats["bias"], color="#cbd5e1", lw=1.7)
    ax.scatter(item_stats["bias"], y, s=46, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(item_stats["item"])
    ax.invert_yaxis()
    ax.set_xlabel("Signed bias: model - human")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    bias_lim = max(0.8, float(np.abs(item_stats["bias"]).max()) * 1.12)
    ax.set_xlim(-bias_lim, bias_lim)
    ax.text(-bias_lim, -0.9, "underestimation", fontsize=8.5, color="#6b7280", ha="left")
    ax.text(bias_lim, -0.9, "overestimation", fontsize=8.5, color="#6b7280", ha="right")
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", ncol=2, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure5{suffix}_panel_b_item_bias.png", dpi=300)
    plt.close(fig)

    contrast_mae = pd.concat([pa_na, activation_self], ignore_index=True)
    x = np.array([0, 0.8, 2.05, 2.85])
    fig, ax = plt.subplots(figsize=(10.2, 1.85))
    ax.bar(x, contrast_mae["mae"], color=contrast_mae["color"], width=0.56, alpha=0.95)
    for xpos, val in zip(x, contrast_mae["mae"]):
        ax.text(xpos, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["PA", "NA", "Activation", "Self-conscious"])
    ax.set_ylabel("Mean absolute error")
    ax.set_ylim(0, max(0.55, float(contrast_mae["mae"].max()) * 1.24))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(1.42, color="#d1d5db", lw=1.0)
    ax.text(0.4, 0.93, "Subscale", ha="center", transform=ax.get_xaxis_transform(), fontsize=8.5, color="#6b7280")
    ax.text(2.45, 0.93, "Item content", ha="center", transform=ax.get_xaxis_transform(), fontsize=8.5, color="#6b7280")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.28, top=0.90)
    fig.savefig(output_dir / f"figure5{suffix}_panel_c_group_contrasts.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Restrict to the top N models ranked by ICC2.",
    )
    parser.add_argument(
        "--separate-panels",
        action="store_true",
        help="Also export untitled PNGs for each panel.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_errors(args.batch_dir, top_n=args.top_n)
    item_stats = summarize_items(df)
    pa_na, activation_self = summarize_contrasts(df)
    suffix = f"_top{args.top_n}" if args.top_n is not None else ""
    write_source_data(item_stats, pa_na, activation_self, args.output_dir, suffix)
    build_figure(
        item_stats,
        pa_na,
        activation_self,
        args.output_dir,
        suffix,
        model_count=df["model"].nunique(),
    )
    if args.separate_panels:
        build_separate_panels(item_stats, pa_na, activation_self, args.output_dir, suffix)

    print(f"Models: {', '.join(df.attrs['model_names'])}")
    print(f"Saved: {args.output_dir / f'figure5_item_error_structure{suffix}.png'}")
    print(f"Saved: {args.output_dir / f'figure5_item_error_structure{suffix}.pdf'}")
    print(f"Saved: {args.output_dir / f'figure5_item_error_structure{suffix}_data.csv'}")
    print(f"Saved: {args.output_dir / f'figure5_item_error_contrasts{suffix}.csv'}")
    if args.separate_panels:
        print(f"Saved: {args.output_dir / f'figure5{suffix}_panel_a_item_difficulty.png'}")
        print(f"Saved: {args.output_dir / f'figure5{suffix}_panel_b_item_bias.png'}")
        print(f"Saved: {args.output_dir / f'figure5{suffix}_panel_c_group_contrasts.png'}")


if __name__ == "__main__":
    main()
