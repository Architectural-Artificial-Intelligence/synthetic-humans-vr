#!/usr/bin/env python3
"""Build manuscript-ready composite figures from batch PANAS outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)

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
SUBSCALE_ITEMS = {"PA": PA_ITEMS, "NA": NA_ITEMS}
SUBSCALE_COLORS = {"PA": "#2563eb", "NA": "#dc2626"}


def rdm_correlation_distance(mat: np.ndarray) -> np.ndarray:
    """Compute square RDM using correlation distance (1 - Pearson r)."""
    corr = np.corrcoef(mat)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - corr


def cohens_d_pooled(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def family_for_model(model: str) -> str:
    if model.startswith("claude-"):
        return "Anthropic"
    if model.startswith("gemini-"):
        return "Google"
    if model.startswith("gpt-"):
        return "OpenAI"
    if model.startswith("llama-"):
        return "Meta"
    if model.startswith("qwen"):
        return "Alibaba"
    return "Other"


def pretty_model(model: str) -> str:
    mapping = {
        "claude-haiku-4-5": "Claude Haiku 4.5",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "claude-opus-4-6": "Claude Opus 4.6",
        "gemini-2.5-flash": "Gemini 2.5-Flash",
        "gemini-2.5-pro": "Gemini 2.5-Pro",
        "gpt-4.1": "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1-mini",
        "gpt-4.1-nano": "GPT-4.1-nano",
        "gpt-5.2": "GPT-5.2",
        "gpt-5.4": "GPT-5.4",
        "gpt-5.4-mini": "GPT-5.4-mini",
        "gpt-5.4-nano": "GPT-5.4-nano",
        "llama-3.2-90b-vision": "LLaMA-3.2-90B-Vision",
        "llama-4-maverick-fp8": "LLaMA-4-Maverick-FP8",
        "llama-4-maverick-turbo": "LLaMA-4-Maverick-Turbo",
        "llama-4-scout": "LLaMA-4-Scout",
        "qwen3-vl-235b": "Qwen3-VL-235B",
    }
    return mapping.get(model, model)


def ci_mean(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    _ = alpha
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    crit = 1.96
    return (mean - crit * sem, mean + crit * sem)


def load_model_bundle(model_dir: Path) -> dict | None:
    required = [
        model_dir / "agreement.json",
        model_dir / "rsa_results.json",
        model_dir / "effect_sizes.json",
        model_dir / "merged_image_item.csv",
        model_dir / "human_long.csv",
    ]
    if not all(p.exists() for p in required):
        return None

    with open(model_dir / "agreement.json", "r", encoding="utf-8") as f:
        agreement = json.load(f)
    with open(model_dir / "rsa_results.json", "r", encoding="utf-8") as f:
        rsa = json.load(f)
    with open(model_dir / "effect_sizes.json", "r", encoding="utf-8") as f:
        effect = json.load(f)

    merged = pd.read_csv(model_dir / "merged_image_item.csv")
    human_long = pd.read_csv(model_dir / "human_long.csv")

    row = {
        "model": model_dir.name,
        "model_label": pretty_model(model_dir.name),
        "family": family_for_model(model_dir.name),
        "icc2": agreement["ICC2"],
        "icc2_lo": agreement.get("ICC2_lo", np.nan),
        "icc2_hi": agreement.get("ICC2_hi", np.nan),
        "pearson_r": agreement["pearson_r"],
        "spearman_rho": agreement.get("spearman_rho", np.nan),
        "ba_mean_diff": agreement["BA_mean_diff"],
        "ba_loa_lo": agreement["BA_LoA"][0],
        "ba_loa_hi": agreement["BA_LoA"][1],
        "ba_loa_width": agreement["BA_LoA"][1] - agreement["BA_LoA"][0],
        "rsa_rho": rsa["rsa_spearman_rho"],
        "rsa_ci_lo": rsa["rsa_ci"][0],
        "rsa_ci_hi": rsa["rsa_ci"][1],
        "mean_cosine": rsa["mean_cosine"],
        "loo_min": rsa["loo_rsa_min"],
        "loo_max": rsa["loo_rsa_max"],
        "loo_mean": rsa["loo_rsa_mean"],
        "pa_d": effect["PA"]["cohens_d"],
        "na_d": effect["NA"]["cohens_d"],
        "pa_bf10": effect["PA"].get("BF10", np.nan),
        "na_bf10": effect["NA"].get("BF10", np.nan),
        "merged": merged,
        "human_long": human_long,
        "cosine_per_image": np.array(rsa["cosine_per_image"], dtype=float),
        "loo_per_image": np.array(rsa["loo_rsa_per_image"], dtype=float),
    }
    return row


def build_summary_df(results_root: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    bundles = {}
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if model_dir.name == "llama-3.2-11b-vision":
            continue
        bundle = load_model_bundle(model_dir)
        if bundle is None:
            continue
        rows.append({
            k: v
            for k, v in bundle.items()
            if k not in {"merged", "human_long", "cosine_per_image", "loo_per_image"}
        })
        bundles[model_dir.name] = bundle

    df = pd.DataFrame(rows)
    df = df.sort_values(["icc2", "rsa_rho"], ascending=[False, False]).reset_index(drop=True)
    return df, bundles


def build_human_rdm(any_bundle: dict) -> tuple[np.ndarray, list[str]]:
    merged = any_bundle["merged"].copy()
    human = merged.pivot(index="image", columns="item", values="human_mean")
    model = merged.pivot(index="image", columns="item", values="gpt_mean")
    common_cols = sorted(set(human.columns) & set(model.columns))
    human = human[common_cols].sort_index()
    h_mat = human.values
    rdm = rdm_correlation_distance(h_mat)
    labels = [str(i) for i in human.index]
    return rdm, labels


def image_subscale_scores(bundle: dict) -> pd.DataFrame:
    """Aggregate to one row per stimulus image and PA/NA subscale."""
    human_long = bundle["human_long"].copy()
    merged = bundle["merged"].copy()
    rows = []

    for subscale, items in SUBSCALE_ITEMS.items():
        human_sub = human_long[human_long["item"].isin(items)]
        model_sub = merged[merged["item"].isin(items)]

        human_participant_image = (
            human_sub.groupby(["image", "participant"], as_index=False)["score"]
            .mean()
            .rename(columns={"score": "participant_image_mean"})
        )
        human_stats = (
            human_participant_image.groupby("image")["participant_image_mean"]
            .agg(human_mean="mean", human_sd="std", n_human="count")
            .reset_index()
        )

        model_stats = (
            model_sub.groupby("image")["gpt_mean"]
            .agg(model_mean="mean", model_item_sd="std", n_items="count")
            .reset_index()
        )
        out = pd.merge(human_stats, model_stats, on="image", how="inner")
        out.insert(1, "subscale", subscale)
        out["model_run_sd"] = np.nan
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def figure1_top_model_mean_scatter(
    df: pd.DataFrame,
    bundles: dict[str, dict],
    out: Path,
    n_models: int = 3,
) -> None:
    """Human mean vs model mean for the best-performing models."""
    shown = df.head(n_models).copy()

    fig, axs = plt.subplots(1, len(shown), figsize=(12.8, 4.9), sharex=True, sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.22, wspace=0.12, hspace=0.08)
    if len(shown) == 1:
        axs = np.array([axs])
    recs = []
    all_x = []
    all_y = []
    per_model_stats = {}
    for _, r in shown.iterrows():
        model = r["model"]
        stats_df = image_subscale_scores(bundles[model])
        stats_df.insert(0, "model", model)
        stats_df.insert(1, "model_label", r["model_label"])
        recs.append(stats_df)
        per_model_stats[model] = stats_df
        all_x.extend(stats_df["human_mean"].to_numpy(dtype=float))
        all_y.extend(stats_df["model_mean"].to_numpy(dtype=float))

    lim0 = min(min(all_x), min(all_y)) - 0.15
    lim1 = max(max(all_x), max(all_y)) + 0.15

    for ax, (_, r) in zip(axs, shown.iterrows()):
        model = r["model"]
        stats_df = per_model_stats[model]

        x = stats_df["human_mean"].to_numpy(dtype=float)
        y = stats_df["model_mean"].to_numpy(dtype=float)

        for subscale in ["PA", "NA"]:
            sub = stats_df[stats_df["subscale"] == subscale]
            ax.scatter(
                sub["human_mean"],
                sub["model_mean"],
                s=72,
                color=SUBSCALE_COLORS[subscale],
                edgecolor="white",
                linewidth=0.9,
                alpha=0.90,
                label=subscale,
                zorder=3,
            )

        for _, row in stats_df.iterrows():
            ax.text(
                row["human_mean"],
                row["model_mean"],
                str(int(row["image"])),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=4,
            )

        ax.plot([lim0, lim1], [lim0, lim1], ls="--", lw=1.0, c="#333333", alpha=0.65, zorder=2)
        ax.set_xlim(lim0, lim1)
        ax.set_ylim(lim0, lim1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{r['model_label']}\nICC2={r['icc2']:.3f}, RSA={r['rsa_rho']:.3f}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Human mean", fontsize=10)

    axs[0].set_ylabel("Model mean", fontsize=10)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.savefig(
        out / "figure1_human_model_scatter_top3_means.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)

    pd.concat(recs, ignore_index=True).to_csv(
        out / "figure1_human_model_scatter_top3_means_data.csv",
        index=False,
    )


def figure1b_top_model_bland_altman(
    df: pd.DataFrame,
    bundles: dict[str, dict],
    out: Path,
    n_models: int = 3,
) -> None:
    """Bland-Altman plots for image-level human/model means."""
    shown = df.head(n_models).copy()

    fig, axs = plt.subplots(1, len(shown), figsize=(12.8, 4.9), sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.22, wspace=0.12, hspace=0.08)
    if len(shown) == 1:
        axs = np.array([axs])

    recs = []
    per_model_stats = {}
    all_diff = []
    all_mean = []
    all_line_y = []
    for _, r in shown.iterrows():
        model = r["model"]
        stats_df = image_subscale_scores(bundles[model])
        stats_df.insert(0, "model", model)
        stats_df.insert(1, "model_label", r["model_label"])
        stats_df["ba_mean"] = (stats_df["human_mean"] + stats_df["model_mean"]) / 2.0
        stats_df["ba_diff"] = stats_df["human_mean"] - stats_df["model_mean"]
        per_model_stats[model] = stats_df
        recs.append(stats_df)
        all_mean.extend(stats_df["ba_mean"].to_numpy(dtype=float))
        all_diff.extend(stats_df["ba_diff"].to_numpy(dtype=float))
        for subscale in ["PA", "NA"]:
            y_sub = stats_df.loc[stats_df["subscale"] == subscale, "ba_diff"].to_numpy(dtype=float)
            mean_diff = float(np.mean(y_sub))
            sd_diff = float(np.std(y_sub, ddof=1))
            all_line_y.extend([mean_diff, mean_diff - 1.96 * sd_diff, mean_diff + 1.96 * sd_diff])

    x_pad = 0.12
    y_pad = 0.10
    xlim = (min(all_mean) - x_pad, max(all_mean) + x_pad)
    ylim = (min(all_diff + all_line_y) - y_pad, max(all_diff + all_line_y) + y_pad)

    for ax, (_, r) in zip(axs, shown.iterrows()):
        stats_df = per_model_stats[r["model"]]
        for subscale in ["PA", "NA"]:
            sub = stats_df[stats_df["subscale"] == subscale]
            y_sub = sub["ba_diff"].to_numpy(dtype=float)
            mean_diff = float(np.mean(y_sub))
            sd_diff = float(np.std(y_sub, ddof=1))
            loa_lo = mean_diff - 1.96 * sd_diff
            loa_hi = mean_diff + 1.96 * sd_diff
            color = SUBSCALE_COLORS[subscale]

            ax.scatter(
                sub["ba_mean"],
                sub["ba_diff"],
                s=72,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.90,
                label=subscale,
                zorder=3,
            )
            ax.axhline(mean_diff, color=color, lw=1.2, zorder=2)
            ax.axhline(loa_lo, color=color, lw=0.9, ls="--", alpha=0.55, zorder=2)
            ax.axhline(loa_hi, color=color, lw=0.9, ls="--", alpha=0.55, zorder=2)

        for _, row in stats_df.iterrows():
            ax.text(
                row["ba_mean"],
                row["ba_diff"],
                str(int(row["image"])),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=4,
            )

        ax.axhline(0, color="#333333", lw=0.9, alpha=0.55, zorder=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(r["model_label"], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Mean of human and model", fontsize=10)

    axs[0].set_ylabel("Human - model", fontsize=10)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.savefig(
        out / "figure1b_bland_altman_top3_image_means.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)

    pd.concat(recs, ignore_index=True).to_csv(
        out / "figure1b_bland_altman_top3_image_means_data.csv",
        index=False,
    )


def figure1_agreement(df: pd.DataFrame, bundles: dict[str, dict], out: Path) -> None:
    n_models = len(df)
    cols = 6
    rows = int(np.ceil(n_models / cols))

    fig = plt.figure(figsize=(21, 14), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.08)

    fam_colors = {
        "Anthropic": "#1f77b4",
        "Google": "#d62728",
        "OpenAI": "#2ca02c",
        "Meta": "#9467bd",
        "Alibaba": "#ff7f0e",
        "Other": "#7f7f7f",
    }

    axs_scatter = subfigs[0].subplots(rows, cols, squeeze=False)
    for idx, (_, r) in enumerate(df.iterrows()):
        rr, cc = divmod(idx, cols)
        ax = axs_scatter[rr, cc]
        b = bundles[r["model"]]
        m = b["merged"]
        x = m["gpt_mean"].values
        y = m["human_mean"].values
        color = fam_colors.get(r["family"], "#7f7f7f")
        ax.scatter(x, y, s=14, alpha=0.55, color=color, linewidth=0)
        lim0 = min(x.min(), y.min()) - 0.05
        lim1 = max(x.max(), y.max()) + 0.05
        ax.plot([lim0, lim1], [lim0, lim1], ls="--", lw=0.8, c="black", alpha=0.7)
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(lim0, lim1, 60)
        ax.plot(xx, slope * xx + intercept, lw=0.9, c="black")
        ax.set_xlim(lim0, lim1)
        ax.set_ylim(lim0, lim1)
        ax.set_title(f"{r['model_label']}\nICC2={r['icc2']:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if rr == rows - 1:
            ax.set_xlabel("Model", fontsize=8)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if cc == 0:
            ax.set_ylabel("Human", fontsize=8)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    for idx in range(n_models, rows * cols):
        rr, cc = divmod(idx, cols)
        axs_scatter[rr, cc].axis("off")

    subfigs[0].suptitle("Figure 1a. Human vs Model Item-Image Means (N=200/model)", fontsize=14, fontweight="bold")

    axs_ba = subfigs[1].subplots(rows, cols, squeeze=False)
    for idx, (_, r) in enumerate(df.iterrows()):
        rr, cc = divmod(idx, cols)
        ax = axs_ba[rr, cc]
        b = bundles[r["model"]]
        m = b["merged"]
        mean_vals = (m["human_mean"].values + m["gpt_mean"].values) / 2.0
        diff_vals = m["human_mean"].values - m["gpt_mean"].values
        is_low = r["icc2"] < 0.75
        color = "#d62728" if is_low else "#7f7f7f"
        ax.scatter(mean_vals, diff_vals, s=14, alpha=0.45, color=color, linewidth=0)
        ax.axhline(r["ba_mean_diff"], color="#d62728", lw=0.9)
        ax.axhline(r["ba_loa_lo"], color="#d62728", ls="--", lw=0.8)
        ax.axhline(r["ba_loa_hi"], color="#d62728", ls="--", lw=0.8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.7)
        ax.set_title(f"{r['model_label']}\nLoA width={r['ba_loa_width']:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if rr == rows - 1:
            ax.set_xlabel("Mean", fontsize=8)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if cc == 0:
            ax.set_ylabel("H - M", fontsize=8)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    for idx in range(n_models, rows * cols):
        rr, cc = divmod(idx, cols)
        axs_ba[rr, cc].axis("off")

    subfigs[1].suptitle("Figure 1b. Bland-Altman per Model (red = ICC2 < 0.75)", fontsize=14, fontweight="bold")
    fig.savefig(out / "figure1_agreement.png", dpi=300)
    plt.close(fig)


def figure2_geometry(df: pd.DataFrame, bundles: dict[str, dict], out: Path) -> None:
    human_rdm, labels = build_human_rdm(next(iter(bundles.values())))

    top_models = df.head(4)["model"].tolist()
    low_models = df.sort_values("rsa_rho", ascending=True).head(2)["model"].tolist()
    shown = top_models + [m for m in low_models if m not in top_models]
    shown = shown[:6]

    fig_rdm = plt.figure(figsize=(10.2, 5.4), constrained_layout=True)
    fig_rdm.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)
    rdm_gs = fig_rdm.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0])

    ax_h = fig_rdm.add_subplot(rdm_gs[:, 0])
    sns.heatmap(
        human_rdm,
        ax=ax_h,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        square=True,
    )
    ax_h.set_title("Human", fontsize=9, fontweight="bold")
    ax_h.tick_params(axis="both", labelsize=7)

    for i, model in enumerate(shown):
        r = bundles[model]
        merged = r["merged"]
        model_mat = merged.pivot(index="image", columns="item", values="gpt_mean").sort_index()
        h_mat = merged.pivot(index="image", columns="item", values="human_mean").sort_index()
        common_cols = sorted(set(model_mat.columns) & set(h_mat.columns))
        g_rdm = rdm_correlation_distance(model_mat[common_cols].values)
        row, col = divmod(i, 3)
        ax = fig_rdm.add_subplot(rdm_gs[row, col + 1])
        sns.heatmap(
            g_rdm,
            ax=ax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True,
        )
        ax.set_title(f"{pretty_model(model)}\nRSA={r['rsa_rho']:.3f}", fontsize=9)

    fig_rdm.savefig(out / "figure2a_rdm_heatmaps.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_rdm)

    fig_rank, ax_rsa = plt.subplots(figsize=(5.6, 6.7), constrained_layout=True)
    df_rsa = df.sort_values("rsa_rho", ascending=True)
    y = np.arange(len(df_rsa))
    ax_rsa.hlines(y, df_rsa["rsa_ci_lo"], df_rsa["rsa_ci_hi"], color="gray", lw=1)
    ax_rsa.scatter(df_rsa["rsa_rho"], y, c=np.where(df_rsa["rsa_rho"] < 0.7, "#d62728", "#1f77b4"), s=30)
    ax_rsa.set_yticks(y)
    ax_rsa.set_yticklabels(df_rsa["model_label"], fontsize=8)
    ax_rsa.set_xlabel("RSA Spearman rho (95% CI)")
    ax_rsa.set_title("RSA Ranking")
    ax_rsa.grid(True, axis="x", alpha=0.3)

    fig_rank.savefig(out / "figure2b_rsa_ranking.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig_rank)


def figure3_bias(df: pd.DataFrame, bundles: dict[str, dict], out: Path) -> None:
    recs = []
    for _, r in df.iterrows():
        model = r["model"]
        gpt_pa_na = pd.read_csv(Path(args.results_root) / model / "gpt_pa_na.csv")
        human_pa_na = pd.read_csv(Path(args.results_root) / model / "human_pa_na.csv")

        h = human_pa_na.groupby("image")[["PA", "NA"]].mean().sort_index()
        g = gpt_pa_na.groupby("image")[["PA", "NA"]].mean().sort_index()

        for subscale, d_key, bf_key in [("PA", "pa_d", "pa_bf10"), ("NA", "na_d", "na_bf10")]:
            hv = h[subscale].values.astype(float)
            gv = g[subscale].values.astype(float)
            rng = np.random.default_rng(42)
            boot = []
            n = len(hv)
            for _ in range(2000):
                idx = rng.integers(0, n, size=n)
                boot.append(cohens_d_pooled(hv[idx], gv[idx]))
            d_lo, d_hi = np.nanpercentile(np.array(boot), [2.5, 97.5])
            recs.append({
                "model": model,
                "model_label": r["model_label"],
                "subscale": subscale,
                "d": r[d_key],
                "bf10": r[bf_key],
                "d_lo": d_lo,
                "d_hi": d_hi,
                "family": r["family"],
            })

    bdf = pd.DataFrame(recs)
    order = df.sort_values("family")["model"].tolist()
    model_to_y = {m: i for i, m in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)
    fam_palette = {
        "Anthropic": "#1f77b4",
        "Google": "#d62728",
        "OpenAI": "#2ca02c",
        "Meta": "#9467bd",
        "Alibaba": "#ff7f0e",
        "Other": "#7f7f7f",
    }

    for ax, sub in zip(axes, ["PA", "NA"]):
        sdf = bdf[bdf["subscale"] == sub].copy()
        for _, row in sdf.iterrows():
            y = model_to_y[row["model"]]
            color = fam_palette.get(row["family"], "#7f7f7f")
            marker = "o" if sub == "PA" else "^"
            size = 46 if row["bf10"] >= 10 else 34
            ax.hlines(y, row["d_lo"], row["d_hi"], color=color, lw=1.5, alpha=0.8)
            ax.scatter(row["d"], y, color=color, s=size, marker=marker, alpha=0.9)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{sub} Effect Size (Cohen d)")
        ax.set_xlabel("d (positive = model underestimates)")
        ax.grid(True, axis="x", alpha=0.3)

    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([pretty_model(m) for m in order], fontsize=9)
    axes[1].set_yticks(np.arange(len(order)))
    axes[1].tick_params(labelleft=False)

    fig.suptitle("Figure 3. Subscale Bias Profiles Across Models", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.22, 0.02, 1, 0.96))
    fig.savefig(out / "figure3_subscale_bias.png", dpi=300)
    plt.close(fig)


def figure4_heatmap(df: pd.DataFrame, out: Path) -> None:
    heat = df[["model_label", "family", "icc2", "rsa_rho", "pa_d", "na_d", "ba_loa_width"]].copy()
    heat["abs_pa_d"] = heat["pa_d"].abs()
    heat["abs_na_d"] = heat["na_d"].abs()
    heat = heat[["model_label", "family", "icc2", "rsa_rho", "abs_pa_d", "abs_na_d", "ba_loa_width"]]

    heat = heat.sort_values(["family", "icc2"], ascending=[True, False]).reset_index(drop=True)
    mat = heat[["icc2", "rsa_rho", "abs_pa_d", "abs_na_d", "ba_loa_width"]].copy()

    # Higher-is-better columns
    for col in ["icc2", "rsa_rho"]:
        mat[col] = (mat[col] - mat[col].min()) / (mat[col].max() - mat[col].min() + 1e-9)
    # Lower-is-better columns (invert)
    for col in ["abs_pa_d", "abs_na_d", "ba_loa_width"]:
        z = (mat[col] - mat[col].min()) / (mat[col].max() - mat[col].min() + 1e-9)
        mat[col] = 1 - z

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    sns.heatmap(
        mat,
        ax=ax,
        cmap="YlGnBu",
        cbar_kws={"label": "Normalized performance (higher is better)"},
        yticklabels=heat["model_label"],
        xticklabels=["ICC2", "RSA rho", "|PA d|", "|NA d|", "LoA width"],
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title("Figure 4. Model Performance Summary Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(out / "figure4_performance_heatmap.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/20260325_234158")
    p.add_argument("--output-dir", default="figures/20260325_234158/manuscript")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, bundles = build_summary_df(results_root)
    if df.empty:
        raise RuntimeError(f"No model bundles found in {results_root}")

    figure1_top_model_mean_scatter(df, bundles, output_dir)
    figure1b_top_model_bland_altman(df, bundles, output_dir)
    figure1_agreement(df, bundles, output_dir)
    figure2_geometry(df, bundles, output_dir)
    figure3_bias(df, bundles, output_dir)
    figure4_heatmap(df, output_dir)

    summary = df[["model", "family", "icc2", "rsa_rho", "pa_d", "na_d", "ba_loa_width"]].copy()
    summary.to_csv(output_dir / "figure_summary_table.csv", index=False)

    print(f"Saved manuscript figures to: {output_dir}")
