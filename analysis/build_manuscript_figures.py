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
        rows.append({k: v for k, v in bundle.items() if k not in {"merged", "cosine_per_image", "loo_per_image"}})
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
    labels = [f"Im.{i}" for i in human.index]
    return rdm, labels


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

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    ax_h = fig.add_subplot(gs[0, 0])
    sns.heatmap(human_rdm, ax=ax_h, cmap="viridis", xticklabels=labels, yticklabels=labels, cbar=False)
    ax_h.set_title("Human RDM")

    for i, model in enumerate(shown):
        r = bundles[model]
        merged = r["merged"]
        model_mat = merged.pivot(index="image", columns="item", values="gpt_mean").sort_index()
        h_mat = merged.pivot(index="image", columns="item", values="human_mean").sort_index()
        common_cols = sorted(set(model_mat.columns) & set(h_mat.columns))
        g_rdm = rdm_correlation_distance(model_mat[common_cols].values)
        row = 0 if i < 3 else 1
        col = 1 + (i % 3)
        ax = fig.add_subplot(gs[row, col])
        sns.heatmap(g_rdm, ax=ax, cmap="viridis", xticklabels=False, yticklabels=False, cbar=False)
        ax.set_title(f"{pretty_model(model)}\nRSA={r['rsa_rho']:.3f}", fontsize=9)

    ax_rsa = fig.add_subplot(gs[2, 0:2])
    df_rsa = df.sort_values("rsa_rho", ascending=True)
    y = np.arange(len(df_rsa))
    ax_rsa.hlines(y, df_rsa["rsa_ci_lo"], df_rsa["rsa_ci_hi"], color="gray", lw=1)
    ax_rsa.scatter(df_rsa["rsa_rho"], y, c=np.where(df_rsa["rsa_rho"] < 0.7, "#d62728", "#1f77b4"), s=30)
    ax_rsa.set_yticks(y)
    ax_rsa.set_yticklabels(df_rsa["model_label"], fontsize=8)
    ax_rsa.set_xlabel("RSA Spearman rho (95% CI)")
    ax_rsa.set_title("RSA Ranking")

    ax_cos = fig.add_subplot(gs[2, 2])
    cos_data = [bundles[m]["cosine_per_image"] for m in df_rsa["model"]]
    ax_cos.boxplot(cos_data, vert=True, patch_artist=True, boxprops={"facecolor": "#9ecae1", "alpha": 0.7}, medianprops={"color": "#08519c"})
    ax_cos.set_xticks([1, len(cos_data)])
    ax_cos.set_xticklabels(["Best", "Worst"])
    ax_cos.set_ylabel("Cosine")
    ax_cos.set_title("Per-image Cosine Spread")

    ax_loo = fig.add_subplot(gs[2, 3])
    df_loo = df.sort_values("loo_mean", ascending=True)
    y2 = np.arange(len(df_loo))
    ax_loo.hlines(y2, df_loo["loo_min"], df_loo["loo_max"], color="#636363", lw=1.4)
    ax_loo.scatter(df_loo["loo_mean"], y2, c="#31a354", s=20)
    ax_loo.set_yticks([])
    ax_loo.set_xlabel("LOO RSA")
    ax_loo.set_title("LOO RSA Range")

    fig.suptitle("Figure 2. Representational Geometry Across Models", fontsize=16, fontweight="bold")
    fig.savefig(out / "figure2_geometry.png", dpi=300)
    plt.close(fig)


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

    figure1_agreement(df, bundles, output_dir)
    figure2_geometry(df, bundles, output_dir)
    figure3_bias(df, bundles, output_dir)
    figure4_heatmap(df, output_dir)

    summary = df[["model", "family", "icc2", "rsa_rho", "pa_d", "na_d", "ba_loa_width"]].copy()
    summary.to_csv(output_dir / "figure_summary_table.csv", index=False)

    print(f"Saved manuscript figures to: {output_dir}")
