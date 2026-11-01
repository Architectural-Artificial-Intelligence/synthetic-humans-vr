"""
Main (Fig 1-6) and supplementary (Fig S1-S3) figures for the paper.

Usage: python analysis/make_main_figures.py --batch results/<batch>
Needs : <batch>/figure_data/* (analysis/figure_stats.py), <batch>/stability/convergence_by_k.csv,
        analysis/scene_info.csv, photos/*.jpg
Writes: <batch>/paper/main_figures/Fig*.png|pdf, thumbnails/scene_<n>.jpg (no metadata), captions/*.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy import stats

import make_paper_outputs as P
import make_scene_thumbnails as T
import panas_lib as L

BLUE, ORANGE, INK, MUTED, GRID, GRAY = P.BLUE, P.ORANGE, P.INK, P.MUTED, P.GRID, P.GRAY
PC, PM, PL = P.PROVIDER_COLOR, P.PROVIDER_MARKER, P.PROVIDER_LABEL
TIER_FILL = {"low": "#f3e9e6", "mid": "#f6f3e3", "top": "#e3f0ea"}
BASELINE_LABEL = "item-mean baseline"
W = 7.2


def tier_bands(ax, orient="y", lo=None, hi=None, labels=True, x_text=None):
    """Shade the three tiers (low < .70 <= mid < .80 <= top)."""
    edges = [(lo, 0.70, "low"), (0.70, 0.80, "mid"), (0.80, hi, "top")]
    for a, b, t in edges:
        if orient == "y":
            ax.axhspan(a, b, color=TIER_FILL[t], lw=0, zorder=0)
        else:
            ax.axvspan(a, b, color=TIER_FILL[t], lw=0, zorder=0)
    if labels and orient == "y":
        for a, b, t in edges:
            ax.text(x_text, (a + b) / 2, t, fontsize=6.5, color=MUTED, ha="right", va="center", style="italic")


def provider_legend(ax, **kw):
    h = [plt.Line2D([], [], marker=PM[p], ls="", color=PC[p], label=PL[p], markersize=5.5) for p in PC]
    ax.legend(handles=h, **kw)


# ── Figure 1 ───────────────────────────────────────────────────────────────
def fig1(out: Path, s1: pd.DataFrame, info: pd.DataFrame):
    order = s1.sort_values("human_NA")["scene"].tolist()
    thumbs = out / "thumbnails"
    thumbs.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(W, 3.45))
    for ax, sc in zip(axes.ravel(), order):
        row = info[info["scene"] == sc].iloc[0]
        im, _ = T.make_thumbnail(sc, row)
        im.save(thumbs / f"scene_{sc}.jpg", quality=92)                    # no EXIF / XMP / GPS
        ax.imshow(im); ax.axis("off"); ax.grid(False)
        r = s1[s1["scene"] == sc].iloc[0]
        ax.set_title(f"{sc}  {row['short_name']}", fontsize=7.8, fontweight="bold", loc="left", pad=3)
        ax.text(0, -0.06, f"PA {r['human_PA']:.2f}", transform=ax.transAxes, fontsize=7.4, color=BLUE, va="top")
        ax.text(1, -0.06, f"NA {r['human_NA']:.2f}", transform=ax.transAxes, fontsize=7.4, color=ORANGE, va="top", ha="right")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, wspace=0.05, hspace=0.27)
    P.save_fig(fig, out / "Fig1_environments")


# ── Figure 2 ───────────────────────────────────────────────────────────────
def fig2(out: Path, em: pd.DataFrame, sl: pd.DataFrame, main_model="gemini-4-argon"):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.5))
    hum = em[em["model"] == "human"].set_index("scene")
    mods = em[em["model"] != "human"]
    for ax, sub, lim in zip(axes, ("PA", "NA"), ((1.5, 4.0), (1.0, 3.5))):
        ax.plot(lim, lim, ls="--", color=GRAY, lw=1, zorder=1)
        for sc in range(1, 11):
            v = mods[mods["scene"] == sc][sub].values
            x = hum.loc[sc, sub]
            ax.plot([x, x], [v.min(), v.max()], color="#c9ccc8", lw=1, zorder=2)
            ax.plot([x, x], [np.percentile(v, 25), np.percentile(v, 75)], color="#9fa7a3", lw=4.5, solid_capstyle="butt", zorder=3)
        a = mods[mods["model"] == main_model].set_index("scene")
        ax.scatter([hum.loc[s, sub] for s in a.index], a[sub], s=26, color=BLUE, zorder=5, label=main_model)
        for s in a.index:
            off = {"NA": {6: (5, 4), 7: (-2, -12), 8: (-13, 4)}}.get(sub, {}).get(s, (4, -9 if s == 9 else 3))
            ax.annotate(str(s), (hum.loc[s, sub], a.loc[s, sub]), xytext=off, textcoords="offset points",
                        fontsize=7.5, color=INK, fontweight="bold" if s == 9 else "normal", zorder=6)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xlabel(f"Human mean {sub}"); ax.set_ylabel(f"Model mean {sub}")
        ax.set_title("Positive affect (PA)" if sub == "PA" else "Negative affect (NA)")
        s_main = sl.set_index("model").loc[main_model, f"{sub}_slope"]
        s_med = sl[f"{sub}_slope"].median()
        ax.text(0.04, 0.95, f"slope on human means\n{main_model}: {s_main:.2f}\nmedian of {len(sl)} models: {s_med:.2f}",
                transform=ax.transAxes, fontsize=7, va="top", color=MUTED)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Gemini 4 Argon", markersize=5),
         plt.Line2D([], [], color="#9fa7a3", lw=4.5, label="IQR across models"),
         plt.Line2D([], [], color="#c9ccc8", lw=1, label="min-max across models"),
         plt.Line2D([], [], ls="--", color=GRAY, label="identity")]
    axes[1].legend(handles=h, loc="lower right", fontsize=7)
    fig.tight_layout()
    P.save_fig(fig, out / "Fig2_human_vs_model_by_environment")


# ── Figure 3 ───────────────────────────────────────────────────────────────
def fig3(out: Path, s3: pd.DataFrame, base: float, prov: dict):
    fig, ax = plt.subplots(figsize=(W, 4.6))
    xl, yl = (0.55, 0.97), (0.52, 0.90)
    tier_bands(ax, "y", yl[0], yl[1], x_text=xl[1] - 0.004)
    ax.axhline(base, color=INK, ls="--", lw=1)
    ax.text(0.665, base + 0.006, f"{BASELINE_LABEL}, ICC {base:.2f}", fontsize=7, va="bottom")
    for p in PC:
        d = s3[s3["model"].map(prov) == p]
        ax.scatter(d["within_item_r"], d["ICC2_1"], s=26, marker=PM[p], color=PC[p], alpha=.9, lw=0, zorder=3)
    label_pos = {"gemini-4-argon": (6, 4), "gemini-2.5-flash": (7, -11), "gemini-3.5-flash-lite": (8, 3),
                 "qwen3.8-27b": (8, -9), "gpt-5.4": (-6, 8), "gpt-4.1-mini": (-62, -3), "claude-sonnet-5-5": (5, -9),
                 "gpt-6-sol": (-12, -11)}
    for m, (dx, dy) in label_pos.items():
        r = s3[s3["model"] == m]
        if len(r):
            r = r.iloc[0]
            ax.annotate(m, (r["within_item_r"], r["ICC2_1"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.8, color=INK, zorder=6)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xlabel("Within-item r  (agreement on which scenes are higher or lower, per item)")
    ax.set_ylabel("ICC(2,1) over the 200 item x scene cells")
    provider_legend(ax, loc="upper left", fontsize=7)
    P.save_fig(fig, out / "Fig3_icc_vs_within_item_r")


# ── Figure 4 ───────────────────────────────────────────────────────────────
def fig4(out: Path, isum: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(W, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for ax, sub, col in zip(axes, ("PA", "NA"), (BLUE, ORANGE)):
        d = isum[isum["subscale"] == sub].sort_values("median").reset_index(drop=True)
        for i, r in d.iterrows():
            ax.plot([r["min"], r["max"]], [i, i], color=col, lw=1, alpha=.45, zorder=2)
            ax.plot([r["q25"], r["q75"]], [i, i], color=col, lw=5, alpha=.55, solid_capstyle="butt", zorder=3)
            if r["adapted"]:
                ax.plot(r["median"], i, "D", color="white", mec=INK, mew=1.4, ms=6.5, zorder=5)
            else:
                ax.plot(r["median"], i, "o", color=INK, ms=4.5, zorder=5)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(d)), d["item"], fontsize=8)
        ax.set_title("Positive-affect items" if sub == "PA" else "Negative-affect items", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Signed error, model - human (rating points; below 0 = model rates lower)")
    h = [plt.Line2D([], [], marker="o", ls="", color=INK, label="median across models", markersize=5),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec=INK, mew=1.3, label="adapted Hebrew item", markersize=6),
         plt.Line2D([], [], color=GRAY, lw=5, alpha=.6, label="IQR"),
         plt.Line2D([], [], color=GRAY, lw=1, label="min-max")]
    axes[0].legend(handles=h, loc="upper left", fontsize=7)
    fig.tight_layout()
    P.save_fig(fig, out / "Fig4_signed_error_by_item")


# ── Figure 5 ───────────────────────────────────────────────────────────────
def fig5(out: Path, s3: pd.DataFrame, base: float, prov: dict):
    d = s3.sort_values("ICC2_1").reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(W, 0.215 * n + 1.5))
    for x in (0.70, 0.80):
        ax.axvline(x, color=GRAY, lw=.8, ls=":", zorder=1)
    ax.axvline(base, color=INK, lw=1, ls="--", zorder=1)
    for i, r in d.iterrows():
        c = PC[prov[r["model"]]]
        ax.plot([r["env_lo"], r["env_hi"]], [i, i], color=c, lw=1, alpha=.7, zorder=2)
        ax.plot([r["part_run_lo"], r["part_run_hi"]], [i, i], color=c, lw=3.8, solid_capstyle="butt", zorder=3)
        ax.plot(r["ICC2_1"], i, PM[prov[r["model"]]], color="white", mec=c, mew=1.3, ms=4.5, zorder=4)
        if r["margin_excludes_zero"]:
            ax.plot(r["env_hi"] + 0.018, i, "*", color=INK, ms=6.5, zorder=5, clip_on=False)
    ax.set_yticks(range(n), d["model"], fontsize=7.3)
    for lab, m in zip(ax.get_yticklabels(), d["model"]):
        lab.set_color(PC[prov[m]])
    ax.set_xlim(0.30, 1.0); ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("ICC(2,1) with human cell means")
    ax.grid(axis="y", visible=False)
    ax.text(base, n - 0.35, f"baseline {base:.2f}", fontsize=6.8, va="bottom", ha="center")
    ax.text(0.70, n - 0.35, ".70", fontsize=6.8, va="bottom", ha="center", color=MUTED)
    ax.text(0.80, n - 0.35, ".80", fontsize=6.8, va="bottom", ha="center", color=MUTED)
    h = [plt.Line2D([], [], color=GRAY, lw=3.8, label="95% CI, participants + runs"),
         plt.Line2D([], [], color=GRAY, lw=1, label="95% CI, environments"),
         plt.Line2D([], [], marker="*", ls="", color=INK, markersize=7, label="margin over baseline excludes 0")]
    l1 = ax.legend(handles=h, loc="upper left", fontsize=6.8, bbox_to_anchor=(0.0, 0.99))
    ax.add_artist(l1)
    provider_legend(ax, loc="upper left", fontsize=6.8, bbox_to_anchor=(0.0, 0.89))
    P.save_fig(fig, out / "Fig5_model_ranking_two_uncertainties")


# ── Figure 6 ───────────────────────────────────────────────────────────────
FAMILIES = [("Claude Sonnet", "claude-sonnet-4-6", "claude-sonnet-5-5", "4.6", "5.5", "anthropic"),
            ("Claude Opus", "claude-opus-4-6", "claude-opus-5-5", "4.6", "5.5", "anthropic"),
            ("Claude Haiku", "claude-haiku-4-5", "claude-haiku-5-5", "4.5", "5.5", "anthropic"),
            ("GPT", "gpt-5.4", "gpt-6-sol", "5.4", "6 Sol", "openai"),
            ("Gemini Pro", "gemini-2.5-pro", "gemini-4-argon", "2.5", "4 Argon", "google"),
            ("Kimi", "kimi-k2.6", "kimi-k3", "K2.6", "K3", "deepinfra")]


def fig6(out: Path, s3: pd.DataFrame):
    v = s3.set_index("model")["ICC2_1"]
    fig, ax = plt.subplots(figsize=(W * 0.82, 4.4))
    lo, hi = 0.55, 0.90
    tier_bands(ax, "y", lo, hi, x_text=1.97)
    ax.axhline(0.70, color=GRAY, lw=.6, ls=":"); ax.axhline(0.80, color=GRAY, lw=.6, ls=":")
    dy = {"Claude Sonnet": 0, "Claude Opus": 0, "Claude Haiku": 0, "GPT": 0, "Gemini Pro": 0, "Kimi": 0}
    for fam, old, new, vo, vn, p in FAMILIES:
        c = PC[p]
        ax.plot([0, 1], [v[old], v[new]], color=c, lw=2, marker=PM[p], ms=6, zorder=3)
        ax.text(-0.04, v[old] + dy[fam], f"{fam} {vo}  {v[old]:.2f}", ha="right", va="center", fontsize=7.2, color=INK)
        ax.text(1.04, v[new] + dy[fam], f"{v[new]:.2f}  {fam} {vn}", ha="left", va="center", fontsize=7.2, color=INK)
    m = "gpt-4.1-mini"
    ax.plot([0.5], [v[m]], "o", mfc="white", mec=PC["openai"], mew=1.5, ms=7, zorder=4)
    ax.text(0.5, v[m] + 0.013, f"GPT-4.1 mini  {v[m]:.2f}\n(no paired successor)", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xlim(-1.0, 2.0); ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1], ["older version", "newer version"])
    ax.set_ylabel("ICC(2,1) with human cell means")
    ax.grid(visible=False)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    P.save_fig(fig, out / "Fig6_model_generation")


# ── Supplement ─────────────────────────────────────────────────────────────
def figS1(out: Path, conv: pd.DataFrame, vs: pd.DataFrame, main_model="gemini-4-argon"):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for m, g in conv.groupby("model"):
        ax.plot(g["k_runs"], g["ICC2_1_mean"], color="#bfc3c0", lw=.8, zorder=1)
    g = conv[conv["model"] == main_model]
    ax.plot(g["k_runs"], g["ICC2_1_mean"], color=BLUE, lw=2, label=main_model, zorder=3)
    ax.set_xticks(range(1, 11)); ax.set_xlabel("Runs averaged (k)"); ax.set_ylabel("ICC(2,1) with humans")
    ax.legend(fontsize=7, loc="center right")
    ax = axes[1]
    m = vs[vs["model"] == "MEAN"].iloc[0]
    parts = [("items", m["var_item"], "#6b8fb8"), ("image x item", m["var_image_x_item"], "#d98c6a"),
             ("image", m["var_image"], "#8ab5a0"), ("runs / residual", m["var_run_residual"], "#b0b0aa")]
    bottom, small = 0, []
    for name, val, c in parts:
        ax.bar(0, val, bottom=bottom, color=c, width=.6, edgecolor="white", lw=1.0)
        if val > 0.06:
            ax.text(0, bottom + val / 2, f"{name}\n{100 * val:.0f}%", ha="center", va="center", fontsize=7, color=INK)
        else:
            small.append((name, val, bottom + val / 2))
        bottom += val
    for k, (name, val, yc) in enumerate(sorted(small, key=lambda t: -t[2])):   # thin segments labelled beside the bar
        yt = 0.90 - 0.075 * k
        ax.annotate(f"{name} {100 * val:.1f}%", (0.3, yc), xytext=(0.4, yt), fontsize=7, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7))
    ax.set_xlim(-.5, 1.25); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_ylabel("Share of variance (mean over models)")
    ax.grid(visible=False)
    fig.tight_layout()
    P.save_fig(fig, out / "FigS1_run_stability")


def figS2(out: Path, A: np.ndarray, G: np.ndarray, order: list[int], main_model="gemini-4-argon"):
    H = A.mean(0)
    idx = [s - 1 for s in order]
    rdm = [squareform(pdist(M[idx], "correlation")) for M in (H, G)]
    vmax = max(r.max() for r in rdm)
    rho = stats.spearmanr(squareform(rdm[0], checks=False), squareform(rdm[1], checks=False))[0]
    fig = plt.figure(figsize=(W * .85, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cax = fig.add_subplot(gs[2])
    for ax, r, t in zip(axes, rdm, ("Human", main_model)):
        im = ax.imshow(r, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(10), order, fontsize=7); ax.set_yticks(range(10), order, fontsize=7)
        ax.set_title(t, fontsize=9); ax.grid(False); ax.set_xlabel("Scene (ordered by human NA)", fontsize=7.5)
    fig.colorbar(im, cax=cax).set_label("Correlation distance", fontsize=8)
    P.save_fig(fig, out / "FigS2_rsa_matrices")


def figS3(out: Path, parts: pd.DataFrame, mods: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(W * .85, 3.2))
    ax.hist(parts["r_with_others_mean"], bins=14, color="#9db7d6", edgecolor="white")
    med = parts["r_with_others_mean"].median()
    ax.axvline(med, color=INK, lw=1.4)
    ax.text(med + 0.008, ax.get_ylim()[1] * 0.60, f"participant\nmedian {med:.2f}", fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    y0 = -ax.get_ylim()[1] * 0.045
    for m in mods["mean_r_with_individuals"]:
        ax.plot([m, m], [y0 * 1.6, y0 * 0.2], color=ORANGE, lw=1.2, clip_on=False)
    best = mods.iloc[0]
    top = ax.get_ylim()[1]
    ax.text(0.675, y0 * 2.6, f"best model: {best['model']}, mean r = {best['mean_r_with_individuals']:.2f}",
            fontsize=7.2, ha="right", va="center")
    ax.set_xlabel("Correlation over the 200 item x scene cells")
    ax.set_ylabel("Participants")
    ax.legend(handles=[Patch(color="#9db7d6", label="participants: r with the mean of all others"),
                       plt.Line2D([], [], color=ORANGE, lw=2, label="models: mean r with individuals (one tick each)")],
              loc="upper left", fontsize=7)
    ax.set_xlim(0.35, 0.85); ax.set_ylim(y0 * 3.6, top)
    P.save_fig(fig, out / "FigS3_noise_ceiling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    batch = Path(args.batch)
    fd = batch / "figure_data"
    out = batch / "paper" / "main_figures"
    out.mkdir(parents=True, exist_ok=True)
    prov = P.provider_map()
    s1 = pd.read_csv(fd / "S1_scene_descriptives.csv")
    s3 = L.drop_excluded(pd.read_csv(fd / "S3_model_table.csv"))
    base = float(pd.read_csv(fd / "baseline.csv")["ICC2_1"].iloc[0])
    em = pd.read_csv(fd / "env_means.csv")
    sl = pd.read_csv(fd / "env_slopes.csv")
    isum = pd.read_csv(fd / "item_error_summary.csv", keep_default_na=False)   # "NA" is a subscale label, not missing
    vs = pd.read_csv(fd / "variance_shares.csv")
    parts = pd.read_csv(fd / "noise_ceiling_participants.csv")
    mods = pd.read_csv(fd / "noise_ceiling_models.csv")
    conv = L.drop_excluded(pd.read_csv(batch / "stability" / "convergence_by_k.csv"))
    info = T.ensure_info()
    A = L.impute_cell_mean(L.human_array(L.load_human_raw()[0]))
    rl = L.drop_excluded(L.load_runs_long())
    G = L.runs_array(rl[rl["model"] == "gemini-4-argon"]).mean(0)
    order = s1.sort_values("human_NA")["scene"].tolist()

    fig1(out, s1, info)
    fig2(out, em, sl)
    fig3(out, s3, base, prov)
    fig4(out, isum)
    fig5(out, s3, base, prov)
    fig6(out, s3)
    figS1(out, conv, vs)
    figS2(out, A, G, order)
    figS3(out, parts, mods)
    print(sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
