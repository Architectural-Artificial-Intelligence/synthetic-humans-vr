"""
Build every table and figure for the paper from one results batch.

Usage: python analysis/make_paper_outputs.py --batch results/20261013_234853
Reads : <batch>/summary/*, <batch>/stability/*, data/runs/*.jsonl, data/runs_long.csv,
        data/runs/model_config.json, results/demographics_table.csv
Writes: <batch>/paper/tables/*.csv|.tex|.md   <batch>/paper/figures/*.png|.pdf

Units (state in the paper): primary = 200 image x item cells (human = mean over
84 participants, model = mean over 10 runs); CIs = participant (+ run) bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import panas_lib as L

# Validated reference palette (categorical slots 1,2,3,7 + neutrals)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
PROVIDER_COLOR = {"openai": BLUE, "anthropic": ORANGE, "google": AQUA, "deepinfra": VIOLET}
PROVIDER_LABEL = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
                  "deepinfra": "Open-weight (DeepInfra)"}
PROVIDER_MARKER = {"openai": "o", "anthropic": "s", "google": "^", "deepinfra": "D"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


# ── helpers ────────────────────────────────────────────────────────────────
def save_fig(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, path: Path, caption: str, floatfmt="%.3f"):
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".tex").write_text(
        df.to_latex(index=False, float_format=lambda x: floatfmt % x, na_rep="--",
                    caption=caption, escape=True))
    path.with_suffix(".md").write_text(
        f"**{caption}**\n\n" + df.to_markdown(index=False, floatfmt=".3f") + "\n")


def ci(v, lo, hi, d=2):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def pfmt(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def provider_map():
    import sys
    sys.path.insert(0, str(L.ROOT))
    import vlm_prompt as v
    return {k: s.provider for k, s in v.ALL_MODELS.items()}


# ── tables ─────────────────────────────────────────────────────────────────
def table_reproducibility(out: Path, prov):
    cfg = json.loads((L.DATA_DIR / "runs" / "model_config.json").read_text())
    rows = []
    for m, c in cfg.items():
        if m in L.ALIAS_DUPLICATES:
            continue
        path = L.DATA_DIR / "runs" / f"{m}.jsonl"
        if not path.exists():
            continue
        log = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        ok = [r for r in log if r["status"] == "ok"]
        returned = sorted({r["returned_model"] for r in log if r.get("returned_model")})
        temps = sorted({str(r["temperature_applied"]) for r in log})
        rows.append({
            "Model key": m, "Provider": PROVIDER_LABEL[prov[m]],
            "Requested ID": c["model_id"], "Returned ID": "; ".join(returned),
            "Temperature requested": c.get("temperature_requested"),
            "Temperature applied": "default" if temps == ["None"] else ", ".join(temps),
            "Max tokens": c["max_tokens"],
            "Valid / total calls": f"{len(ok)}/{len(log)}",
            "Output tokens (total)": int(sum(r.get("out_tokens") or 0 for r in log)),
            "First call (UTC)": min(r["timestamp"] for r in log)[:10],
            "Last call (UTC)": max(r["timestamp"] for r in log)[:10],
        })
    df = pd.DataFrame(rows).sort_values(["Provider", "Model key"])
    save_table(df, out / "T1_model_reproducibility",
               "VLM configuration and run accounting (every call logged; seed per call = 1000 x image + call index where supported).")
    return df


def table_agreement(out: Path, comp: pd.DataFrame, prov):
    c = comp.sort_values("ICC2_1", ascending=False).copy()
    df = pd.DataFrame({
        "Rank": c["rank_point"],
        "95% rank interval": [f"{a:.0f}-{b:.0f}" for a, b in zip(c["rank_lo"], c["rank_hi"])],
        "Model": c["model"], "Provider": [PROVIDER_LABEL[prov[m]] for m in c["model"]],
        "ICC(2,1) [95% CI]": [ci(*r) for r in zip(c["ICC2_1"], c["ICC2_1_both_lo"], c["ICC2_1_both_hi"], [3] * len(c))],
        "ICC(2,k) [95% CI]": [ci(*r) for r in zip(c["ICC2_k"], c["ICC2_k_both_lo"], c["ICC2_k_both_hi"], [3] * len(c))],
        "Spearman rho [95% CI]": [ci(*r) for r in zip(c["spearman"], c["spearman_both_lo"], c["spearman_both_hi"], [3] * len(c))],
        "MAE [95% CI]": [ci(*r) for r in zip(c["MAE"], c["MAE_both_lo"], c["MAE_both_hi"], [3] * len(c))],
        "RMSE [95% CI]": [ci(*r) for r in zip(c["RMSE"], c["RMSE_both_lo"], c["RMSE_both_hi"], [3] * len(c))],
        "nRMSE (range)": c["nRMSE_range"], "nRMSE (human SD)": c["nRMSE_sd"],
        "Bias (human - model)": c["BA_bias"],
    })
    save_table(df, out / "T2_agreement_main",
               "Agreement with the human reference (unit: 200 image x item cells). ICC = two-way random, absolute agreement; CIs from a participant + run bootstrap (B=1000). nRMSE normalised by the scale range (4) and by the SD of all human ratings.")
    # RSA robustness
    r = pd.DataFrame({
        "Model": c["model"],
        "RSA correlation distance": [ci(*x, d=2) for x in zip(c["RSA_correlation"], c["RSA_correlation_both_lo"], c["RSA_correlation_both_hi"])],
        "RSA Euclidean": [ci(*x, d=2) for x in zip(c["RSA_euclidean"], c["RSA_euclidean_both_lo"], c["RSA_euclidean_both_hi"])],
        "RSA cosine": [ci(*x, d=2) for x in zip(c["RSA_cosine"], c["RSA_cosine_both_lo"], c["RSA_cosine_both_hi"])],
    })
    save_table(r, out / "T3_rsa_distance_metrics",
               "Representational similarity (Spearman between 10x10 image RDMs) under three distance metrics; participant + run bootstrap CIs.")
    return df


def table_bias(out: Path, comp, bias):
    b = bias.sort_values("overall_bias").copy()
    df = pd.DataFrame({
        "Model": b["model"],
        "Overall bias": b["overall_bias"], "p (Holm)": [pfmt(p) for p in b["overall_p_holm"]],
        "PA bias": b["PA_bias"], "PA p (Holm)": [pfmt(p) for p in b["PA_p_holm"]],
        "NA bias": b["NA_bias"], "NA p (Holm)": [pfmt(p) for p in b["NA_p_holm"]],
    })
    save_table(df, out / "T4_bias_tests",
               "Participant-level bias (human - model; N = 84 participants, one-sample t) for all ratings, PA and NA; Holm-adjusted across models. Positive = humans rate higher.")


def table_stability(out: Path, stab: pd.DataFrame):
    s = stab[stab["tag"] == "main"].sort_values("interrun_ICC2_1", ascending=False)
    df = pd.DataFrame({
        "Model": s["model"],
        "Mean cell SD": s["mean_cell_sd"], "% cells SD = 0": s["pct_cells_sd_zero"],
        "Inter-run ICC(2,1)": s["interrun_ICC2_1"], "Inter-run ICC(2,k=10)": s["interrun_ICC2_k"],
        "Split-half (SB)": s["interrun_splithalf_SB"],
        "Image-order tau (PA)": s["image_order_tau_PA_mean"], "Image-order tau (NA)": s["image_order_tau_NA_mean"],
        "Single-run ICC vs human (M)": s["single_run_ICC2_1_mean"], "Single-run ICC (SD)": s["single_run_ICC2_1_sd"],
        "Mean-of-10 ICC vs human": s["meanofruns_ICC2_1"],
        "Run/residual variance share": s["var_run_residual"],
    })
    save_table(df, out / "T5_run_stability",
               "Run-to-run stability of the 10 independent runs per image (unit: 200 cells). Inter-run ICC treats runs as raters; single-run columns evaluate each run separately against the human means.")
    return df


def table_human(out: Path, summ: Path, demo: Path):
    acc = json.loads((summ / "sample_accounting.json").read_text())
    imp = json.loads((summ / "imputation_sensitivity.json").read_text())
    rel = json.loads((summ / "reliability_hebrew_panas.json").read_text())
    rows = [("Rows in raw export", acc["rows_in_csv"]), ("Rows with all PANAS blank (excluded)", acc["rows_all_panas_blank"]),
            ("Duplicate timestamps in final sample", acc["duplicate_timestamps"]), ("Final analytic N", acc["final_N"]),
            ("Missing PANAS cells (%)", round(imp["pct_missing"], 2)),
            ("Participants with any missing", imp["participants_with_any_missing"]),
            ("Complete cases", imp["n_complete_cases"]),
            ("Cell means identical after imputation", imp["cell_means_identical_after_imputation"])]
    for sub in ("PA", "NA"):
        r = rel[sub]
        rows += [(f"{sub} Cronbach alpha, pooled [95% CI]", ci(r["alpha_pooled"], *r["alpha_pooled_ci"], d=3)),
                 (f"{sub} McDonald omega (total), pooled", round(r["omega_pooled"], 3)),
                 (f"{sub} alpha per image, range", f"{r['alpha_per_image_range'][0]:.2f}-{r['alpha_per_image_range'][1]:.2f}")]
    df = pd.DataFrame(rows, columns=["Item", "Value"]).astype({"Value": str})
    save_table(df, out / "T6_human_sample_reliability",
               "Human sample accounting, missing data, and internal consistency of the Hebrew PANAS subscales.")
    if demo.exists():
        d = pd.read_csv(demo).fillna("")
        save_table(d, out / "T7_demographics", "Participant demographics.")


def table_robustness(out: Path, summ: Path, comp):
    r = json.loads((summ / "robustness_summary.json").read_text())
    nc, sel, loo = r["noise_ceiling"], r["selection_validation_top5"], r["loo_image_ranking_kendall_tau"]
    rows = [("Leave-one-image-out: Kendall tau of model ranking vs full (min / mean)", f"{loo['min']:.2f} / {loo['mean']:.2f}"),
            ("Human split-half reliability (Spearman-Brown) [95% CI]", ci(nc["split_half_spearman_brown_mean"], *nc["split_half_spearman_brown_ci"], d=3)),
            ("Individual participant vs mean of others: mean r (noise-ceiling lower bound)", round(nc["individual_vs_others_mean_r"], 3)),
            ("Individual participant vs mean of all: mean r (upper bound)", round(nc["individual_vs_all_mean_r"], 3)),
            (f"Top-{sel['top_k']} selection: mean ICC on selection half", round(sel["mean_ICC_selected_on_A"], 3)),
            (f"Top-{sel['top_k']} selection: same models on held-out half", round(sel["mean_ICC_same_models_on_B"], 3)),
            ("Selection optimism (ICC)", round(sel["optimism"], 3)),
            (f"Mean overlap of top-{sel['top_k']} sets between halves", round(sel["mean_top_k_overlap"], 2))]
    save_table(pd.DataFrame(rows, columns=["Analysis", "Result"]).astype({"Result": str}),
               out / "T8_robustness_summary",
               "Robustness to the choice of images, participants, and model selection.")
    m = pd.DataFrame(nc["models"]).T.reset_index().rename(columns={"index": "Model"})
    save_table(m.sort_values("model_vs_individual_r_mean", ascending=False), out / "T9_noise_ceiling_by_model",
               "Mean correlation between each model's cell means and individual participants' ratings, and the share of participants whose agreement with the rest of the sample is lower than the model's.")


def table_t0(out: Path, stab):
    if "t0" not in set(stab["tag"]):
        return None
    a = stab[stab["tag"] == "main"].set_index("model")
    b = stab[stab["tag"] == "t0"].set_index("model")
    keep = b.index.intersection(a.index)
    df = pd.DataFrame({
        "Model": keep,
        "Mean cell SD (T=0.7)": a.loc[keep, "mean_cell_sd"].values, "Mean cell SD (T=0)": b.loc[keep, "mean_cell_sd"].values,
        "% cells SD=0 (T=0.7)": a.loc[keep, "pct_cells_sd_zero"].values, "% cells SD=0 (T=0)": b.loc[keep, "pct_cells_sd_zero"].values,
        "Inter-run ICC (T=0.7)": a.loc[keep, "interrun_ICC2_1"].values, "Inter-run ICC (T=0)": b.loc[keep, "interrun_ICC2_1"].values,
        "ICC vs human, mean of 10 (T=0.7)": a.loc[keep, "meanofruns_ICC2_1"].values,
        "ICC vs human, mean of 10 (T=0)": b.loc[keep, "meanofruns_ICC2_1"].values,
    })
    save_table(df, out / "T10_temperature_control",
               "Temperature control: the same 10 runs per image at T=0.7 (main) vs T=0.")
    return df


# ── figures ────────────────────────────────────────────────────────────────
def legend_providers(ax, **kw):
    h = [plt.Line2D([], [], marker=PROVIDER_MARKER[p], ls="", color=PROVIDER_COLOR[p], label=PROVIDER_LABEL[p],
                    markersize=6) for p in PROVIDER_COLOR]
    ax.legend(handles=h, **kw)


def fig_forest(out: Path, comp, prov):
    c = comp.sort_values("ICC2_1").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.19 * len(c) + 1.2))
    for i, r in c.iterrows():
        p = prov[r["model"]]
        ax.errorbar(r["ICC2_1"], i, xerr=[[r["ICC2_1"] - r["ICC2_1_both_lo"]], [r["ICC2_1_both_hi"] - r["ICC2_1"]]],
                    fmt=PROVIDER_MARKER[p], color=PROVIDER_COLOR[p], ecolor=PROVIDER_COLOR[p], ms=4.5,
                    elinewidth=1.2, capsize=0, alpha=.95)
    ax.set_yticks(range(len(c)), c["model"], fontsize=7.5)
    ax.set_xlabel("ICC(2,1) with human cell means (95% CI; participant + run bootstrap)")
    ax.set_title("Agreement with human PANAS ratings across 45 VLMs")
    ax.grid(axis="y", visible=False)
    legend_providers(ax, loc="lower right", fontsize=7.5)
    save_fig(fig, out / "F1_forest_icc")


def fig_rank_intervals(out: Path, comp, prov, top=25):
    c = comp.sort_values("rank_point").head(top).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(c) + 1.2))
    for i, r in c.iterrows():
        p = prov[r["model"]]
        ax.plot([r["rank_lo"], r["rank_hi"]], [i, i], color=PROVIDER_COLOR[p], lw=2.2, solid_capstyle="round", alpha=.6)
        ax.plot(r["rank_point"], i, PROVIDER_MARKER[p], color=PROVIDER_COLOR[p], ms=5)
    ax.set_yticks(range(len(c)), c["model"], fontsize=7.5)
    ax.set_xlabel("Rank by ICC(2,1) (95% bootstrap interval)")
    ax.set_title(f"Ranking uncertainty (top {top} models)")
    ax.invert_xaxis() if False else None
    ax.grid(axis="y", visible=False)
    legend_providers(ax, loc="lower right", fontsize=7.5)
    save_fig(fig, out / "F2_rank_intervals")


def model_cells(runs_long: pd.DataFrame, model: str):
    g = runs_long[(runs_long["model"] == model) & (runs_long["tag"] == "main")]
    return L.runs_array(g)


def fig_scatter_ba(out: Path, H, runs_long, comp, shown):
    n = len(shown)
    fig, axes = plt.subplots(2, n, figsize=(2.35 * n, 5.0))
    pa = np.tile(np.isin(np.arange(20), L.PA_IDX), (10, 1)).ravel()
    for j, m in enumerate(shown):
        G = model_cells(runs_long, m).mean(0)
        h, g = H.ravel(), G.ravel()
        row = comp[comp["model"] == m].iloc[0]
        ax = axes[0, j]
        ax.scatter(h[pa], g[pa], s=9, color=BLUE, alpha=.7, lw=0, label="PA items")
        ax.scatter(h[~pa], g[~pa], s=9, color=ORANGE, alpha=.7, lw=0, label="NA items")
        ax.plot([1, 5], [1, 5], color=GRAY, lw=1, ls="--")
        ax.set_xlim(1, 4.8); ax.set_ylim(1, 4.8)
        ax.set_title(f"{m}\nICC={row['ICC2_1']:.2f}, rho={row['spearman']:.2f}", fontsize=8)
        ax.set_xlabel("Human mean"); ax.set_ylabel("VLM mean" if j == 0 else "")
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax = axes[1, j]
        mean, diff = (h + g) / 2, h - g
        ax.scatter(mean[pa], diff[pa], s=9, color=BLUE, alpha=.7, lw=0)
        ax.scatter(mean[~pa], diff[~pa], s=9, color=ORANGE, alpha=.7, lw=0)
        b, sd = diff.mean(), diff.std(ddof=1)
        for y, ls in ((b, "-"), (b + 1.96 * sd, "--"), (b - 1.96 * sd, "--")):
            ax.axhline(y, color=INK, lw=.9, ls=ls)
        ax.axhline(0, color=GRAY, lw=.6)
        ax.set_xlabel("Mean of human and VLM"); ax.set_ylabel("Human - VLM" if j == 0 else "")
        ax.set_title(f"bias {b:+.2f}; LoA [{b - 1.96 * sd:+.2f}, {b + 1.96 * sd:+.2f}]", fontsize=7.5, color=MUTED)
        ax.set_ylim(-2.2, 2.2)
    fig.suptitle("Item-level agreement (200 cells per model)", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out / "F3_scatter_bland_altman")


def fig_rdm(out: Path, H, runs_long, best):
    G = model_cells(runs_long, best).mean(0)
    fig = plt.figure(figsize=(7.6, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.25, 1.25], wspace=0.35)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cax, ax3 = fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
    rd = [squareform(pdist(H, "correlation")), squareform(pdist(G, "correlation"))]
    vmax = max(r.max() for r in rd)
    for ax, r, t in zip(axes, rd, ("Human", best)):
        im = ax.imshow(r, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(10), range(1, 11), fontsize=7); ax.set_yticks(range(10), range(1, 11), fontsize=7)
        ax.set_title(t, fontsize=9); ax.grid(False)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Correlation distance", fontsize=8)
    iu = np.triu_indices(10, 1)
    ax3.scatter(rd[0][iu], rd[1][iu], s=14, color=BLUE, alpha=.8, lw=0)
    rho = stats.spearmanr(rd[0][iu], rd[1][iu])[0]
    ax3.set_xlabel("Human RDM entry"); ax3.set_ylabel("VLM RDM entry")
    ax3.set_title(f"RSA rho = {rho:.2f}", fontsize=9)
    fig.suptitle("Representational geometry of the 10 environments", x=0.01, ha="left", fontweight="bold", fontsize=10, y=1.04)
    save_fig(fig, out / "F4_rsa_rdm")


def fig_profiles(out: Path, A, runs_long, shown):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    P = A.shape[0]
    x = np.arange(1, 11)
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    for ax, (name, idx) in zip(axes, (("Positive affect (PA)", L.PA_IDX), ("Negative affect (NA)", L.NA_IDX))):
        hp = A[:, :, idx].mean(2)                       # participants x images
        m, se = hp.mean(0), hp.std(0, ddof=1) / np.sqrt(P)
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=GRAY, alpha=.3, lw=0)
        ax.plot(x, m, color=INK, lw=2, label="Humans (95% CI)")
        for c, mod in zip(cols, shown):
            G = model_cells(runs_long, mod).mean(0)[:, idx].mean(1)
            ax.plot(x, G, color=c, lw=1.2, marker="o", ms=3, label=mod)
        ax.set_xticks(x); ax.set_xlabel("Environment"); ax.set_ylabel("Mean rating (1-5)")
        ax.set_title(name)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    save_fig(fig, out / "F5_pa_na_profiles")


def fig_bias_pa_na(out: Path, bias, prov):
    b = bias.sort_values("NA_bias").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 0.19 * len(b) + 1.2))
    for i, r in b.iterrows():
        ax.plot([r["PA_bias"], r["NA_bias"]], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(r["PA_bias"], i, "o", color=BLUE, ms=4.5, mfc=BLUE if r["PA_p_holm"] < .05 else "white", zorder=2)
        ax.plot(r["NA_bias"], i, "o", color=ORANGE, ms=4.5, mfc=ORANGE if r["NA_p_holm"] < .05 else "white", zorder=2)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(b)), b["model"], fontsize=7.5)
    ax.set_xlabel("Bias = human - VLM (rating points); filled = Holm p < .05")
    ax.set_title("Systematic offsets: positive vs negative affect")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, label="PA"),
                       plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="NA")], loc="lower right")
    save_fig(fig, out / "F6_bias_pa_na")


def fig_stability(out: Path, stab, prov):
    s = stab[stab["tag"] == "main"].sort_values("meanofruns_ICC2_1").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.19 * len(s) + 1.4), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
    ax = axes[0]
    for i, r in s.iterrows():
        ax.plot([r["single_run_ICC2_1_min"], r["single_run_ICC2_1_max"]], [i, i], color=GRAY, lw=2, alpha=.5, zorder=1)
        ax.plot(r["single_run_ICC2_1_mean"], i, "o", color=ORANGE, ms=3.5, zorder=2)
        ax.plot(r["meanofruns_ICC2_1"], i, "o", color=BLUE, ms=4.2, zorder=3)
    ax.set_yticks(range(len(s)), s["model"], fontsize=7.5)
    ax.set_xlabel("ICC(2,1) with humans")
    ax.set_title("Single runs vs mean of 10")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="single run: mean (bar = range of 10)"),
                       plt.Line2D([], [], marker="o", ls="", color=BLUE, label="mean of 10 runs")],
              loc="lower right", fontsize=7)
    ax = axes[1]
    ax.hlines(range(len(s)), 0.85, s["interrun_ICC2_1"], color=GRID, lw=1.2)
    ax.plot(s["interrun_ICC2_1"], range(len(s)), "o", color=BLUE, ms=4.2)
    ax.set_xlim(.85, 1.0); ax.set_xlabel("Inter-run ICC(2,1)")
    ax.set_title("Run-to-run reliability")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, out / "F7_run_stability")


def fig_convergence(out: Path, conv, comp, prov):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    top = comp.sort_values("ICC2_1", ascending=False)["model"].head(3).tolist()
    for m, g in conv.groupby("model"):
        axes[0].plot(g["k_runs"], g["ICC2_1_mean"], color=GRAY, lw=.8, alpha=.5)
    for c, m in zip((BLUE, ORANGE, AQUA), top):
        g = conv[conv["model"] == m]
        axes[0].plot(g["k_runs"], g["ICC2_1_mean"], color=c, lw=1.8, label=m)
    axes[0].set_xlabel("Runs averaged (k)"); axes[0].set_ylabel("ICC(2,1) with humans")
    axes[0].set_title("Convergence of the run average"); axes[0].legend(fontsize=7)
    axes[0].set_xticks(range(1, 11))
    gain = conv.pivot(index="model", columns="k_runs", values="ICC2_1_mean")
    d = (gain[10] - gain[1])
    axes[1].hist(d, bins=15, color=BLUE, edgecolor="white")
    axes[1].set_xlabel("ICC gain, 10 runs vs 1 run"); axes[1].set_ylabel("Models")
    axes[1].set_title("Benefit of averaging")
    fig.tight_layout()
    save_fig(fig, out / "F8_convergence")


def fig_loo(out: Path, summ: Path, comp, prov):
    loo = pd.read_csv(summ / "loo_image_metrics.csv")
    rng = loo.groupby("model")["ICC2_1"].agg(["min", "max", "mean"])
    full = comp.set_index("model")["ICC2_1"]
    order = full.sort_values().index
    fig, ax = plt.subplots(figsize=(6.6, 0.19 * len(order) + 1.2))
    for i, m in enumerate(order):
        ax.plot([rng.loc[m, "min"], rng.loc[m, "max"]], [i, i], color=PROVIDER_COLOR[prov[m]], lw=2.4, alpha=.45)
        ax.plot(full[m], i, PROVIDER_MARKER[prov[m]], color=PROVIDER_COLOR[prov[m]], ms=4.5)
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.set_xlabel("ICC(2,1): all 10 images (marker) and range when one image is left out (bar)")
    ax.set_title("Leave-one-environment-out sensitivity")
    ax.grid(axis="y", visible=False)
    legend_providers(ax, loc="lower right", fontsize=7.5)
    save_fig(fig, out / "F9_leave_one_image_out")


def fig_noise_ceiling(out: Path, summ: Path, prov, top=20):
    r = json.loads((summ / "robustness_summary.json").read_text())["noise_ceiling"]
    m = pd.DataFrame(r["models"]).T.sort_values("model_vs_individual_r_mean").tail(top)
    fig, ax = plt.subplots(figsize=(6.2, 0.2 * len(m) + 1.3))
    ax.axvspan(r["individual_vs_others_mean_r"], r["individual_vs_all_mean_r"], color=GRAY, alpha=.3, lw=0,
               label="noise ceiling (lower-upper)")
    for i, (name, row) in enumerate(m.iterrows()):
        ax.plot(row["model_vs_individual_r_mean"], i, PROVIDER_MARKER[prov[name]], color=PROVIDER_COLOR[prov[name]], ms=5)
    ax.set_yticks(range(len(m)), m.index, fontsize=7.5)
    ax.set_xlabel("Mean r between model cell means and individual participants' ratings")
    ax.set_title(f"Model vs individual participants (top {top})")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, out / "F10_noise_ceiling")


def fig_alpha(out: Path, summ: Path):
    rel = json.loads((summ / "reliability_hebrew_panas.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(1, 11)
    for sub, c in (("PA", BLUE), ("NA", ORANGE)):
        ax.plot(x, rel[sub]["alpha_per_image"], "o-", color=c, lw=1.4, ms=4, label=f"{sub} (pooled alpha = {rel[sub]['alpha_pooled']:.2f})")
    ax.axhline(.7, color=GRAY, ls="--", lw=.8)
    ax.set_xticks(x); ax.set_xlabel("Environment"); ax.set_ylabel("Cronbach's alpha")
    ax.set_ylim(.5, 1); ax.set_title("Internal consistency of the Hebrew PANAS"); ax.legend(fontsize=7.5)
    save_fig(fig, out / "F11_hebrew_panas_alpha")


def fig_t0(out: Path, t0df):
    if t0df is None or t0df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(len(t0df)); w = .38
    for ax, a, b, t in ((axes[0], "Mean cell SD (T=0.7)", "Mean cell SD (T=0)", "Within-cell SD across runs"),
                        (axes[1], "Inter-run ICC (T=0.7)", "Inter-run ICC (T=0)", "Inter-run ICC(2,1)")):
        ax.bar(x - w / 2, t0df[a], w, color=BLUE, label="T = 0.7")
        ax.bar(x + w / 2, t0df[b], w, color=ORANGE, label="T = 0")
        ax.set_xticks(x, t0df["Model"], rotation=25, ha="right", fontsize=7.5)
        ax.set_title(t); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, out / "F12_temperature_control")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    batch = Path(args.batch)
    summ, stabd = batch / "summary", batch / "stability"
    tdir, fdir = batch / "paper" / "tables", batch / "paper" / "figures"
    tdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)

    prov = provider_map()
    comp = pd.read_csv(summ / "model_comparison.csv")
    bias = pd.read_csv(summ / "bias_tests.csv")
    stab = pd.read_csv(stabd / "stability_by_model.csv")
    conv = pd.read_csv(stabd / "convergence_by_k.csv")
    runs_long = L.drop_aliases(pd.read_csv(L.DATA_DIR / "runs_long.csv"))
    A = L.impute_cell_mean(L.human_array(L.load_human_raw()[0]))
    H = A.mean(0)

    table_reproducibility(tdir, prov)
    table_agreement(tdir, comp, prov)
    table_bias(tdir, comp, bias)
    table_stability(tdir, stab)
    table_human(tdir, summ, L.ROOT / "results" / "demographics_table.csv")
    table_robustness(tdir, summ, comp)
    t0df = table_t0(tdir, stab)

    ranked = comp.sort_values("ICC2_1", ascending=False)["model"].tolist()
    shown = ranked[:3] + [ranked[len(ranked) // 2], ranked[-1]]
    fig_forest(fdir, comp, prov)
    fig_rank_intervals(fdir, comp, prov)
    fig_scatter_ba(fdir, H, runs_long, comp, shown)
    fig_rdm(fdir, H, runs_long, ranked[0])
    fig_profiles(fdir, A, runs_long, ranked[:3])
    fig_bias_pa_na(fdir, bias, prov)
    fig_stability(fdir, stab, prov)
    fig_convergence(fdir, conv, comp, prov)
    fig_loo(fdir, summ, comp, prov)
    fig_noise_ceiling(fdir, summ, prov)
    fig_alpha(fdir, summ)
    fig_t0(fdir, t0df)
    print("tables:", sorted(p.name for p in tdir.glob("*.csv")))
    print("figures:", sorted(p.name for p in fdir.glob("*.png")))


if __name__ == "__main__":
    main()
