"""
Robust human-vs-VLM inference using the full 84 x 10 x 20 human data.

Addresses reviewer requests: CIs for every metric, explicit units, ICC forms,
MAE/RMSE normalisation, RSA metrics, leave-one-image-out, noise ceiling,
selection-bias control (top-5), multiple-comparison correction, individual-level
slopes, Cronbach alpha / McDonald omega, imputation sensitivity, sample accounting.

Unit of analysis: the 200 (image x item) cells (PA/NA x image as secondary).
Resampling unit for CIs: PARTICIPANTS (cluster bootstrap; B replicates) and, for
the "both" interval, model RUNS within each model as well.

Usage
-----
  python analysis/robust_inference.py                      # data/runs_long.csv
  python analysis/robust_inference.py --legacy --B 500     # old mean-only CSVs
  python analysis/robust_inference.py --mixed              # + MixedLM slopes
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

import panas_lib as L

SEED = 2026
KEY_METRICS = ["ICC2_1", "ICC2_k", "pearson", "spearman", "MAE", "RMSE",
               "nMAE_range", "nRMSE_range", "nMAE_sd", "nRMSE_sd", "BA_bias",
               "r_within_item", "RSA_correlation", "RSA_euclidean", "RSA_cosine"]
# metrics that are meaningful when IMAGES are resampled (RSA is not: duplicated
# images create zero distances in both RDMs; use leave-one-image-out instead)
IMG_METRICS = [m for m in KEY_METRICS if not m.startswith("RSA")]


def human_cells(arr):                       # (P,10,20) -> (10,20) cell means
    return np.nanmean(arr, axis=0)


def load_models(args) -> dict[str, np.ndarray]:
    """model -> array (runs, images, items). Legacy: a single 'run' (the mean)."""
    out = {}
    if args.legacy:
        for p in sorted((L.DATA_DIR / "legacy_2026-03").glob("*.csv")):
            g = L.load_legacy_means(p)
            if not np.isnan(g).any():
                out[p.stem] = g[None]
        return out
    df = L.load_runs_long(Path(args.runs_long) if args.runs_long else None, tag=args.tag)
    for model, g in df.groupby("model"):
        arr = L.runs_array(g)
        if np.isnan(arr).any() or g["image"].nunique() < L.N_IMAGES:
            print(f"  [skip] {model}: incomplete runs")
            continue
        out[model] = arr
    return out


# ── Sample accounting + reliability ────────────────────────────────────────
def reliability(arr: np.ndarray, rng, B: int) -> dict:
    """Cronbach alpha and McDonald omega_total for PA and NA, pooled over images
    (rows = participant x image) and per image."""
    def alpha(x):
        k = x.shape[1]
        return k / (k - 1) * (1 - x.var(0, ddof=1).sum() / x.sum(1).var(ddof=1))

    def omega(x):
        z = (x - x.mean(0)) / x.std(0, ddof=1)
        fa = FactorAnalysis(1).fit(z)
        lam = np.abs(fa.components_[0])
        psi = fa.noise_variance_
        return lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum())

    out = {}
    for name, idx in (("PA", L.PA_IDX), ("NA", L.NA_IDX)):
        pooled = arr[:, :, idx].reshape(-1, len(idx))          # (P*10, 10)
        per_image = [alpha(arr[:, i, idx]) for i in range(L.N_IMAGES)]
        P = arr.shape[0]
        boots = []
        for _ in range(B):
            b = rng.integers(0, P, P)
            boots.append(alpha(arr[b][:, :, idx].reshape(-1, len(idx))))
        out[name] = {
            "alpha_pooled": float(alpha(pooled)),
            "alpha_pooled_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "omega_pooled": float(omega(pooled)),
            "alpha_per_image": [float(a) for a in per_image],
            "alpha_per_image_range": [float(min(per_image)), float(max(per_image))],
        }
    return out


def imputation_sensitivity(raw: np.ndarray) -> dict:
    miss = np.isnan(raw)
    res = {"n_missing": int(miss.sum()), "pct_missing": float(100 * miss.mean()),
           "participants_with_any_missing": int(miss.any(axis=(1, 2)).sum())}
    imp = L.impute_cell_mean(raw)
    res["cell_means_identical_after_imputation"] = bool(
        np.allclose(np.nanmean(raw, 0), imp.mean(0)))
    cc = raw[~miss.any(axis=(1, 2))]
    res["n_complete_cases"] = int(cc.shape[0])
    res["max_abs_cell_mean_shift_complete_case_vs_imputed"] = float(
        np.abs(np.nanmean(cc, 0) - imp.mean(0)).max()) if len(cc) else None
    res["sd_all_ratings_imputed"] = float(imp.std(ddof=1))
    res["sd_all_ratings_observed_only"] = float(np.nanstd(raw, ddof=1))
    return res


# ── Core resampling ────────────────────────────────────────────────────────
def bootstrap_metrics(A, runs, B, rng, human_sd):
    """Return dict model -> {metric: (point, lo, hi) for 'part' and 'both'} and
    the ICC2_1 replicate matrix (for rank intervals)."""
    P = A.shape[0]
    pidx = rng.integers(0, P, (B, P))
    H_b = np.stack([np.nanmean(A[i], axis=0) for i in pidx])       # (B,10,20)
    H0 = human_cells(A)
    iidx = rng.integers(0, L.N_IMAGES, (B, L.N_IMAGES))             # image bootstrap
    results, icc_rep = {}, {}
    for model, R in runs.items():
        G0 = R.mean(0)
        point = L.agreement_metrics(H0, G0, human_sd=human_sd)
        img = pd.DataFrame([L.agreement_metrics(H0[i], G0[i], rsa_metrics=(), human_sd=human_sd)
                            for i in iidx])
        ridx = rng.integers(0, R.shape[0], (B, R.shape[0]))
        reps = {"part": [], "both": []}
        for b in range(B):
            reps["part"].append(L.agreement_metrics(H_b[b], G0, human_sd=human_sd))
            Gb = R[ridx[b]].mean(0) if R.shape[0] > 1 else G0
            reps["both"].append(L.agreement_metrics(H_b[b], Gb, human_sd=human_sd))
        entry = {"point": point,
                 "img": {m: np.nanpercentile(img[m], [2.5, 97.5]).tolist() for m in IMG_METRICS}}
        for kind in ("part", "both"):
            df = pd.DataFrame(reps[kind])
            entry[kind] = {m: np.nanpercentile(df[m], [2.5, 97.5]).tolist()
                           for m in df.columns}
            if kind == "both":
                icc_rep[model] = df["ICC2_1"].values
        results[model] = entry
    return results, icc_rep


def loo_images(H, runs, human_sd):
    rows = []
    for model, R in runs.items():
        G = R.mean(0)
        for left in range(L.N_IMAGES):
            keep = [i for i in range(L.N_IMAGES) if i != left]
            m = L.agreement_metrics(H[keep], G[keep], human_sd=None)
            m.update(model=model, left_out=left + 1)
            rows.append(m)
    return pd.DataFrame(rows)


def noise_ceiling(A, runs, rng, n_splits=1000):
    """Individual-participant ceiling (Nili et al.): lower = corr(participant,
    mean of the others); upper = corr(participant, mean of all). Model-to-
    participant correlation is compared with the lower bound."""
    P = A.shape[0]
    cells = A.reshape(P, -1)
    allmean = np.nanmean(cells, 0)
    lower, upper = [], []
    for p in range(P):
        others = np.nanmean(np.delete(cells, p, 0), 0)
        lower.append(np.corrcoef(cells[p], others)[0, 1])
        upper.append(np.corrcoef(cells[p], allmean)[0, 1])
    res = {"individual_vs_others_mean_r": float(np.mean(lower)),
           "individual_vs_all_mean_r": float(np.mean(upper))}
    sb = []
    for _ in range(n_splits):
        perm = rng.permutation(P)
        a, b = np.nanmean(cells[perm[:P // 2]], 0), np.nanmean(cells[perm[P // 2:]], 0)
        r = np.corrcoef(a, b)[0, 1]
        sb.append(2 * r / (1 + r))
    res["split_half_spearman_brown_mean"] = float(np.mean(sb))
    res["split_half_spearman_brown_ci"] = np.percentile(sb, [2.5, 97.5]).tolist()
    models = {}
    for model, R in runs.items():
        g = R.mean(0).ravel()
        r = [np.corrcoef(cells[p], g)[0, 1] for p in range(P)]
        models[model] = {"model_vs_individual_r_mean": float(np.mean(r)),
                         "prop_participants_below_model":
                             float(np.mean(np.array(lower) < np.array(r)))}
    res["models"] = models
    return res


def selection_validation(A, runs, rng, n_rep=1000, top=5):
    """Pick top-`top` models by ICC2_1 on a random half of participants (A),
    evaluate the same models on the other half (B)."""
    P = A.shape[0]
    names = list(runs)
    G = {m: runs[m].mean(0) for m in names}
    sel_a, sel_b, overlap, best_b = [], [], [], []
    for _ in range(n_rep):
        perm = rng.permutation(P)
        HA, HB = human_cells(A[perm[:P // 2]]), human_cells(A[perm[P // 2:]])
        ia = np.array([L.icc2(HA.ravel(), G[m].ravel())[0] for m in names])
        ib = np.array([L.icc2(HB.ravel(), G[m].ravel())[0] for m in names])
        ta, tb = np.argsort(-ia)[:top], np.argsort(-ib)[:top]
        sel_a.append(ia[ta].mean()); sel_b.append(ib[ta].mean())
        best_b.append(ib[tb].mean()); overlap.append(len(set(ta) & set(tb)))
    return {"top_k": top, "n_replicates": n_rep,
            "mean_ICC_selected_on_A": float(np.mean(sel_a)),
            "mean_ICC_same_models_on_B": float(np.mean(sel_b)),
            "optimism": float(np.mean(sel_a) - np.mean(sel_b)),
            "mean_ICC_best_models_on_B": float(np.mean(best_b)),
            "mean_top_k_overlap": float(np.mean(overlap)),
            "overlap_distribution": np.bincount(overlap, minlength=top + 1).tolist()}


def bias_tests(A, runs):
    """Participant-level bias tests (N = participants): overall, PA, NA.
    Participant's mean rating vs the model's mean; one-sample t over participants."""
    rows = []
    for model, R in runs.items():
        G = R.mean(0)
        row = {"model": model}
        for name, idx in (("overall", np.arange(20)), ("PA", L.PA_IDX), ("NA", L.NA_IDX)):
            h = np.nanmean(A[:, :, idx], axis=(1, 2))
            d = h - G[:, idx].mean()
            t, p = stats.ttest_1samp(d, 0.0)
            row.update({f"{name}_bias": float(d.mean()), f"{name}_t": float(t),
                        f"{name}_p": float(p)})
        rows.append(row)
    df = pd.DataFrame(rows)
    for name in ("overall", "PA", "NA"):
        p = df[f"{name}_p"].values
        df[f"{name}_p_holm"] = multipletests(p, method="holm")[1]
        df[f"{name}_p_fdr"] = multipletests(p, method="fdr_bh")[1]
    return df


def individual_slopes(A, runs, mixed=False):
    """Individual-level: rating_{p,cell} ~ model_cell. OLS with participant-
    clustered SEs (honest about N=84 participants, NOT about the 10 images)."""
    P = A.shape[0]
    rows = []
    for model, R in runs.items():
        g = R.mean(0).ravel()
        y = A.reshape(P, -1)
        df = pd.DataFrame({"y": y.ravel(), "x": np.tile(g, P),
                           "pid": np.repeat(np.arange(P), g.size),
                           "img": np.tile(np.repeat(np.arange(L.N_IMAGES), 20), P)})
        df = df.dropna()
        fit = smf.ols("y ~ x", df).fit(cov_type="cluster", cov_kwds={"groups": df["pid"]})
        ci = fit.conf_int().loc["x"]
        row = {"model": model, "slope": fit.params["x"], "slope_lo": ci[0],
               "slope_hi": ci[1], "intercept": fit.params["Intercept"],
               "r2_individual": fit.rsquared}
        if mixed:
            try:
                m = smf.mixedlm("y ~ x", df, groups=df["pid"],
                                vc_formula={"img": "0 + C(img)"}).fit(reml=True)
                row.update(mixed_slope=m.params["x"], mixed_se=m.bse["x"])
            except Exception as exc:                      # noqa: BLE001
                row["mixed_error"] = str(exc)[:80]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Output ────────────────────────────────────────────────────────────────
def forest_plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table.sort_values("ICC2_1")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.errorbar(t["ICC2_1"], y, xerr=[t["ICC2_1"] - t["ICC2_1_both_lo"],
                                       t["ICC2_1_both_hi"] - t["ICC2_1"]],
                fmt="o", color="#2a5d9f", ecolor="#8fb0d9", capsize=2)
    ax.set_yticks(y, t["model"])
    ax.set_xlabel("ICC(2,1) vs human (95% CI, participants + runs bootstrap)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--runs-long", default=None, help="path to runs_long.csv")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out = Path(args.out) if args.out else (
        L.ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S") / "summary")
    out.mkdir(parents=True, exist_ok=True)

    df_h, acc = L.load_human_raw()
    raw = L.human_array(df_h)
    A = L.impute_cell_mean(raw)                 # same imputation as the paper
    human_sd = float(A.std(ddof=1))
    (out / "sample_accounting.json").write_text(json.dumps(acc, indent=2))
    (out / "imputation_sensitivity.json").write_text(
        json.dumps(imputation_sensitivity(raw), indent=2))
    (out / "reliability_hebrew_panas.json").write_text(
        json.dumps(reliability(A, rng, min(args.B, 1000)), indent=2))
    print("Human:", acc)

    runs = load_models(args)
    print(f"Models: {len(runs)}")
    H0 = human_cells(A)

    boot, icc_rep = bootstrap_metrics(A, runs, args.B, rng, human_sd)
    rows = []
    for model, e in boot.items():
        row = {"model": model, "n_runs": runs[model].shape[0], **e["point"]}
        for m in KEY_METRICS:
            row[f"{m}_part_lo"], row[f"{m}_part_hi"] = e["part"][m]
            row[f"{m}_both_lo"], row[f"{m}_both_hi"] = e["both"][m]
            row[f"{m}_img_lo"], row[f"{m}_img_hi"] = e["img"].get(m, [np.nan, np.nan])
        rows.append(row)
    table = pd.DataFrame(rows)

    # rank intervals (ICC2_1) from the joint bootstrap
    M = np.vstack([icc_rep[m] for m in table["model"]])          # (models, B)
    ranks = (-M).argsort(0).argsort(0) + 1
    table["rank_point"] = table["ICC2_1"].rank(ascending=False).astype(int)
    table["rank_lo"], table["rank_hi"] = (np.percentile(ranks, 2.5, axis=1),
                                          np.percentile(ranks, 97.5, axis=1))

    loo = loo_images(H0, runs, human_sd)
    loo.to_csv(out / "loo_image_metrics.csv", index=False)
    full_rank = table.set_index("model")["ICC2_1"]
    taus = []
    for left, g in loo.groupby("left_out"):
        s = g.set_index("model")["ICC2_1"].reindex(full_rank.index)
        taus.append(stats.kendalltau(full_rank, s)[0])
    table.attrs["loo_kendall_tau"] = taus

    ba_rows = []
    for model, R in runs.items():
        ba_rows.append({"model": model,
                        **L.bland_altman_checks(H0.ravel(), R.mean(0).ravel())})
    pd.DataFrame(ba_rows).to_csv(out / "bland_altman_checks.csv", index=False)

    bias = bias_tests(A, runs)
    slopes = individual_slopes(A, runs, mixed=args.mixed)
    table = table.merge(bias[["model", "overall_p_holm", "PA_p_holm", "NA_p_holm"]], on="model")
    table = table.merge(slopes[["model", "slope", "slope_lo", "slope_hi"]]
                        .rename(columns={"slope": "indiv_slope", "slope_lo": "indiv_slope_lo",
                                         "slope_hi": "indiv_slope_hi"}), on="model")
    table.sort_values("ICC2_1", ascending=False).to_csv(out / "model_comparison.csv", index=False)
    bias.to_csv(out / "bias_tests.csv", index=False)
    slopes.to_csv(out / "individual_level_slopes.csv", index=False)

    base = L.item_mean_baseline(H0)
    base_pt = L.agreement_metrics(H0, base, rsa_metrics=(), human_sd=human_sd)
    iidx = rng.integers(0, L.N_IMAGES, (args.B, L.N_IMAGES))
    base_img = pd.DataFrame([L.agreement_metrics(H0[i], L.item_mean_baseline(H0[i]), rsa_metrics=(),
                                                 human_sd=human_sd) for i in iidx])
    baseline = {"description": "no-image model: each image gets the human item means over images",
                **{m: base_pt[m] for m in ("ICC2_1", "ICC2_k", "pearson", "spearman", "MAE", "RMSE")},
                "ICC2_1_img_ci": np.nanpercentile(base_img["ICC2_1"], [2.5, 97.5]).tolist(),
                "n_models_ICC2_1_above_baseline": int((table["ICC2_1"] > base_pt["ICC2_1"]).sum()),
                "n_models": int(len(table))}
    summary = {
        "item_mean_baseline": baseline,
        "loo_image_ranking_kendall_tau": {"values": [float(t) for t in taus],
                                          "min": float(np.min(taus)), "mean": float(np.mean(taus))},
        "noise_ceiling": noise_ceiling(A, runs, rng),
        "selection_validation_top5": selection_validation(A, runs, rng, top=min(5, len(runs))),
        "settings": {"B": args.B, "seed": SEED, "tag": args.tag, "legacy": args.legacy,
                     "human_sd_used_for_nMAE_sd": human_sd, "scale_range": L.SCALE_RANGE,
                     "ICC": "ICC(2,1) single / ICC(2,k=2) average; two-way random, "
                            "absolute agreement; targets = 200 image x item cells; "
                            "raters = human mean, model mean"},
    }
    (out / "robustness_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    forest_plot(table, out / "forest_icc.png")
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
