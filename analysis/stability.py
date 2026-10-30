"""
Run-to-run stability of VLM PANAS ratings (data/runs_long.csv, every run kept).

Per model (tag 'main'; 't0' control compared if present):
  1. within-cell variability (SD / range across runs, % cells with SD == 0)
  2. single-run agreement with humans (ICC2_1, Spearman, MAE, RMSE): mean/SD/range
     over runs, vs agreement of the mean of all runs
  3. inter-run reliability: ICC(2,1) and ICC(2,k) with runs as raters over the
     200 cells, Cronbach alpha, mean pairwise run-run Spearman, split-half
     (random 5 vs 5 runs, Spearman-Brown)
  4. convergence: agreement with humans vs number of averaged runs k = 1..R
  5. image-ordering stability: Kendall tau of the PA / NA image ranking across runs
  6. variance decomposition (image, item, image x item, run/residual)
  7. T=0 vs T=0.7 comparison when a 't0' tag exists

Usage: python analysis/stability.py [--runs-long data/runs_long.csv] [--out DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import panas_lib as L

SEED = 2026
N_SUBSETS = 200          # random run subsets per k in the convergence curve


def icc_runs(R: np.ndarray) -> tuple[float, float]:
    """ICC(2,1)/ICC(2,k) with runs as raters. R: (runs, cells)."""
    d = R.T                                    # (cells, runs)
    n, k = d.shape
    gm = d.mean()
    msr = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    if msr + (k - 1) * mse + k * (msc - mse) / n == 0:
        return np.nan, np.nan
    return (float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)),
            float((msr - mse) / (msr + (msc - mse) / n)))


def cronbach_alpha_runs(R):
    k = R.shape[0]
    tot = R.sum(0).var(ddof=1)
    return float(k / (k - 1) * (1 - R.var(1, ddof=1).sum() / tot)) if tot > 0 else np.nan


def safe_spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b)[0])


def variance_components(runs: np.ndarray) -> dict:
    """Balanced random-effects ANOVA for image x item x run (one score per cell
    per run): shares of variance from image, item, image x item, run/residual."""
    R, I, K = runs.shape
    gm = runs.mean()
    m_i, m_k, m_ik = runs.mean((0, 2)), runs.mean((0, 1)), runs.mean(0)
    ss_i = R * K * ((m_i - gm) ** 2).sum()
    ss_k = R * I * ((m_k - gm) ** 2).sum()
    ss_ik = R * ((m_ik - m_i[:, None] - m_k[None, :] + gm) ** 2).sum()
    ss_e = ((runs - m_ik[None]) ** 2).sum()
    ms_i, ms_k = ss_i / (I - 1), ss_k / (K - 1)
    ms_ik, ms_e = ss_ik / ((I - 1) * (K - 1)), ss_e / (I * K * (R - 1))
    v_e = ms_e
    v_ik = max((ms_ik - ms_e) / R, 0)
    v_i = max((ms_i - ms_ik) / (R * K), 0)
    v_k = max((ms_k - ms_ik) / (R * I), 0)
    tot = v_e + v_ik + v_i + v_k
    return {"var_image": v_i / tot, "var_item": v_k / tot,
            "var_image_x_item": v_ik / tot, "var_run_residual": v_e / tot}


def model_stability(runs: np.ndarray, H: np.ndarray, rng) -> dict:
    R = runs.shape[0]
    cells = runs.reshape(R, -1)
    h = H.ravel()
    sd_cell = cells.std(0, ddof=1)
    out = {"n_runs": R, "mean_cell_sd": float(sd_cell.mean()),
           "median_cell_sd": float(np.median(sd_cell)),
           "max_cell_sd": float(sd_cell.max()),
           "pct_cells_sd_zero": float(100 * (sd_cell < 1e-9).mean()),
           "mean_cell_range": float(np.ptp(cells, axis=0).mean())}

    # single-run vs mean-of-runs agreement with humans
    per_run = pd.DataFrame([L.agreement_metrics(H, runs[r], rsa_metrics=("correlation",))
                            for r in range(R)])
    for m in ("ICC2_1", "spearman", "MAE", "RMSE", "RSA_correlation"):
        out[f"single_run_{m}_mean"] = float(per_run[m].mean())
        out[f"single_run_{m}_sd"] = float(per_run[m].std(ddof=1))
        out[f"single_run_{m}_min"] = float(per_run[m].min())
        out[f"single_run_{m}_max"] = float(per_run[m].max())
    mean_m = L.agreement_metrics(H, runs.mean(0), rsa_metrics=("correlation",))
    for m in ("ICC2_1", "spearman", "MAE", "RMSE", "RSA_correlation"):
        out[f"meanofruns_{m}"] = mean_m[m]

    # inter-run reliability
    i1, ik = icc_runs(cells)
    out["interrun_ICC2_1"], out["interrun_ICC2_k"] = i1, ik
    out["interrun_alpha"] = cronbach_alpha_runs(cells)
    pair = [safe_spearman(cells[a], cells[b]) for a, b in itertools.combinations(range(R), 2)]
    out["interrun_pairwise_spearman_mean"] = float(np.nanmean(pair)) if pair else np.nan
    sb = []
    for _ in range(500):
        perm = rng.permutation(R)
        r = safe_spearman(cells[perm[:R // 2]].mean(0), cells[perm[R // 2:]].mean(0))
        sb.append(2 * r / (1 + r) if r == r else np.nan)
    out["interrun_splithalf_SB"] = float(np.nanmean(sb))

    # image-ordering stability (PA / NA subscale per image)
    for name, idx in (("PA", L.PA_IDX), ("NA", L.NA_IDX)):
        sub = runs[:, :, idx].mean(2)                      # (runs, images)
        taus = []
        for a, b in itertools.combinations(range(R), 2):
            if np.ptp(sub[a]) and np.ptp(sub[b]):
                taus.append(stats.kendalltau(sub[a], sub[b])[0])
        out[f"image_order_tau_{name}_mean"] = float(np.mean(taus)) if taus else np.nan
        out[f"image_order_tau_{name}_min"] = float(np.min(taus)) if taus else np.nan

    out.update(variance_components(runs) if R > 1 else {})
    return out


def convergence(runs: np.ndarray, H: np.ndarray, rng) -> pd.DataFrame:
    R = runs.shape[0]
    rows = []
    for k in range(1, R + 1):
        vals = []
        for _ in range(1 if k == R else N_SUBSETS):
            sel = rng.choice(R, k, replace=False)
            vals.append(L.agreement_metrics(H, runs[sel].mean(0),
                                            rsa_metrics=("correlation",)))
        df = pd.DataFrame(vals)
        rows.append({"k_runs": k, **{f"{m}_mean": df[m].mean() for m in
                                     ("ICC2_1", "spearman", "RMSE", "RSA_correlation")},
                     "ICC2_1_sd": df["ICC2_1"].std(ddof=1) if k < R else 0.0})
    return pd.DataFrame(rows)


def plot_convergence(conv: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in conv.groupby("model"):
        ax.plot(g["k_runs"], g["ICC2_1_mean"], lw=1, alpha=.7)
    ax.set_xlabel("number of runs averaged (k)")
    ax.set_ylabel("ICC(2,1) with human means")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-long", default=str(L.DATA_DIR / "runs_long.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = Path(args.out) if args.out else (
        L.ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S") / "stability")
    out.mkdir(parents=True, exist_ok=True)

    A = L.impute_cell_mean(L.human_array(L.load_human_raw()[0]))
    H = np.nanmean(A, 0)

    allruns = L.drop_aliases(pd.read_csv(args.runs_long))
    rows, conv_all, t0_rows = [], [], []
    for (model, tag), g in allruns.groupby(["model", "tag"]):
        if g["image"].nunique() < L.N_IMAGES:
            print(f"  [skip] {model}/{tag}: incomplete")
            continue
        runs = L.runs_array(g)
        if np.isnan(runs).any() or runs.shape[0] < 2:
            print(f"  [skip] {model}/{tag}: need >=2 complete runs")
            continue
        st = model_stability(runs, H, rng)
        st.update(model=model, tag=tag)
        rows.append(st)
        if tag == "main":
            c = convergence(runs, H, rng)
            c.insert(0, "model", model)
            conv_all.append(c)
    table = pd.DataFrame(rows)
    first = ["model", "tag", "n_runs"]
    table = table[first + [c for c in table.columns if c not in first]]
    table.to_csv(out / "stability_by_model.csv", index=False)

    if conv_all:
        conv = pd.concat(conv_all)
        conv.to_csv(out / "convergence_by_k.csv", index=False)
        plot_convergence(conv, out / "convergence_icc_vs_runs.png")

    # T=0 control vs main
    if "t0" in set(table["tag"]):
        keep = ["model", "mean_cell_sd", "pct_cells_sd_zero", "interrun_ICC2_1",
                "single_run_ICC2_1_mean", "single_run_ICC2_1_sd", "meanofruns_ICC2_1"]
        a = table[table["tag"] == "main"][keep].set_index("model")
        b = table[table["tag"] == "t0"][keep].set_index("model")
        a.join(b, lsuffix="_main", rsuffix="_t0", how="inner").to_csv(
            out / "temperature_control_t0_vs_main.csv")

    # ranking stability across runs: rank models by single run r, Kendall W
    main = table[table["tag"] == "main"]
    if len(main) >= 3:
        per_run = {}
        for model, g in allruns[allruns["tag"] == "main"].groupby("model"):
            if g["image"].nunique() == L.N_IMAGES:
                runs = L.runs_array(g)
                if not np.isnan(runs).any():
                    per_run[model] = [L.icc2(H.ravel(), runs[r].ravel())[0]
                                      for r in range(runs.shape[0])]
        n_r = min(len(v) for v in per_run.values())
        M = np.array([v[:n_r] for v in per_run.values()])           # (models, runs)
        ranks = (-M).argsort(0).argsort(0) + 1
        m_, n_ = ranks.shape
        s = ((ranks.sum(1) - ranks.sum(1).mean()) ** 2).sum()
        W = 12 * s / (n_ ** 2 * (m_ ** 3 - m_))
        pd.DataFrame({"model": list(per_run), "rank_mean": ranks.mean(1),
                      "rank_min": ranks.min(1), "rank_max": ranks.max(1)}) \
            .sort_values("rank_mean").to_csv(out / "model_rank_across_runs.csv", index=False)
        (out / "ranking_concordance.json").write_text(json.dumps(
            {"kendall_W_single_run_rankings": float(W), "n_models": m_, "n_runs_used": n_}, indent=2))
    print(f"Wrote stability results to {out}")


if __name__ == "__main__":
    main()
