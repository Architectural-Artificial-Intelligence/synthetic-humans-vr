"""
Data behind the main and supplementary paper figures (43 models; aliased Gemini
models are excluded, see panas_lib.EXCLUDED_MODELS).

Writes <batch>/figure_data/:
  S1_scene_descriptives.csv   human PA/NA per scene (+95% participant CI), NA rank
  S3_model_table.csv          ICC(2,1) with participant+run CI, environment CI,
                              within-item r, tier, paired margin over the baseline
  env_means.csv / env_slopes.csv   per model x scene PA/NA means, slope on human
  item_errors.csv / item_error_summary.csv   signed item errors (model - human)
  variance_shares.csv         variance decomposition across models
  noise_ceiling_participants.csv / noise_ceiling_models.csv

Definitions
  ICC            ICC(2,1), two-way random, absolute agreement, unit = 200 cells.
  environment CI percentile bootstrap over the 10 environments (resample images
                 with replacement; participants and runs fixed).
  participant+run CI   from robust_inference (resample participants and runs).
  within-item r  correlation over the 200 cells after subtracting each item's mean
                 over the 10 scenes from both human and model values (removes
                 knowing which items are high or low overall; keeps sensitivity
                 to the scenes). mean_item_r = average of per-item Pearson r.
  baseline       scene-blind predictor: each (scene, item) cell is predicted by the human
                 mean of that item over the 10 scenes (in-sample, so it is a demanding
                 baseline). It "knows the items" but does not see the images.
  paired margin  ICC(model) - ICC(baseline), baseline recomputed in every replicate,
                 joint resampling of environments, participants and runs;
                 `margin_excludes_zero` if the 95% CI excludes 0.
                 `margin_env_only` resamples environments only.
  signed error   model - human (negative = the model rates lower than humans).

Usage: python analysis/figure_stats.py --batch results/<batch> [--B 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import panas_lib as L

SEED = 2026
ITEM_BASELINE = "item_mean_baseline"   # scene-blind: every scene gets the human item mean
TIER_CUTS = (0.70, 0.80)          # low < .70 <= mid < .80 <= top
ADAPTED_ITEMS = ("hostile", "jittery", "distressed")


def tier(icc: float) -> str:
    return "top" if icc >= TIER_CUTS[1] else "mid" if icc >= TIER_CUTS[0] else "low"


def icc21_batch(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """ICC(2,1) for two raters over the last axis; leading axes are replicates.
    h, g: (..., n)."""
    n = h.shape[-1]
    k = 2
    m_i = (h + g) / 2
    gm = m_i.mean(-1, keepdims=True)
    ch, cg = h.mean(-1, keepdims=True), g.mean(-1, keepdims=True)
    msr = k * ((m_i - gm) ** 2).sum(-1) / (n - 1)
    msc = n * (((ch - gm) ** 2) + ((cg - gm) ** 2)).squeeze(-1) / (k - 1)
    sse = (((h - m_i - ch + gm) ** 2) + ((g - m_i - cg + gm) ** 2)).sum(-1)
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def within_item_r(H: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    """Pooled item-centred correlation and mean per-item Pearson r."""
    Hc, Gc = H - H.mean(0), G - G.mean(0)
    pooled = float(np.corrcoef(Hc.ravel(), Gc.ravel())[0, 1])
    rs = [np.corrcoef(H[:, j], G[:, j])[0, 1] for j in range(H.shape[1])
          if H[:, j].std() > 0 and G[:, j].std() > 0]
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999)).mean()
    return pooled, float(np.tanh(z))


def load_inputs():
    A = L.impute_cell_mean(L.human_array(L.load_human_raw()[0]))      # (P, 10, 20)
    rl = L.drop_excluded(L.load_runs_long())
    runs = {m: L.runs_array(g) for m, g in rl.groupby("model")}
    return A, runs


# ── tables ─────────────────────────────────────────────────────────────────
def scene_descriptives(A):
    P = A.shape[0]
    rows = []
    for s in range(L.N_IMAGES):
        row = {"scene": s + 1}
        for name, idx in (("PA", L.PA_IDX), ("NA", L.NA_IDX)):
            per = A[:, s, idx].mean(1)
            se = per.std(ddof=1) / np.sqrt(P)
            t = stats.t.ppf(0.975, P - 1)
            row.update({f"human_{name}": per.mean(), f"human_{name}_lo": per.mean() - t * se,
                        f"human_{name}_hi": per.mean() + t * se})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["NA_rank_low_to_high"] = df["human_NA"].rank(method="first").astype(int)
    return df


def model_table(A, runs, comp, B, rng):
    P = A.shape[0]
    H0 = A.mean(0)
    # shared resamples so that margins are paired across models
    pidx = rng.integers(0, P, (B, P))
    eidx = rng.integers(0, L.N_IMAGES, (B, L.N_IMAGES))
    H_b = np.stack([A[i].mean(0) for i in pidx])                       # (B,10,20)
    rows, icc_all, icc_env = [], {}, {}
    # scene-blind item-mean baseline, recomputed inside every replicate
    Hbe_all = np.take_along_axis(H_b, eidx[:, :, None], 1)             # participants + environments
    base_all = np.broadcast_to(Hbe_all.mean(1, keepdims=True), Hbe_all.shape)
    He_only = H0[eidx]
    base_env = np.broadcast_to(He_only.mean(1, keepdims=True), He_only.shape)
    base_point = np.tile(H0.mean(0), (L.N_IMAGES, 1))
    b_all = icc21_batch(Hbe_all.reshape(B, -1), base_all.reshape(B, -1))
    b_env = icc21_batch(He_only.reshape(B, -1), base_env.reshape(B, -1))
    baseline = {"model": ITEM_BASELINE,
                "ICC2_1": float(icc21_batch(H0.ravel()[None], base_point.ravel()[None])[0]),
                "env_lo": np.percentile(b_env, 2.5), "env_hi": np.percentile(b_env, 97.5),
                "part_run_lo": np.percentile(b_all, 2.5), "part_run_hi": np.percentile(b_all, 97.5)}
    for m, R in runs.items():
        G0 = R.mean(0)
        icc = float(icc21_batch(H0.ravel()[None], G0.ravel()[None])[0])
        # environment-only bootstrap
        He, Ge = H0[eidx], G0[eidx]                                    # (B,10,20)
        e = icc21_batch(He.reshape(B, -1), Ge.reshape(B, -1))
        ridx = rng.integers(0, R.shape[0], (B, R.shape[0]))
        Gb = np.stack([R[ridx[b]].mean(0) for b in range(B)])          # (B,10,20)
        Hbe = np.take_along_axis(H_b, eidx[:, :, None], 1)
        Gbe = np.take_along_axis(Gb, eidx[:, :, None], 1)
        a = icc21_batch(Hbe.reshape(B, -1), Gbe.reshape(B, -1))
        icc_env[m], icc_all[m] = e, a
        wi, wi_mean = within_item_r(H0, G0)
        c = comp.set_index("model").loc[m]
        rows.append({
            "model": m, "ICC2_1": icc,
            "part_run_lo": c["ICC2_1_both_lo"], "part_run_hi": c["ICC2_1_both_hi"],
            "env_lo": np.percentile(e, 2.5), "env_hi": np.percentile(e, 97.5),
            "within_item_r": wi, "mean_item_r": wi_mean, "tier": tier(icc),
            "spearman": float(stats.spearmanr(H0.ravel(), G0.ravel())[0]),
            "bias_human_minus_model": float((H0 - G0).mean()),
        })
    df = pd.DataFrame(rows)
    for kind, store, base in (("margin", icc_all, b_all), ("margin_env_only", icc_env, b_env)):
        lo, hi = [], []
        for m in df["model"]:
            d = store[m] - base
            lo.append(np.percentile(d, 2.5)); hi.append(np.percentile(d, 97.5))
        df[kind] = df["ICC2_1"] - baseline["ICC2_1"]
        df[f"{kind}_lo"], df[f"{kind}_hi"] = lo, hi
        df[f"{kind}_excludes_zero"] = [(l > 0) or (h < 0) for l, h in zip(lo, hi)]
    df.attrs["baseline"] = baseline
    df["env_to_part_run_width_ratio"] = (df["env_hi"] - df["env_lo"]) / (df["part_run_hi"] - df["part_run_lo"])
    return df.sort_values("ICC2_1", ascending=False).reset_index(drop=True)


def env_means(A, runs):
    H = A.mean(0)
    rows = [{"model": "human", "scene": s + 1, "PA": H[s, L.PA_IDX].mean(), "NA": H[s, L.NA_IDX].mean()}
            for s in range(L.N_IMAGES)]
    for m, R in runs.items():
        G = R.mean(0)
        rows += [{"model": m, "scene": s + 1, "PA": G[s, L.PA_IDX].mean(), "NA": G[s, L.NA_IDX].mean()}
                 for s in range(L.N_IMAGES)]
    df = pd.DataFrame(rows)
    hum = df[df["model"] == "human"].set_index("scene")
    sl = []
    for m, g in df[df["model"] != "human"].groupby("model"):
        g = g.set_index("scene")
        rec = {"model": m}
        for sub in ("PA", "NA"):
            lr = stats.linregress(hum[sub], g[sub])
            rec[f"{sub}_slope"], rec[f"{sub}_intercept"] = lr.slope, lr.intercept
        sl.append(rec)
    return df, pd.DataFrame(sl)


def item_errors(A, runs):
    H = A.mean(0)
    rows = []
    for m, R in runs.items():
        err = (R.mean(0) - H).mean(0)                                  # model - human, mean over scenes
        rows += [{"model": m, "item": it, "signed_error": float(err[j])}
                 for j, it in enumerate(L.PANAS_ITEMS)]
    df = pd.DataFrame(rows)
    summ = df.groupby("item")["signed_error"].agg(
        median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
        min="min", max="max", share_models_below_zero=lambda x: (x < 0).mean()).reset_index()
    summ["subscale"] = summ["item"].map(lambda i: "PA" if i in L.PA_ITEMS else "NA")
    summ["adapted"] = summ["item"].isin(ADAPTED_ITEMS)
    return df, summ


def noise_ceiling(A, runs):
    P = A.shape[0]
    cells = A.reshape(P, -1)
    rows = []
    for p in range(P):
        others = np.delete(cells, p, 0).mean(0)
        rows.append({"participant": p, "r_with_others_mean": float(np.corrcoef(cells[p], others)[0, 1])})
    parts = pd.DataFrame(rows)
    mrows = []
    for m, R in runs.items():
        g = R.mean(0).ravel()
        r = np.array([np.corrcoef(cells[p], g)[0, 1] for p in range(P)])
        mrows.append({"model": m, "mean_r_with_individuals": r.mean(), "median_r_with_individuals": np.median(r),
                      "max_r_with_individuals": r.max()})
    return parts, pd.DataFrame(mrows).sort_values("mean_r_with_individuals", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    batch = Path(args.batch)
    out = batch / "figure_data"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    A, runs = load_inputs()
    comp = L.drop_excluded(pd.read_csv(batch / "summary" / "model_comparison.csv"))
    runs = {m: R for m, R in runs.items() if m in set(comp["model"])}
    print(f"{len(runs)} models; B = {args.B}")

    scene_descriptives(A).to_csv(out / "S1_scene_descriptives.csv", index=False)
    s3 = model_table(A, runs, comp, args.B, rng)
    s3.to_csv(out / "S3_model_table.csv", index=False)
    pd.DataFrame([s3.attrs["baseline"]]).to_csv(out / "baseline.csv", index=False)
    em, sl = env_means(A, runs)
    em.to_csv(out / "env_means.csv", index=False); sl.to_csv(out / "env_slopes.csv", index=False)
    ie, isum = item_errors(A, runs)
    ie.to_csv(out / "item_errors.csv", index=False); isum.to_csv(out / "item_error_summary.csv", index=False)
    stab = L.drop_excluded(pd.read_csv(batch / "stability" / "stability_by_model.csv"))
    stab = stab[stab["tag"] == "main"]
    vs = stab[["model", "var_image", "var_item", "var_image_x_item", "var_run_residual"]]
    pd.concat([vs, pd.DataFrame([{"model": "MEAN", **vs.drop(columns="model").mean().to_dict()},
                                 {"model": "MEDIAN", **vs.drop(columns="model").median().to_dict()}])]) \
        .to_csv(out / "variance_shares.csv", index=False)
    parts, nm = noise_ceiling(A, runs)
    parts.to_csv(out / "noise_ceiling_participants.csv", index=False)
    nm.to_csv(out / "noise_ceiling_models.csv", index=False)

    print("tiers:", s3["tier"].value_counts().to_dict())
    bl = s3.attrs["baseline"]
    print(f"item-mean baseline ICC {bl['ICC2_1']:.3f} (env CI {bl['env_lo']:.2f}-{bl['env_hi']:.2f})")
    print("paired margin over the item-mean baseline excludes 0:",
          int(s3["margin_excludes_zero"].sum()), "models (all sources);",
          int(s3["margin_env_only_excludes_zero"].sum()), "(environments only)")
    print("median env/part+run CI width ratio:", round(float(s3["env_to_part_run_width_ratio"].median()), 2))
    nas = sl["NA_slope"]
    print(f"NA slope: argon {sl.set_index('model').loc['gemini-4-argon','NA_slope']:.2f}, "
          f"median of models {nas.median():.2f}, min {nas.min():.2f}")
    print("noise ceiling median r:", round(float(parts['r_with_others_mean'].median()), 3),
          "| best model mean r:", round(float(nm['mean_r_with_individuals'].iloc[0]), 3), nm['model'].iloc[0])


if __name__ == "__main__":
    main()
