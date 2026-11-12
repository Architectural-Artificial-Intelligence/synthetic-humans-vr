"""
Female vs male differences in human PANAS ratings of the ten VR scenes, and
whether VLM agreement depends on the gender of the human reference group.

Design: 84 participants (55 F, 29 M) x 10 images x 20 PANAS items, fully
crossed, no missing data. Gender is between-subjects; image and item are
within-subjects, so every test either works on one score per participant or
models the participant as a random effect / cluster.

Analyses (primary = 1, 3, 4, 5; supplementary = 2, 6, 7, 8)
  0  Confounds: age, education, architecture background by gender
  1  Person-level PA / NA means: Welch t, Mann-Whitney, Hedges g (bootstrap CI),
     permutation p, Holm over PA/NA; sensitivity (minimum detectable d)
  1b Mixed models: scale level (participant random PA/NA intercepts, image
     fixed) and item level (cell fixed effects, participant + participant x image
     random), +age, +architecture; ordinal GEE (cluster = participant)
  2  Gender x image interaction (likelihood-ratio test) + per-image FDR
  3  Equivalence (TOST, d = +/-0.5 and +/-0.3) and JZS Bayes factors
  4  Profile similarity of the female and male 10 x 20 mean profiles vs a
     label-permutation null
  5  VLM agreement with female vs male means, equal-n participant bootstrap
  6  Dispersion: Brown-Forsythe on person means; within-group inter-rater r
  7  Item-level DIF screen (gender | rest score), cluster-robust, FDR
  8  Architecture-only sensitivity

Usage:  python analysis/gender_analysis.py [--B 2000] [--perm 5000]
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

import panas_lib as L

SEED = 2026
SCALES = {"PA": L.PA_IDX, "NA": L.NA_IDX}
ARCH = {"architecture"}
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass


# ── Data ───────────────────────────────────────────────────────────────────
def load():
    df, _ = L.load_human_raw()
    arr = L.human_array(df)                                    # (84,10,20)
    assert not np.isnan(arr).any()
    demo = pd.DataFrame({
        "pid": np.arange(len(df)),
        "female": (df.iloc[:, 3] == "female").astype(int).values,
        "age": (2025 - df.iloc[:, 2]).values.astype(float),
        "student": (df.iloc[:, 4].astype(str).str.strip() == "student").astype(int).values,
        "arch": df.iloc[:, 5].astype(str).str.strip().isin(ARCH).astype(int).values,
    })
    demo["age_z"] = (demo["age"] - demo["age"].mean()) / demo["age"].std()
    return arr, demo


def person_scores(arr):
    return pd.DataFrame({s: arr[:, :, idx].mean(axis=(1, 2)) for s, idx in SCALES.items()})


def long_scale(arr, demo):
    rows = []
    for s, idx in SCALES.items():
        sc = arr[:, :, idx].mean(2)                            # (P,10)
        for p in range(arr.shape[0]):
            for i in range(L.N_IMAGES):
                rows.append((p, i + 1, s, sc[p, i]))
    d = pd.DataFrame(rows, columns=["pid", "image", "affect", "score"])
    return d.merge(demo, on="pid")


def long_item(arr, demo, scale):
    idx = SCALES[scale]
    P = arr.shape[0]
    p, i, k = np.meshgrid(np.arange(P), np.arange(L.N_IMAGES), idx, indexing="ij")
    d = pd.DataFrame({"pid": p.ravel(), "image": i.ravel() + 1, "item": k.ravel(),
                      "rating": arr[:, :, idx].ravel()})
    d["cell"] = d["image"].astype(str) + "_" + d["item"].astype(str)
    return d.merge(demo, on="pid")


# ── Helpers ────────────────────────────────────────────────────────────────
def hedges_g(f, m):
    nf, nm = len(f), len(m)
    sp = np.sqrt(((nf - 1) * f.var(ddof=1) + (nm - 1) * m.var(ddof=1)) / (nf + nm - 2))
    d = (f.mean() - m.mean()) / sp
    return d * (1 - 3 / (4 * (nf + nm) - 9)), sp


def perm_p(f, m, rng, n):
    x = np.concatenate([f, m]); nf = len(f)
    obs = abs(f.mean() - m.mean())
    cnt = 0
    for _ in range(n):
        rng.shuffle(x)
        cnt += abs(x[:nf].mean() - x[nf:].mean()) >= obs
    return (cnt + 1) / (n + 1)


def tost(f, m, bound_raw):
    """Welch TOST: H0 |diff| >= bound. Returns p (max of the two one-sided)."""
    nf, nm = len(f), len(m)
    vf, vm = f.var(ddof=1) / nf, m.var(ddof=1) / nm
    se = np.sqrt(vf + vm)
    df = (vf + vm) ** 2 / (vf ** 2 / (nf - 1) + vm ** 2 / (nm - 1))
    diff = f.mean() - m.mean()
    p_lo = 1 - stats.t.cdf((diff + bound_raw) / se, df)
    p_hi = stats.t.cdf((diff - bound_raw) / se, df)
    ci90 = diff + np.array([-1, 1]) * stats.t.ppf(0.95, df) * se
    return max(p_lo, p_hi), ci90


# ── 0 Confounds ────────────────────────────────────────────────────────────
def confounds(demo):
    f, m = demo[demo.female == 1], demo[demo.female == 0]
    out = []
    t = stats.ttest_ind(f.age, m.age, equal_var=False)
    u = stats.mannwhitneyu(f.age, m.age)
    out.append({"variable": "age (years)", "female": f"{f.age.mean():.1f} ({f.age.std():.1f}), median {f.age.median():.0f}",
                "male": f"{m.age.mean():.1f} ({m.age.std():.1f}), median {m.age.median():.0f}",
                "test": f"Welch t = {t.statistic:.2f}, p = {t.pvalue:.3f}; Mann-Whitney p = {u.pvalue:.3f}"})
    for v, lab in (("arch", "architecture occupation"), ("student", "current student")):
        tab = pd.crosstab(demo.female, demo[v]).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
        orr, p = stats.fisher_exact(tab.values)
        out.append({"variable": lab,
                    "female": f"{tab.loc[1, 1]}/{tab.loc[1].sum()} ({tab.loc[1, 1] / tab.loc[1].sum():.0%})",
                    "male": f"{tab.loc[0, 1]}/{tab.loc[0].sum()} ({tab.loc[0, 1] / tab.loc[0].sum():.0%})",
                    "test": f"Fisher exact OR = {orr:.2f}, p = {p:.3f}"})
    return pd.DataFrame(out)


# ── 1 + 3 Person-level tests, equivalence, Bayes ───────────────────────────
def person_level(ps, demo, rng, B, n_perm, label="all"):
    fem = demo.female.values == 1
    rows = []
    for s in SCALES:
        f, m = ps[s].values[fem], ps[s].values[~fem]
        g, sp = hedges_g(f, m)
        boot = []
        for _ in range(B):
            boot.append(hedges_g(rng.choice(f, len(f)), rng.choice(m, len(m)))[0])
        t = stats.ttest_ind(f, m, equal_var=False)
        u = stats.mannwhitneyu(f, m)
        bf = float(pg.bayesfactor_ttest(stats.ttest_ind(f, m).statistic, len(f), len(m), r=0.707))
        r = {"subset": label, "scale": s, "n_F": len(f), "n_M": len(m),
             "mean_F": f.mean(), "sd_F": f.std(ddof=1), "mean_M": m.mean(), "sd_M": m.std(ddof=1),
             "diff_F_minus_M": f.mean() - m.mean(),
             "welch_t": t.statistic, "welch_df": t.df, "welch_p": t.pvalue,
             "mwu_U": u.statistic, "mwu_p": u.pvalue,
             "rank_biserial": 1 - 2 * u.statistic / (len(f) * len(m)),
             "perm_p": perm_p(f.copy(), m.copy(), rng, n_perm),
             "hedges_g": g, "g_lo": np.percentile(boot, 2.5), "g_hi": np.percentile(boot, 97.5),
             "BF10": bf, "BF01": 1 / bf}
        for d in (0.5, 0.3):
            p, ci = tost(f, m, d * sp)
            r[f"tost_p_d{d}"] = p
            r[f"tost_bound_raw_d{d}"] = d * sp
        r["diff_ci90_lo"], r["diff_ci90_hi"] = ci
        rows.append(r)
    out = pd.DataFrame(rows)
    out["welch_p_holm"] = multipletests(out.welch_p, method="holm")[1]
    return out


def sensitivity(n_f, n_m):
    pw = TTestIndPower()
    return {f"min_d_power{int(p * 100)}": pw.solve_power(nobs1=n_f, ratio=n_m / n_f, alpha=0.05, power=p)
            for p in (0.8, 0.9)}


# ── 1b Mixed models ────────────────────────────────────────────────────────
def mixed_scale(ls, extra=""):
    """score ~ affect-specific intercepts, gender, image (+covariates);
    participant random PA and NA intercepts (correlated)."""
    f = f"score ~ 0 + C(affect) + C(affect):female + C(affect):C(image) + C(affect):age_z{extra}"
    m = smf.mixedlm(f, ls, groups="pid", re_formula="0 + C(affect)").fit(reml=True, method="lbfgs")
    out = []
    for s in SCALES:
        k = f"C(affect)[{s}]:female"
        ci = m.conf_int().loc[k]
        out.append({"scale": s, "beta_female": m.params[k], "se": m.bse[k],
                    "ci_lo": ci[0], "ci_hi": ci[1], "p": m.pvalues[k]})
    return pd.DataFrame(out), m


def mixed_item(arr, demo):
    """Item-level: rating ~ female + age + cell FE; random participant intercept
    and participant x image variance component (images crossed via cell FE)."""
    out = []
    for s in SCALES:
        d = long_item(arr, demo, s)
        # re_formula="1" is required: with vc_formula set, statsmodels otherwise
        # drops the participant intercept. Default optimiser (lbfgs alone
        # converged to a degenerate variance component).
        m = smf.mixedlm("rating ~ female + age_z + C(cell)", d, groups="pid", re_formula="1",
                        vc_formula={"pimg": "0 + C(image)"}).fit(reml=True)
        ci = m.conf_int().loc["female"]
        out.append({"scale": s, "beta_female": m.params["female"], "se": m.bse["female"],
                    "ci_lo": ci[0], "ci_hi": ci[1], "p": m.pvalues["female"]})
    return pd.DataFrame(out)


def ordinal_gee(arr, demo):
    """Proportional-odds GEE, independence working correlation, robust SE
    clustered by participant. OR > 1 = women rate higher."""
    out = []
    for s in SCALES:
        d = long_item(arr, demo, s).sort_values("pid")
        d["rating"] = d["rating"].astype(int)
        m = smf.ordinal_gee("rating ~ 0 + female + age_z + C(image) + C(item)", "pid", d,
                            cov_struct=sm.cov_struct.Independence()).fit()
        b, se = m.params["female"], m.bse["female"]
        out.append({"scale": s, "logOR_female": b, "se": se, "OR": np.exp(b),
                    "OR_lo": np.exp(b - 1.96 * se), "OR_hi": np.exp(b + 1.96 * se),
                    "p": m.pvalues["female"]})
    return pd.DataFrame(out)


# ── 2 Gender x image ───────────────────────────────────────────────────────
def gender_by_image(ls, arr, demo):
    """Balanced, complete design -> exact split-plot (mixed) ANOVA:
    gender between, image within, Greenhouse-Geisser corrected."""
    rows = []
    for s in SCALES:
        d = ls[ls.affect == s]
        a = pg.mixed_anova(d, dv="score", within="image", between="female", subject="pid",
                           correction=True)
        eps = float(a.loc[a.Source == "image", "eps"].iloc[0])
        for _, r in a.iterrows():
            p_gg = r.get("p_GG_corr", np.nan)
            if r["Source"] == "Interaction":       # GG applies to the within part too
                p_gg = stats.f.sf(r["F"], r["DF1"] * eps, r["DF2"] * eps)
            rows.append({"scale": s, "effect": {"female": "gender", "image": "image",
                                                "Interaction": "gender x image"}[r["Source"]],
                         "F": r["F"], "df1": r["DF1"], "df2": r["DF2"], "p_unc": r["p_unc"],
                         "p_GG": p_gg, "eps": eps if r["Source"] != "female" else np.nan,
                         "np2": r["np2"]})
    lrt = pd.DataFrame(rows)
    fem = demo.female.values == 1
    per = []
    for s, idx in SCALES.items():
        sc = arr[:, :, idx].mean(2)
        for i in range(L.N_IMAGES):
            f, m = sc[fem, i], sc[~fem, i]
            t = stats.ttest_ind(f, m, equal_var=False)
            per.append({"scale": s, "image": i + 1, "mean_F": f.mean(), "mean_M": m.mean(),
                        "diff": f.mean() - m.mean(), "hedges_g": hedges_g(f, m)[0], "p": t.pvalue})
    per = pd.DataFrame(per)
    per["p_fdr"] = per.groupby("scale")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    return lrt, per


# ── 4 Profile similarity ───────────────────────────────────────────────────
def profile_stats(F, M):
    a = L.agreement_metrics(F, M, rsa_metrics=("correlation",))
    return {k: a[k] for k in ("ICC2_1", "pearson", "spearman", "r_within_item",
                              "RSA_correlation", "MAE", "PA_bias", "NA_bias")}


def profile_similarity(arr, demo, rng, n_perm):
    fem = demo.female.values == 1
    obs = profile_stats(arr[fem].mean(0), arr[~fem].mean(0))
    null = []
    lab = fem.copy()
    for _ in range(n_perm):
        rng.shuffle(lab)
        null.append(profile_stats(arr[lab].mean(0), arr[~lab].mean(0)))
    null = pd.DataFrame(null)
    rows = []
    for k, v in obs.items():
        nv = null[k].values
        if k in ("MAE",):                     # larger = less similar
            p = (np.sum(nv >= v) + 1) / (len(nv) + 1)
        elif k.endswith("_bias"):             # two-sided
            p = (np.sum(np.abs(nv) >= abs(v)) + 1) / (len(nv) + 1)
        else:                                 # smaller = less similar
            p = (np.sum(nv <= v) + 1) / (len(nv) + 1)
        rows.append({"metric": k, "observed_F_vs_M": v, "null_mean": nv.mean(),
                     "null_2.5": np.percentile(nv, 2.5), "null_97.5": np.percentile(nv, 97.5),
                     "perm_p": p})
    # cell-wise screen
    cells = []
    for i in range(L.N_IMAGES):
        for k in range(20):
            f, m = arr[fem, i, k], arr[~fem, i, k]
            cells.append(stats.ttest_ind(f, m, equal_var=False).pvalue)
    cells = np.nan_to_num(np.array(cells), nan=1.0)
    cell_info = {"cells_p05_uncorrected": int((cells < 0.05).sum()),
                 "cells_expected_by_chance": 0.05 * 200,
                 "cells_sig_fdr": int(multipletests(cells, method="fdr_bh")[0].sum())}
    return pd.DataFrame(rows), cell_info


# ── 5 VLM agreement by gender ──────────────────────────────────────────────
MODEL_METRICS = ["ICC2_1", "pearson", "r_within_item", "MAE", "PA_bias", "NA_bias"]


def load_model_means():
    df = L.drop_excluded(L.load_runs_long(tag="main"))
    out = {}
    for model, g in df.groupby("model"):
        a = L.runs_array(g)
        if np.isnan(a).any():
            continue
        out[model] = a.mean(0)
    return out


def _metrics(H, G):
    a = L.agreement_metrics(H, G, rsa_metrics=())
    return [a[k] for k in MODEL_METRICS]


def vlm_by_gender(arr, demo, models, rng, B):
    fem = demo.female.values == 1
    Af, Am = arr[fem], arr[~fem]
    n = len(Am)                                        # equal n = 29
    names = sorted(models)
    G = [models[k] for k in names]
    obs_full = {k: (_metrics(Af.mean(0), g), _metrics(Am.mean(0), g)) for k, g in zip(names, G)}
    # equal-n point estimate: mean over subsamples of 29 women (no replacement)
    sub = np.zeros((B, len(names), len(MODEL_METRICS)))
    # participant bootstrap of the difference, equal n, both groups resampled
    boot = np.zeros((B, len(names), len(MODEL_METRICS)))
    for b in range(B):
        Fs = Af[rng.choice(len(Af), n, replace=False)].mean(0)
        Fb = Af[rng.choice(len(Af), n, replace=True)].mean(0)
        Mb = Am[rng.choice(len(Am), n, replace=True)].mean(0)
        Mm = Am.mean(0)
        for j, g in enumerate(G):
            sub[b, j] = np.subtract(_metrics(Fs, g), _metrics(Mm, g))
            boot[b, j] = np.subtract(_metrics(Fb, g), _metrics(Mb, g))
    rows = []
    for j, k in enumerate(names):
        for q, met in enumerate(MODEL_METRICS):
            bd = boot[:, j, q]
            p = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
            rows.append({"model": k, "metric": met,
                         "vs_female_all55": obs_full[k][0][q], "vs_male": obs_full[k][1][q],
                         "diff_F_minus_M_equal_n": sub[:, j, q].mean(),
                         "boot_lo": np.percentile(bd, 2.5), "boot_hi": np.percentile(bd, 97.5),
                         "boot_p": max(p, 1 / B)})
    per_model = pd.DataFrame(rows)
    per_model["p_holm"] = per_model.groupby("metric")["boot_p"].transform(
        lambda p: multipletests(p, method="holm")[1])
    pooled = []
    for q, met in enumerate(MODEL_METRICS):
        bd = boot[:, :, q].mean(1)
        p = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
        favours = per_model[(per_model.metric == met)]["diff_F_minus_M_equal_n"]
        pooled.append({"metric": met, "mean_diff_F_minus_M": sub[:, :, q].mean(),
                       "boot_lo": np.percentile(bd, 2.5), "boot_hi": np.percentile(bd, 97.5),
                       "boot_p": max(p, 1 / B),
                       "n_models_diff_pos": int((favours > 0).sum()), "n_models": len(names)})
    return per_model, pd.DataFrame(pooled)


# ── 6 Dispersion ───────────────────────────────────────────────────────────
def dispersion(arr, ps, demo, rng, n_perm):
    fem = demo.female.values == 1
    rows = []
    for s in SCALES:
        f, m = ps[s].values[fem], ps[s].values[~fem]
        bf = stats.levene(f, m, center="median")
        rows.append({"measure": f"{s} person-mean SD (Brown-Forsythe)", "female": f.std(ddof=1),
                     "male": m.std(ddof=1), "stat": bf.statistic, "p": bf.pvalue})
    X = arr.reshape(len(arr), -1)

    def mean_r(rows_):
        c = np.corrcoef(X[rows_])
        return c[np.triu_indices_from(c, 1)].mean()

    obs = mean_r(fem) - mean_r(~fem)
    lab = fem.copy(); null = []
    for _ in range(min(n_perm, 2000)):
        rng.shuffle(lab)
        null.append(mean_r(lab) - mean_r(~lab))
    null = np.array(null)
    rows.append({"measure": "mean pairwise inter-rater r over 200 cells", "female": mean_r(fem),
                 "male": mean_r(~fem), "stat": obs,
                 "p": (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)})
    return pd.DataFrame(rows)


# ── 7 DIF screen ───────────────────────────────────────────────────────────
def dif(arr, demo):
    rows = []
    for s, idx in SCALES.items():
        for k in idx:
            rest = arr[:, :, [j for j in idx if j != k]].mean(2)
            d = pd.DataFrame({"y": arr[:, :, k].ravel(), "rest": rest.ravel(),
                              "pid": np.repeat(np.arange(len(arr)), L.N_IMAGES),
                              "image": np.tile(np.arange(L.N_IMAGES), len(arr))}).merge(demo, on="pid")
            m = smf.ols("y ~ female + rest + C(image)", d).fit(
                cov_type="cluster", cov_kwds={"groups": d["pid"]})
            rows.append({"scale": s, "item": L.PANAS_ITEMS[k], "beta_female": m.params["female"],
                         "se": m.bse["female"], "p": m.pvalues["female"]})
    out = pd.DataFrame(rows)
    out["p_fdr"] = multipletests(out.p, method="fdr_bh")[1]
    return out


# ── Report ─────────────────────────────────────────────────────────────────
def fmt_p(p):
    return "< .001" if p < 0.001 else f"{p:.3f}"


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = Path(args.out) if args.out else L.ROOT / "results" / f"gender_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    arr, demo = load()
    ps = person_scores(arr)
    ls = long_scale(arr, demo)
    nF, nM = int(demo.female.sum()), int((1 - demo.female).sum())
    print(f"N = {len(demo)} ({nF} F, {nM} M)")

    res = {}
    print("0 confounds"); res["conf"] = confounds(demo)
    print("1/3 person level"); res["person"] = person_level(ps, demo, rng, args.B, args.perm)
    res["sens"] = sensitivity(nF, nM)
    print("1b mixed models")
    res["mixed"], _ = mixed_scale(ls)
    res["mixed_arch"], _ = mixed_scale(ls, extra=" + C(affect):arch")
    res["mixed_item"] = mixed_item(arr, demo)
    res["gee"] = ordinal_gee(arr, demo)
    print("2 gender x image"); res["lrt"], res["per_image"] = gender_by_image(ls, arr, demo)
    print("4 profile"); res["profile"], res["cells"] = profile_similarity(arr, demo, rng, args.perm)
    print("5 VLM by gender")
    models = load_model_means()
    res["vlm"], res["vlm_pooled"] = vlm_by_gender(arr, demo, models, rng, args.B)
    print("6 dispersion"); res["disp"] = dispersion(arr, ps, demo, rng, args.perm)
    print("7 DIF"); res["dif"] = dif(arr, demo)
    print("8 architecture only")
    a_idx = demo.arch.values == 1
    demo_a = demo[a_idx].reset_index(drop=True); demo_a["pid"] = np.arange(len(demo_a))
    res["person_arch"] = person_level(ps[a_idx].reset_index(drop=True), demo_a, rng, args.B, args.perm,
                                      label="architecture only")

    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"sensitivity": res["sens"], "cells": res["cells"], "B": args.B, "perm": args.perm,
         "seed": SEED, "n_models": len(models)}, indent=2))

    P = res["person"].set_index("scale")
    cols_p = ["scale", "mean_F", "sd_F", "mean_M", "sd_M", "diff_F_minus_M", "welch_t", "welch_p",
              "welch_p_holm", "mwu_p", "perm_p", "hedges_g", "g_lo", "g_hi"]
    cols_e = ["scale", "diff_ci90_lo", "diff_ci90_hi", "tost_bound_raw_d0.5", "tost_p_d0.5",
              "tost_bound_raw_d0.3", "tost_p_d0.3", "BF10", "BF01"]
    vp = res["vlm_pooled"]
    vm = res["vlm"]
    icc_m = vm[vm.metric == "ICC2_1"].sort_values("diff_F_minus_M_equal_n")
    lines = [
        "# Female vs male PANAS ratings of the VR scenes",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `analysis/gender_analysis.py` "
        f"(seed {SEED}, B = {args.B} bootstraps, {args.perm} permutations). "
        f"N = {len(demo)}: {nF} female, {nM} male; 10 images x 20 items, no missing ratings. "
        f"{len(models)} VLMs (main batch).",
        "",
        "Signs: differences are **female minus male**. Positive = women rate higher.",
        "",
        "## 0. Confounds",
        md(res["conf"]),
        "",
        "## 1. Primary test: person-level PA and NA means",
        "One score per participant (mean over 10 images x 10 items), so the test unit is the person.",
        "",
        md(res["person"][cols_p]),
        "",
        f"Sensitivity: with {nF} vs {nM}, a two-sided Welch-type test at alpha = .05 detects "
        f"d >= {res['sens']['min_d_power80']:.2f} with 80% power (d >= {res['sens']['min_d_power90']:.2f} at 90%).",
        "",
        "### 1b. Mixed-model and ordinal robustness checks (female coefficient)",
        "Scale level, participant random PA/NA intercepts, image fixed, + age:",
        md(res["mixed"]), "",
        "Same + architecture occupation:",
        md(res["mixed_arch"]), "",
        "Item level, 200 cell fixed effects, random participant + participant x image, + age:",
        md(res["mixed_item"]), "",
        "Ordinal (proportional-odds) GEE, clustered by participant, + age, image, item (OR > 1 = women higher):",
        md(res["gee"]), "",
        "## 2. Does the gender gap depend on the scene?",
        "Split-plot ANOVA on PA / NA image scores (gender between, image within); "
        "Greenhouse-Geisser epsilon from the image factor applied to the within effects:",
        md(res["lrt"]), "",
        "Per-image differences (Welch, BH-FDR within scale):",
        md(res["per_image"]), "",
        "## 3. Equivalence and Bayes factors",
        "TOST with smallest effect of interest d = 0.5 and d = 0.3 (converted to rating points with the pooled SD). "
        "Equivalence is supported when tost_p < .05 (i.e. the 90% CI lies inside the bounds). "
        "JZS Bayes factor, Cauchy prior r = 0.707.",
        "",
        md(res["person"][cols_e]), "",
        "## 4. Profile similarity (female vs male 10 x 20 mean profiles)",
        "Null distribution: gender labels permuted (55/29 kept). For similarity metrics p is the share "
        "of random splits that are *less* similar than the real F/M split; a small p means the genders "
        "differ more than random halves of the sample.",
        "",
        md(res["profile"]), "",
        f"Cell-wise Welch screen (200 cells): {res['cells']['cells_p05_uncorrected']} with p < .05 uncorrected "
        f"(about {res['cells']['cells_expected_by_chance']:.0f} expected by chance), "
        f"{res['cells']['cells_sig_fdr']} significant after BH-FDR.",
        "",
        "## 5. Do the VLMs agree more with women or with men?",
        f"Each model's run-mean profile vs the female and male mean profiles. Women are subsampled to n = {nM} "
        "so both reference means have equal reliability; CIs come from a participant bootstrap of both groups "
        "(n = 29 each). For ICC / r a positive difference means the model agrees more with women; "
        "for MAE a positive difference means it is *further* from women.",
        "",
        "Pooled over models:",
        md(vp), "",
        "Per model, ICC(2,1) (sorted):",
        md(icc_m[["model", "vs_female_all55", "vs_male", "diff_F_minus_M_equal_n", "boot_lo", "boot_hi",
                  "boot_p", "p_holm"]]), "",
        "All metrics per model: `vlm.csv`.",
        "",
        "## 6. Dispersion and rater consistency",
        md(res["disp"]), "",
        "## 7. Item-level DIF screen",
        "Item rating ~ female + rest-of-scale score + image, cluster-robust by participant; BH-FDR over 20 items. "
        "A non-zero female coefficient means women rate this item differently from men at the same overall "
        "PA/NA level. Formal multi-group CFA invariance is not feasible with 29 men.",
        "",
        md(res["dif"]), "",
        "## 8. Sensitivity: architecture participants only",
        md(res["person_arch"][cols_p + ["tost_p_d0.5", "BF10"]]),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
