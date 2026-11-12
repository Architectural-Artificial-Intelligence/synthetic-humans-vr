"""
Supplementary tables and figure for the exploratory gender analysis.

Usage: python analysis/gender_supplement.py --gender results/gender_<ts> --batch results/<batch>
Needs : <gender>/*.csv from analysis/gender_analysis.py, analysis/scene_info.csv, data/human.csv
Writes: <batch>/paper/tables/T11_gender_{a_person,b_environment,c_vlm_pooled}.{csv,md}
        <batch>/paper/tables/T12_gender_vlm_by_model.{csv,md}
        <batch>/paper/main_figures/FigS4_gender.{png,pdf} + captions/FigS4_gender.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import make_paper_outputs as P
import panas_lib as L

INK, MUTED, GRAY = P.INK, P.MUTED, P.GRAY
WOMEN, MEN = P.VIOLET, P.AQUA          # validated pair; men hollow (aqua < 3:1 contrast)
W = 7.2


def read(d: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(d / f"{name}.csv", keep_default_na=False, na_values=[""])   # "NA" = subscale


def nz(s: str) -> str:
    """Drop the sign of a rounded zero ("-0.00" -> "0.00")."""
    return re.sub(r"[-+](0\.0+)(?!\d*[1-9])", r"\1", s)


def fp(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def write_table(df: pd.DataFrame, path: Path, title: str, note: str):
    df.to_csv(path.with_suffix(".csv"), index=False)
    df = df.map(lambda x: nz(x) if isinstance(x, str) else x)
    path.with_suffix(".md").write_text(f"**{title}**\n\n{df.to_markdown(index=False, disable_numparse=True)}\n\n*{note}*\n")


# ── Tables ─────────────────────────────────────────────────────────────────
def tables(g: Path, out: Path, scenes: pd.DataFrame):
    per, mix = read(g, "person"), read(g, "mixed_arch").set_index("scale")
    a = []
    for _, r in per.iterrows():
        m = mix.loc[r.scale]
        a.append({
            "Subscale": r.scale,
            "Women M (SD)": f"{r.mean_F:.2f} ({r.sd_F:.2f})",
            "Men M (SD)": f"{r.mean_M:.2f} ({r.sd_M:.2f})",
            "Diff (W-M)": f"{r.diff_F_minus_M:+.2f}",
            "Hedges g [95% CI]": f"{r.hedges_g:.2f} [{r.g_lo:.2f}, {r.g_hi:.2f}]",
            "Welch t (df)": f"{r.welch_t:.2f} ({r.welch_df:.1f})",
            "p (Holm)": fp(r.welch_p_holm),
            "TOST p (d = ±0.5)": fp(r["tost_p_d0.5"]),
            "BF10": f"{r.BF10:.2f}",
            "Adjusted β [95% CI]": f"{m.beta_female:+.2f} [{m.ci_lo:.2f}, {m.ci_hi:.2f}]",
            "Adjusted p": fp(m.p),
        })
    write_table(pd.DataFrame(a), out / "T11_gender_a_person",
                "Table S11a | Women vs men: participant-level PA and NA means.",
                "N = 84 (55 women, 29 men). One score per participant (mean of 10 environments × 10 items). "
                "Hedges g CI: participant bootstrap (2,000). TOST: equivalence bounds d = ±0.5; p < .05 supports equivalence. "
                "BF10: JZS, Cauchy r = 0.707. Adjusted β: linear mixed model on environment-level scores with "
                "participant random PA/NA intercepts, environment fixed, adjusted for age and architecture background.")

    pi, lrt = read(g, "per_image"), read(g, "lrt")
    rows = []
    for img in range(1, L.N_IMAGES + 1):
        r = {"#": img, "Environment": scenes.loc[img, "short_name"]}
        for s in ("PA", "NA"):
            x = pi[(pi.scale == s) & (pi.image == img)].iloc[0]
            r[f"{s} women"] = f"{x.mean_F:.2f}"
            r[f"{s} men"] = f"{x.mean_M:.2f}"
            r[f"{s} g"] = f"{x.hedges_g:+.2f}"
            r[f"{s} p (FDR)"] = fp(x.p_fdr)
        rows.append(r)
    ix = lrt[lrt.effect == "gender x image"].set_index("scale")
    write_table(pd.DataFrame(rows), out / "T11_gender_b_environment",
                "Table S11b | Women vs men by environment.",
                "Environment-level subscale means; g = Hedges g (women − men); Welch tests with Benjamini–Hochberg "
                "FDR within subscale. Gender × environment (mixed ANOVA, Greenhouse–Geisser): "
                f"PA F({ix.loc['PA', 'df1']:.0f}, {ix.loc['PA', 'df2']:.0f}) = {ix.loc['PA', 'F']:.2f}, p = {fp(ix.loc['PA', 'p_GG'])}; "
                f"NA F({ix.loc['NA', 'df1']:.0f}, {ix.loc['NA', 'df2']:.0f}) = {ix.loc['NA', 'F']:.2f}, p = {fp(ix.loc['NA', 'p_GG'])}.")

    vp = read(g, "vlm_pooled")
    lab = {"ICC2_1": "ICC(2,1)", "pearson": "Pearson r", "r_within_item": "Within-item r", "MAE": "MAE"}
    c = [{"Index": lab[r.metric], "Diff (women − men)": f"{r.mean_diff_F_minus_M:+.3f}",
          "95% CI": f"[{r.boot_lo:+.3f}, {r.boot_hi:+.3f}]", "p": fp(r.boot_p),
          "Models with diff > 0": f"{r.n_models_diff_pos} / {r.n_models}"}
         for _, r in vp[vp.metric.isin(lab)].iterrows()]
    write_table(pd.DataFrame(c), out / "T11_gender_c_vlm_pooled",
                "Table S11c | Model agreement with women vs men (mean over 43 models).",
                "Each model's run-mean profile scored against the women's and the men's 200 cell means. Women "
                "subsampled to n = 29 to equalise reliability; CI from a participant bootstrap of both groups (2,000). "
                "For ICC and r, negative = closer to men; for MAE, positive = further from women.")

    v = read(g, "vlm")
    v = v[v.metric == "ICC2_1"].copy()
    v["vs_women_eq"] = v.vs_male + v.diff_F_minus_M_equal_n
    v = v.sort_values("diff_F_minus_M_equal_n")
    t12 = pd.DataFrame({
        "Model": v.model, "ICC vs women (n = 55)": v.vs_female_all55.round(3),
        "ICC vs women (n = 29)": v.vs_women_eq.round(3), "ICC vs men (n = 29)": v.vs_male.round(3),
        "Diff (W − M)": v.diff_F_minus_M_equal_n.round(3),
        "95% CI": [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(v.boot_lo, v.boot_hi)],
        "p": v.boot_p.map(fp), "p (Holm)": v.p_holm.map(fp)})
    write_table(t12, out / "T12_gender_vlm_by_model",
                "Table S12 | ICC(2,1) of each model with women and with men.",
                "Women (n = 29): mean over 2,000 random subsamples of 29 women. CI and p: participant bootstrap of "
                "both groups; Holm correction across 43 models.")
    return v


# ── Figure ─────────────────────────────────────────────────────────────────
def scene_ci(arr, fem, idx):
    sc = arr[:, :, idx].mean(2)
    out = {}
    for name, mask in (("w", fem), ("m", ~fem)):
        x = sc[mask]
        m, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x))
        out[name] = (m, stats.t.ppf(0.975, len(x) - 1) * se)
    return out


def figure(g: Path, out: Path, scenes: pd.DataFrame, v: pd.DataFrame):
    df, _ = L.load_human_raw()
    arr = L.human_array(df)
    fem = (df.iloc[:, 3] == "נקבה").values
    pi = read(g, "per_image")
    order = np.argsort(arr[:, :, L.NA_IDX].mean(axis=(0, 2)))       # environments by human NA, low -> high
    names = [scenes.loc[i + 1, "short_name"] for i in order]

    fig = plt.figure(figsize=(W, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.28, 1.15], wspace=0.12)
    ax_pa = fig.add_subplot(gs[0]); ax_na = fig.add_subplot(gs[1], sharey=ax_pa)
    ax_v = fig.add_subplot(gs[3])
    dy = 0.17
    for ax, s, idx, title in ((ax_pa, "PA", L.PA_IDX, "a  Positive affect"),
                              (ax_na, "NA", L.NA_IDX, "b  Negative affect")):
        c = scene_ci(arr, fem, idx)
        for y, i in enumerate(order):
            mw, ew = c["w"][0][i], c["w"][1][i]
            mm, em = c["m"][0][i], c["m"][1][i]
            ax.plot([mw - ew, mw + ew], [y + dy] * 2, color=WOMEN, lw=1.2, zorder=2)
            ax.plot([mm - em, mm + em], [y - dy] * 2, color=MEN, lw=1.2, zorder=2)
            ax.plot(mw, y + dy, "o", color=WOMEN, ms=5, mec="white", mew=.8, zorder=3)
            ax.plot(mm, y - dy, "o", mfc="white", mec=MEN, mew=1.4, ms=5, zorder=3)
            p = pi[(pi.scale == s) & (pi.image == i + 1)].p_fdr.iloc[0]
            if p < .05:
                ax.text(max(mw + ew, mm + em) + 0.06, y, "*" if p >= .01 else "**",
                        va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean rating (1–5), 95% CI")
        ax.set_xlim(0.9, 4.1)
        ax.grid(axis="y", visible=False)
    ax_pa.set_yticks(range(len(order)), names, fontsize=7.5)
    ax_pa.set_ylim(-0.6, len(order) - 0.4)
    plt.setp(ax_na.get_yticklabels(), visible=False)
    ax_na.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], marker="o", ls="-", lw=1.2, color=WOMEN, mec="white", ms=5, label="women (n = 55)"),
         plt.Line2D([], [], marker="o", ls="-", lw=1.2, color=MEN, mfc="white", mec=MEN, mew=1.4, ms=5,
                    label="men (n = 29)")]
    ax_na.legend(handles=h, loc="lower right", fontsize=7, frameon=False)

    lo, hi = 0.45, 0.95
    ax_v.plot([lo, hi], [lo, hi], color=GRAY, ls="--", lw=1, zorder=1)
    ax_v.scatter(v.vs_male, v.vs_women_eq, s=22, color=INK, alpha=.8, lw=0, zorder=3)
    ax_v.set_xlim(lo, hi); ax_v.set_ylim(lo, hi)
    ax_v.set_xlabel("ICC(2,1) with men")
    ax_v.set_ylabel("ICC(2,1) with women (n = 29)")
    ax_v.set_title("c  Model agreement", loc="left", fontsize=9, fontweight="bold")
    n_below = int((v.vs_women_eq < v.vs_male).sum())
    ax_v.text(0.93, 0.50, f"{n_below} of {len(v)} models\ncloser to men", ha="right", va="bottom",
              fontsize=7.5, color=MUTED)
    ax_v.text(0.50, 0.92, "closer to women", ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.91)
    P.save_fig(fig, out / "FigS4_gender")
    return n_below


def caption(g: Path, out: Path, n_below: int, n_models: int):
    per = read(g, "person").set_index("scale")
    vp = read(g, "vlm_pooled").set_index("metric")
    lrt = read(g, "lrt").set_index(["scale", "effect"])
    pi = read(g, "per_image")
    na = per.loc["NA"]; pa = per.loc["PA"]
    sig = pi[(pi.scale == "NA") & (pi.p_fdr < .05)]
    icc, wi = vp.loc["ICC2_1"], vp.loc["r_within_item"]
    ix = lrt.loc[("NA", "gender x image")]
    txt = (
        "Figure S4. Women and men rate positive affect alike; women report more negative affect in some environments.\n"
        "(a, b) Environment-level PA and NA means for women (filled, n = 55) and men (hollow, n = 29) with 95% CIs; "
        "environments sorted by overall human NA. Asterisks: Welch test, Benjamini–Hochberg FDR within subscale "
        "(* < .05, ** < .01). (c) ICC(2,1) of each of the "
        f"{n_models} models with the men's and with the women's 200 cell means; women subsampled to n = 29 so both "
        "references are equally reliable (mean over 2,000 subsamples). Points below the dashed identity line agree "
        "more with men.\n"
        f"Participant-level PA did not differ (g = {pa.hedges_g:.2f}, p = {fp(pa.welch_p_holm)}) and was equivalent "
        f"within d = ±0.5 (TOST p = {fp(pa['tost_p_d0.5'])}, BF01 = {pa.BF01:.1f}). Women reported higher NA "
        f"(+{na.diff_F_minus_M:.2f} points, g = {na.hedges_g:.2f} [{na.g_lo:.2f}, {na.g_hi:.2f}], Holm p = "
        f"{fp(na.welch_p_holm)}, BF10 = {na.BF10:.1f}). The NA difference varied by environment (gender × environment "
        f"F({ix.df1:.0f}, {ix.df2:.0f}) = {ix.F:.2f}, Greenhouse–Geisser p = {fp(ix.p_GG)}) and was significant only in "
        + " and ".join(f"{n} (g = {x:.2f})" for n, x in zip(sig.image.map(lambda i: SCENES.loc[i, 'short_name']), sig.hedges_g))
        + f". {n_below} of {n_models} models agreed more with men (mean ΔICC = {icc.mean_diff_F_minus_M:+.3f} "
        f"[{icc.boot_lo:+.3f}, {icc.boot_hi:+.3f}]), but within-item r did not differ "
        f"(Δ = {wi.mean_diff_F_minus_M:+.3f} [{wi.boot_lo:+.3f}, {wi.boot_hi:+.3f}]): the gap reflects the level of NA, "
        "which models under-rate and men rate lower, not the pattern across environments. Exploratory analysis; "
        "full results in Tables S11–S12.\n")
    cap = out / "captions"; cap.mkdir(exist_ok=True)
    txt = nz(txt)
    (cap / "FigS4_gender.txt").write_text(txt)
    print(txt)


SCENES: pd.DataFrame


def main():
    global SCENES
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", required=True)
    ap.add_argument("--batch", required=True)
    a = ap.parse_args()
    g, batch = Path(a.gender), Path(a.batch)
    SCENES = pd.read_csv(Path(__file__).parent / "scene_info.csv").set_index("scene")
    tdir, fdir = batch / "paper" / "tables", batch / "paper" / "main_figures"
    v = tables(g, tdir, SCENES)
    n_below = figure(g, fdir, SCENES, v)
    caption(g, fdir, n_below, len(v))


if __name__ == "__main__":
    main()
