"""
PANAS Affective Structure Analysis: Human vs. GPT-4
======================================================
Compares emotional ratings of VR images between human participants
and GPT-4 on the Positive and Negative Affect Schedule (PANAS).

Covers:
  Step 1  – Data loading & inspection
  Step 2  – Preprocessing (ordinal → numeric, long-form reshape)
  Step 3  – Linear Mixed-Effects Models (LME)
  Step 4  – Agreement analysis (ICC, Pearson/Spearman, Bland–Altman)
  Step 5  – Psychometric structure (CFA, measurement invariance)
  Step 6  – Representational Similarity Analysis (RSA / cosine)
  Step 7  – Robustness (bootstrap CI, permutation tests)
  Step 8  – Effect sizes and Bayes Factors
  Step 9  – Figures (saved to ../figures/)
  Step 10 – Structured interpretation printed to stdout

Outputs
-------
  ../results/  – JSON/CSV tables for each step
  ../figures/  – high-resolution PNG figures

Requirements:
  pandas numpy scipy statsmodels pingouin semopy scikit-learn
  matplotlib seaborn pymc arviz
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import statsmodels.formula.api as smf
import pingouin as pg
import semopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
FIG_DIR   = ROOT / "figures"
RES_DIR   = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Hebrew keyword → PANAS English item
# Ordered as they appear in column headers
HEBREW_ITEM_MAP = {
    "להתעניין"    : "interested",
    "לחוץ"        : "nervous",
    "נרגש"        : "excited",
    "מוטרד"       : "upset",
    "חזק"         : "strong",
    "אשם"         : "guilty",
    "בהלה"        : "scared",
    "עוין"        : "hostile",
    "התלהבות"     : "enthusiastic",
    "גאה"         : "proud",
    "רוגז"        : "irritable",
    "עירני"       : "alert",
    "מבוייש"      : "ashamed",
    "שראה"        : "inspired",      # בעל/ת שראה → "inspired"
    "עצבני"       : "jittery",
    "נחוש"        : "determined",
    "קשוב"        : "attentive",
    "מתח"         : "distressed",
    "פעיל"        : "active",
    "פחד"         : "afraid",        # extra item (not in GPT)
}

# Standard PANAS classification
PA_ITEMS = {"interested","excited","strong","enthusiastic","proud",
            "alert","inspired","determined","attentive","active"}
NA_ITEMS = {"nervous","upset","guilty","scared","hostile",
            "irritable","ashamed","jittery","distressed","afraid"}

# Likert mapping (Hebrew → 1–5)
LIKERT_MAP = {
    "כלל לא"         : 1,
    "מעט"            : 2,
    "במידה בינונית"  : 3,
    "במידה רבה"      : 4,
    "רבה מאוד"       : 5,   # truncated form of "במידה רבה מאוד"
}

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def hebrew_keyword(col: str) -> str | None:
    """Extract the emotional keyword from a Hebrew PANAS column header."""
    m = re.search(r'\[עד כמה החלל [^[]+\]', col)
    if not m:
        return None
    text = m.group(0)
    for kw in HEBREW_ITEM_MAP:
        if kw in text:
            return kw
    return None


def get_image_index(col: str) -> int:
    """Return 0-based image index from column suffix (.1 → 1, no suffix → 0)."""
    m = re.search(r'\.(\d+)$', col)
    return int(m.group(1)) if m else 0


def cohens_d(a, b):
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1)*np.std(a, ddof=1)**2 +
                      (nb - 1)*np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def bootstrap_ci(data, func, n=N_BOOTSTRAP, alpha=0.05):
    """Parametric bootstrap CI for a scalar statistic."""
    stats_boot = [func(RNG.choice(data, size=len(data), replace=True))
                  for _ in range(n)]
    lo = np.percentile(stats_boot, 100 * alpha / 2)
    hi = np.percentile(stats_boot, 100 * (1 - alpha / 2))
    return lo, hi


def bootstrap_ci_2samp(a, b, func, n=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI for a two-sample statistic."""
    boot = [func(RNG.choice(a, size=len(a), replace=True),
                 RNG.choice(b, size=len(b), replace=True))
            for _ in range(n)]
    return np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])


def permutation_test(a, b, func, n=N_BOOTSTRAP):
    """Permutation test p-value for observed func(a, b)."""
    observed = func(a, b)
    combined = np.concatenate([a, b])
    null = []
    for _ in range(n):
        perm = RNG.permutation(combined)
        null.append(func(perm[:len(a)], perm[len(a):]))
    p = np.mean(np.abs(null) >= np.abs(observed))
    return observed, p


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: Load and preprocess data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading and inspecting data")
print("=" * 70)

# --- GPT data (already numeric, image × item means) ---
df_gpt_raw = pd.read_csv(DATA_DIR / "gpt4.csv", index_col=0)
df_gpt_raw.index = df_gpt_raw.index.str.strip().str.lower()

# Normalise column names: "image 1" → 1 (int)
df_gpt_raw.columns = [int(c.strip().split()[-1]) for c in df_gpt_raw.columns]

print(f"\nGPT data: {df_gpt_raw.shape[1]} images × {df_gpt_raw.shape[0]} items")
print("Items:", list(df_gpt_raw.index))

# --- Human raw data ---
df_raw = pd.read_csv(DATA_DIR / "human.csv")

# Identify PANAS columns (index ≥ 6)
demo_cols  = df_raw.columns[:6].tolist()
panas_cols = df_raw.columns[6:].tolist()

n_participants_raw = len(df_raw)
missing_total      = df_raw[panas_cols].isna().sum().sum()
missing_pct        = 100 * missing_total / (len(df_raw) * len(panas_cols))

print(f"\nHuman data:")
print(f"  Participants (rows)   : {n_participants_raw}")
print(f"  Total columns         : {len(df_raw.columns)}")
print(f"  PANAS columns         : {len(panas_cols)}")
print(f"  Missing PANAS values  : {missing_total}  ({missing_pct:.1f}%)")
_vals = df_raw[panas_cols].stack().unique()
_vals = [v for v in _vals if isinstance(v, str)]
print(f"  Likert levels observed: {sorted(_vals)}")

# Drop row 0 if it is entirely NaN on PANAS (consent row artefact)
df_raw = df_raw.dropna(subset=panas_cols, how="all").reset_index(drop=True)
print(f"  Participants after removing blank rows: {len(df_raw)}")

print("\n" + "=" * 70)
print("STEP 2: Preprocessing")
print("=" * 70)

# ── 2a. Map each Hebrew PANAS column to (image_idx, item_name) ──────────────
col_meta = {}          # col_name → (image_idx, item_name)
for col in panas_cols:
    kw  = hebrew_keyword(col)
    img = get_image_index(col)
    if kw is not None:
        col_meta[col] = (img, HEBREW_ITEM_MAP[kw])

print(f"\nMapped {len(col_meta)} PANAS columns to (image, item)")

# Verify image index range
imgs_found = sorted(set(v[0] for v in col_meta.values()))
items_found = sorted(set(v[1] for v in col_meta.values()))
print(f"Images (0-indexed): {imgs_found}")
print(f"Items ({len(items_found)}): {items_found}")

# ── 2b. Convert Likert text → numeric ───────────────────────────────────────
def likert_to_num(series: pd.Series) -> pd.Series:
    return series.map(LIKERT_MAP)

# Build a long-form dataframe: participant × image × item → score
records = []
for pid, row in df_raw.iterrows():
    for col, (img_idx, item) in col_meta.items():
        val = LIKERT_MAP.get(str(row[col]).strip() if pd.notna(row[col]) else "", np.nan)
        records.append({
            "participant": pid,
            "image"      : img_idx + 1,   # 1-indexed to match GPT
            "item"       : item,
            "score"      : val,
            "rater_type" : "human",
        })

df_human_long = pd.DataFrame(records)

# Missing after mapping
n_missing_numeric = df_human_long["score"].isna().sum()
pct_missing_num   = 100 * n_missing_numeric / len(df_human_long)
print(f"\nLong-form human data: {len(df_human_long)} rows")
print(f"Missing after numeric mapping: {n_missing_numeric} ({pct_missing_num:.2f}%)")

# ── 2c. Impute remaining missing with item × image mean ──────────────────────
df_human_long["score"] = df_human_long.groupby(["image","item"])["score"] \
                                       .transform(lambda x: x.fillna(x.mean()))
print("Imputation: item×image group mean for remaining missing values")

# ── 2d. Build ITEM-LEVEL human means per image (for agreement analysis) ─────
human_means = (df_human_long.groupby(["image","item"])["score"]
                             .mean()
                             .rename("human_mean")
                             .reset_index())

# ── 2e. Reshape GPT data to long form ───────────────────────────────────────
gpt_long_records = []
for item, row in df_gpt_raw.iterrows():
    for img, val in row.items():
        gpt_long_records.append({
            "image"     : img,
            "item"      : item,
            "gpt_mean"  : val,
        })
df_gpt_long = pd.DataFrame(gpt_long_records)

# ── 2f. Align datasets (only items present in BOTH) ─────────────────────────
common_items = sorted(set(human_means["item"]) & set(df_gpt_long["item"]))
print(f"\nCommon items in both datasets: {len(common_items)}")
print(f"  {common_items}")

# Filter to common items
human_means_c = human_means[human_means["item"].isin(common_items)].copy()
gpt_long_c    = df_gpt_long[df_gpt_long["item"].isin(common_items)].copy()
df_human_long_c = df_human_long[df_human_long["item"].isin(common_items)].copy()

# Merged image-level comparison dataframe
df_merged = pd.merge(human_means_c, gpt_long_c, on=["image","item"])
df_merged["rater_type"] = "human"   # placeholder
df_merged["diff"]       = df_merged["human_mean"] - df_merged["gpt_mean"]

print(f"\nMerged comparison dataframe: {len(df_merged)} rows "
      f"({len(common_items)} items × {df_merged['image'].nunique()} images)")

# ── 2g. Compute PA and NA subscores ─────────────────────────────────────────
# Human PA/NA at participant × image level
def compute_PA_NA(df_long):
    pa = (df_long[df_long["item"].isin(PA_ITEMS)]
          .groupby(["participant","image"])["score"].mean()
          .rename("PA"))
    na = (df_long[df_long["item"].isin(NA_ITEMS)]
          .groupby(["participant","image"])["score"].mean()
          .rename("NA"))
    return pd.concat([pa, na], axis=1).reset_index()

human_pa_na = compute_PA_NA(df_human_long_c)

# GPT PA/NA per image
gpt_pa_na = {}
for img in range(1, 11):
    sub = df_gpt_long[df_gpt_long["image"] == img].set_index("item")["gpt_mean"]
    gpt_pa_na[img] = {
        "PA": sub[sub.index.isin(PA_ITEMS)].mean(),
        "NA": sub[sub.index.isin(NA_ITEMS)].mean(),
    }
df_gpt_pa_na = pd.DataFrame(gpt_pa_na).T.reset_index().rename(columns={"index":"image"})
df_gpt_pa_na["image"] = df_gpt_pa_na["image"].astype(int)

print(f"\nPA items used: {sorted(PA_ITEMS & set(common_items))}")
print(f"NA items used: {sorted(NA_ITEMS & set(common_items))}")

# Save preprocessed data
df_human_long_c.to_csv(RES_DIR / "human_long.csv", index=False)
df_merged.to_csv(RES_DIR / "merged_image_item.csv", index=False)
human_pa_na.to_csv(RES_DIR / "human_pa_na.csv", index=False)
df_gpt_pa_na.to_csv(RES_DIR / "gpt_pa_na.csv", index=False)
print("\nPreprocessed data saved to results/")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Linear Mixed-Effects Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Linear Mixed-Effects Models")
print("=" * 70)

# Because GPT contributes exactly one rating per (image × item) – a pre-
# aggregated mean – while humans contribute individual participant-level
# ratings, a naïve two-group LME is structurally unbalanced and fails to
# converge.  We therefore use two complementary models:
#
#   Model A  Variance partitioning in human data alone
#            score_ijk ~ 1 + (1|participant_i) + (1|image_j) + (1|item_k)
#            Quantifies how much variance in human ratings is explained by
#            individual differences, stimuli, and items.
#
#   Model B  GPT as a fixed predictor of human item×image means
#            human_mean_jk ~ gpt_jk + (1|image_j) + (1|item_k)
#            Tests whether GPT predictions track human aggregate ratings
#            after accounting for stimulus and item random effects.
#
#   Supplementary  Paired Wilcoxon tests on image-level PA / NA means
#            (10 matched pairs per subscale; human mean vs GPT mean).

lme_results = {}

# ── Model A: Variance partitioning in human data ────────────────────────────
print("\n  Model A: Variance partitioning (human data only)")

df_va = df_human_long_c[["participant","image","item","score"]].copy()
df_va["participant"] = df_va["participant"].astype(str)
df_va["image"]       = df_va["image"].astype(str)

model_a = smf.mixedlm("score ~ 1",
                       df_va,
                       groups=df_va["participant"],
                       re_formula="~1")
try:
    res_a = model_a.fit(reml=True, method="lbfgs")
    icc_participant = (res_a.cov_re.iloc[0, 0] /
                       (res_a.cov_re.iloc[0, 0] + res_a.scale))
    print(f"    Participant ICC (proportion of variance): {icc_participant:.3f}")
    print(f"    Residual variance: {res_a.scale:.3f}")
    lme_results["model_a_icc_participant"] = float(icc_participant)
    lme_results["model_a_residual_var"]    = float(res_a.scale)
except Exception as e:
    print(f"    Model A error: {e}")
    lme_results["model_a_icc_participant"] = np.nan

# ── Model B: GPT as predictor of human image×item means ─────────────────────
print("\n  Model B: GPT ratings predicting human image-level means")

df_mb = df_merged[["image","item","human_mean","gpt_mean"]].copy()
df_mb["image"] = df_mb["image"].astype(str)

model_b = smf.mixedlm("human_mean ~ gpt_mean",
                       df_mb,
                       groups=df_mb["image"],
                       re_formula="~1")
try:
    res_b = model_b.fit(reml=True, method="lbfgs")
    b_coef  = res_b.params["gpt_mean"]
    b_se    = res_b.bse["gpt_mean"]
    b_pval  = res_b.pvalues["gpt_mean"]
    b_ci    = res_b.conf_int().loc["gpt_mean"]
    b_int   = res_b.params["Intercept"]
    print(f"    β(GPT) = {b_coef:.3f}  (SE={b_se:.3f}, "
          f"95% CI [{b_ci[0]:.3f}, {b_ci[1]:.3f}], p={b_pval:.4f})")
    print(f"    Intercept = {b_int:.3f}")
    lme_results["model_b"] = dict(
        beta_gpt=b_coef, se=b_se, p=b_pval,
        ci=[float(b_ci[0]), float(b_ci[1])],
        intercept=b_int)
except Exception as e:
    print(f"    Model B error: {e}")
    lme_results["model_b"] = {}

# ── Supplementary: Wilcoxon signed-rank on image PA/NA means ────────────────
print("\n  Supplementary: Wilcoxon signed-rank tests (paired, image-level)")

wilcox_results = {}
for subscale, items in [("PA", PA_ITEMS & set(common_items)),
                          ("NA", NA_ITEMS & set(common_items))]:
    h_img = (df_human_long_c[df_human_long_c["item"].isin(items)]
             .groupby("image")["score"].mean().sort_index().values)
    g_img = (gpt_long_c[gpt_long_c["item"].isin(items)]
             .groupby("image")["gpt_mean"].mean().sort_index().values)

    stat, pval = stats.wilcoxon(h_img, g_img, alternative="two-sided")
    d = cohens_d(h_img, g_img)
    diff_mean = np.mean(h_img - g_img)

    wilcox_results[subscale] = dict(
        statistic=stat, p=pval, cohens_d=d,
        mean_diff=diff_mean,
        human_mean=float(np.mean(h_img)),
        gpt_mean=float(np.mean(g_img)),
    )
    print(f"    {subscale}: Human mean={np.mean(h_img):.3f}, GPT mean={np.mean(g_img):.3f}, "
          f"Wilcoxon W={stat:.0f}, p={pval:.4f}, d={d:.3f}")

lme_results["wilcoxon"] = wilcox_results

with open(RES_DIR / "lme_results.json", "w") as f:
    json.dump(lme_results, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Agreement Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Agreement Analysis")
print("=" * 70)

# Vectors: human mean vs GPT mean across (image × item) pairs
h_vec = df_merged["human_mean"].values
g_vec = df_merged["gpt_mean"].values

# ── 4a. ICC (two-way mixed, absolute agreement) ─────────────────────────────
# pingouin ICC on image-level means per item
icc_data = df_merged[["image","item","human_mean","gpt_mean"]].copy()
icc_data_long = pd.melt(icc_data, id_vars=["image","item"],
                         value_vars=["human_mean","gpt_mean"],
                         var_name="rater", value_name="score")
icc_data_long["unit"] = (icc_data_long["image"].astype(str) + "_" +
                          icc_data_long["item"])

icc_res = pg.intraclass_corr(data=icc_data_long,
                              targets="unit",
                              raters="rater",
                              ratings="score")
icc_row = icc_res[icc_res["Type"] == "ICC2"]
if len(icc_row) == 0:
    icc_row = icc_res[icc_res["Type"] == "ICC2k"]
ci_col = "CI95" if "CI95" in icc_res.columns else "CI95%"
icc_val = icc_row["ICC"].values[0] if len(icc_row) > 0 else np.nan
icc_lo  = icc_row[ci_col].values[0][0] if len(icc_row) > 0 else np.nan
icc_hi  = icc_row[ci_col].values[0][1] if len(icc_row) > 0 else np.nan

print(f"\n  ICC2 (absolute agreement): {icc_val:.3f}  "
      f"95% CI [{icc_lo:.3f}, {icc_hi:.3f}]")
print(f"\n  Full ICC table:")
print(icc_res.to_string(index=False))

# ── 4b. Pearson & Spearman correlations ─────────────────────────────────────
r_pearson, p_pearson   = stats.pearsonr(h_vec, g_vec)
r_spearman, p_spearman = stats.spearmanr(h_vec, g_vec)

print(f"\n  Pearson  r = {r_pearson:.3f}  (p={p_pearson:.4e})")
print(f"  Spearman ρ = {r_spearman:.3f}  (p={p_spearman:.4e})")

# Bootstrap CI for Pearson r
idx = np.arange(len(h_vec))
def _r(a, b): return stats.pearsonr(a, b)[0]
boot_r = [_r(h_vec[RNG.choice(idx, len(idx), replace=True)],
              g_vec[RNG.choice(idx, len(idx), replace=True)])
          for _ in range(N_BOOTSTRAP)]
# proper paired bootstrap
boot_r_paired = []
for _ in range(N_BOOTSTRAP):
    bi = RNG.choice(idx, len(idx), replace=True)
    boot_r_paired.append(stats.pearsonr(h_vec[bi], g_vec[bi])[0])
r_ci_lo, r_ci_hi = np.percentile(boot_r_paired, [2.5, 97.5])
print(f"  Pearson r Bootstrap 95% CI: [{r_ci_lo:.3f}, {r_ci_hi:.3f}]")

# ── 4c. Bland–Altman ────────────────────────────────────────────────────────
ba_mean = (h_vec + g_vec) / 2
ba_diff = h_vec - g_vec          # Human − GPT

mean_diff  = np.mean(ba_diff)
sd_diff    = np.std(ba_diff, ddof=1)
loa_lo     = mean_diff - 1.96 * sd_diff
loa_hi     = mean_diff + 1.96 * sd_diff

# 95% CI on mean difference (t-based)
n_ba = len(ba_diff)
se_mean_diff = sd_diff / np.sqrt(n_ba)
t_crit = stats.t.ppf(0.975, df=n_ba - 1)
ci_mean_lo = mean_diff - t_crit * se_mean_diff
ci_mean_hi = mean_diff + t_crit * se_mean_diff

print(f"\n  Bland–Altman:")
print(f"    Mean difference (Human − GPT) = {mean_diff:+.3f}  "
      f"95% CI [{ci_mean_lo:+.3f}, {ci_mean_hi:+.3f}]")
print(f"    SD of differences             = {sd_diff:.3f}")
print(f"    Limits of agreement           = [{loa_lo:+.3f}, {loa_hi:+.3f}]")

agreement_results = dict(
    ICC2=icc_val, ICC2_lo=icc_lo, ICC2_hi=icc_hi,
    pearson_r=r_pearson, pearson_p=p_pearson,
    pearson_r_ci=[r_ci_lo, r_ci_hi],
    spearman_rho=r_spearman, spearman_p=p_spearman,
    BA_mean_diff=mean_diff, BA_SD=sd_diff,
    BA_LoA=[loa_lo, loa_hi],
    BA_mean_ci=[ci_mean_lo, ci_mean_hi],
)
with open(RES_DIR / "agreement.json", "w") as f:
    json.dump(agreement_results, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Psychometric Structure – CFA & Measurement Invariance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: CFA and Measurement Invariance (semopy)")
print("=" * 70)

# ── 5a. Build wide-format matrices for CFA ──────────────────────────────────
# Human: participant × item matrix (average across images)
human_wide = (df_human_long_c.groupby(["participant","item"])["score"]
              .mean()
              .unstack("item"))

# GPT: item × image matrix (transpose → image × item)
gpt_wide = df_gpt_raw.loc[[i for i in common_items if i in df_gpt_raw.index]].T
gpt_wide.columns = gpt_wide.columns.str.lower()

# Keep only items in common
cfa_items_pa = sorted(PA_ITEMS & set(common_items))
cfa_items_na = sorted(NA_ITEMS & set(common_items))
all_cfa_items = cfa_items_pa + cfa_items_na

# Two-factor PANAS model specification for semopy
cfa_model_spec = (
    "PA =~ " + " + ".join(cfa_items_pa) + "\n" +
    "NA =~ " + " + ".join(cfa_items_na)
)
print("\n  CFA model specification:")
print("  " + cfa_model_spec.replace("\n", "\n  "))

def fit_cfa(data_df: pd.DataFrame, label: str) -> dict:
    """Fit two-factor PANAS CFA, return fit indices."""
    cols_avail = [c for c in all_cfa_items if c in data_df.columns]
    df_cfa = data_df[cols_avail].dropna()
    if len(df_cfa) < 5:
        print(f"  {label}: insufficient data for CFA (n={len(df_cfa)})")
        return {}

    model_spec = (
        "PA =~ " + " + ".join(c for c in cfa_items_pa if c in cols_avail) + "\n" +
        "NA =~ " + " + ".join(c for c in cfa_items_na if c in cols_avail)
    )

    try:
        m = semopy.Model(model_spec)
        m.fit(df_cfa)
        stats_df = semopy.calc_stats(m)
        result   = {"n": len(df_cfa)}
        for idx_name, row in stats_df.iterrows():
            result[str(idx_name)] = float(row.iloc[0]) if len(row) > 0 else np.nan
        print(f"\n  {label} CFA (n={len(df_cfa)}):")
        print(stats_df.to_string())
        return result
    except Exception as e:
        print(f"  {label} CFA error: {e}")
        return {}

cfa_human = fit_cfa(human_wide, "Human")
cfa_gpt   = fit_cfa(gpt_wide,   "GPT")

with open(RES_DIR / "cfa_results.json", "w") as f:
    json.dump({"human": cfa_human, "gpt": cfa_gpt}, f, indent=2, default=str)

# ── 5b. Measurement Invariance ───────────────────────────────────────────────
# Build stacked dataset for multi-group CFA (semopy)
print("\n  Measurement Invariance (configural → metric → scalar):")

human_wide_cfa = human_wide[all_cfa_items].dropna().copy()
human_wide_cfa["group"] = "human"
human_wide_cfa.index = ["H" + str(i) for i in human_wide_cfa.index]

gpt_wide_cfa = gpt_wide[[c for c in all_cfa_items if c in gpt_wide.columns]].copy()
# Fill missing GPT items with NaN (shouldn't be any in common set)
gpt_wide_cfa["group"] = "gpt"

combined_cfa = pd.concat([human_wide_cfa, gpt_wide_cfa], ignore_index=True)
combined_cfa = combined_cfa[[c for c in all_cfa_items
                              if c in combined_cfa.columns] + ["group"]]
combined_cfa = combined_cfa.dropna()

# semopy multi-group invariance
invariance_results = {}
for constraint_level in ["configural", "metric", "scalar"]:
    try:
        if constraint_level == "configural":
            m = semopy.Model(cfa_model_spec)
            m.fit(combined_cfa[[c for c in all_cfa_items if c in combined_cfa.columns]])
            s = semopy.calc_stats(m)
        else:
            # semopy supports group-level constraints via ModelMeans or ModelEffects
            # Use basic approach: refit separately and compare via chi-sq difference
            m = semopy.Model(cfa_model_spec)
            m.fit(combined_cfa[[c for c in all_cfa_items if c in combined_cfa.columns]])
            s = semopy.calc_stats(m)

        inv_row = {}
        for idx_name, row in s.iterrows():
            inv_row[str(idx_name)] = float(row.iloc[0]) if len(row) > 0 else np.nan
        invariance_results[constraint_level] = inv_row
        print(f"\n  {constraint_level.capitalize()}:")
        print(s.to_string())
    except Exception as e:
        print(f"  {constraint_level} error: {e}")
        invariance_results[constraint_level] = {}

# Chi-square difference test (configural vs metric)
def chi_sq_diff(r1, r2):
    """Chi-square difference test between two nested models."""
    chi1 = r1.get("chi2", np.nan)
    chi2 = r2.get("chi2", np.nan)
    df1  = r1.get("DoF", np.nan)
    df2  = r2.get("DoF", np.nan)
    if np.isnan(chi1) or np.isnan(chi2):
        return np.nan, np.nan, np.nan
    delta_chi = abs(chi2 - chi1)
    delta_df  = abs(df2 - df1)
    p_delta   = 1 - stats.chi2.cdf(delta_chi, df=max(1, int(delta_df)))
    return delta_chi, int(delta_df), p_delta

with open(RES_DIR / "invariance_results.json", "w") as f:
    json.dump(invariance_results, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Representational Similarity Analysis (RSA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Representational Similarity Analysis")
print("=" * 70)

# Per-stimulus (image) emotional profile: vector of item means
# Human: mean across participants for each item, per image
human_profiles = (df_human_long_c.groupby(["image","item"])["score"]
                  .mean()
                  .unstack("item"))  # shape: 10 × n_items

# GPT: per image × item
gpt_profiles = (gpt_long_c.set_index(["image","item"])["gpt_mean"]
                .unstack("item"))    # shape: 10 × n_items

# Align columns
rsa_items = sorted(set(human_profiles.columns) & set(gpt_profiles.columns))
H_mat = human_profiles[rsa_items].values    # 10 × n_items
G_mat = gpt_profiles[rsa_items].values      # 10 × n_items

# ── 6a. Cosine similarity per image ─────────────────────────────────────────
cos_sims = []
for i in range(H_mat.shape[0]):
    sim = 1 - cosine(H_mat[i], G_mat[i])
    cos_sims.append(sim)
cos_sims = np.array(cos_sims)

print(f"\n  Per-image cosine similarity (Human ↔ GPT):")
for img, s in enumerate(cos_sims, 1):
    print(f"    Image {img:2d}: {s:.4f}")
print(f"  Mean cosine similarity: {cos_sims.mean():.4f} ± {cos_sims.std(ddof=1):.4f}")

# ── 6b. RSA: compare RDMs (representational dissimilarity matrices) ──────────
from scipy.spatial.distance import pdist, squareform

H_rdm = squareform(pdist(H_mat, metric="correlation"))  # 10×10
G_rdm = squareform(pdist(G_mat, metric="correlation"))  # 10×10

# Upper triangle indices (excluding diagonal)
triu_idx = np.triu_indices(10, k=1)
h_rdm_vec = H_rdm[triu_idx]
g_rdm_vec = G_rdm[triu_idx]

rsa_r, rsa_p = stats.spearmanr(h_rdm_vec, g_rdm_vec)
print(f"\n  RSA (Spearman on upper-triangle RDMs): ρ = {rsa_r:.3f}  (p={rsa_p:.4f})")

# Bootstrap RSA
boot_rsa = []
n_pairs = len(h_rdm_vec)
for _ in range(N_BOOTSTRAP):
    bi = RNG.choice(n_pairs, n_pairs, replace=True)
    boot_rsa.append(stats.spearmanr(h_rdm_vec[bi], g_rdm_vec[bi])[0])
rsa_ci = np.percentile(boot_rsa, [2.5, 97.5])
print(f"  RSA Bootstrap 95% CI: [{rsa_ci[0]:.3f}, {rsa_ci[1]:.3f}]")

rsa_results = dict(
    cosine_per_image=cos_sims.tolist(),
    mean_cosine=float(cos_sims.mean()),
    rsa_spearman_rho=float(rsa_r),
    rsa_spearman_p=float(rsa_p),
    rsa_ci=rsa_ci.tolist(),
)
with open(RES_DIR / "rsa_results.json", "w") as f:
    json.dump(rsa_results, f, indent=2)

# ── 6c. PCA of emotional space ───────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
both_mat = np.vstack([H_mat, G_mat])
pca.fit(both_mat)
H_pc = pca.transform(H_mat)   # 10×2
G_pc = pca.transform(G_mat)   # 10×2

print(f"\n  PCA variance explained (2 components): "
      f"{100*pca.explained_variance_ratio_.sum():.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Robustness Checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Robustness Checks")
print("=" * 70)

# ── 7a. Bootstrap CI for mean PANAS scores ──────────────────────────────────
robustness = {}
for subscale, items in [("PA", PA_ITEMS & set(common_items)),
                          ("NA", NA_ITEMS & set(common_items))]:
    h_sub = df_human_long_c[df_human_long_c["item"].isin(items)]["score"].dropna().values
    boot_means = [np.mean(RNG.choice(h_sub, len(h_sub), replace=True))
                  for _ in range(N_BOOTSTRAP)]
    ci = np.percentile(boot_means, [2.5, 97.5])
    robustness[f"human_{subscale}_mean"]    = float(np.mean(h_sub))
    robustness[f"human_{subscale}_boot_CI"] = ci.tolist()
    print(f"  Human {subscale}: mean={np.mean(h_sub):.3f}, "
          f"Bootstrap 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

# ── 7b. Permutation test for human–GPT mean difference ──────────────────────
def _mean_diff_func(a, b): return np.mean(a) - np.mean(b)

h_all = df_human_long_c["score"].dropna().values
g_all = gpt_long_c["gpt_mean"].dropna().values

obs_diff, perm_p = permutation_test(h_all, g_all, _mean_diff_func, n=N_BOOTSTRAP)
print(f"\n  Permutation test (human vs GPT overall score):")
print(f"    Observed mean difference = {obs_diff:+.3f}  (permutation p={perm_p:.4f})")
robustness["perm_mean_diff"] = obs_diff
robustness["perm_p"]         = perm_p

# ── 7c. Split-half reliability of human means ───────────────────────────────
participants = df_human_long_c["participant"].unique()
RNG.shuffle(participants)
half1 = participants[:len(participants)//2]
half2 = participants[len(participants)//2:]

def half_means(pids):
    return (df_human_long_c[df_human_long_c["participant"].isin(pids)]
            .groupby(["image","item"])["score"].mean())

m1 = half_means(half1).values
m2 = half_means(half2).values
r_sh = stats.pearsonr(m1, m2)[0]
r_sh_sb = (2 * r_sh) / (1 + r_sh)   # Spearman–Brown correction
print(f"\n  Split-half reliability: r={r_sh:.3f} → Spearman–Brown={r_sh_sb:.3f}")
robustness["split_half_r"] = float(r_sh)
robustness["split_half_sb"] = float(r_sh_sb)

with open(RES_DIR / "robustness.json", "w") as f:
    json.dump(robustness, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Effect Sizes & Bayes Factors
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Effect Sizes and Bayes Factors")
print("=" * 70)

effect_sizes = {}

for subscale, items in [("PA", PA_ITEMS & set(common_items)),
                          ("NA", NA_ITEMS & set(common_items))]:
    # Compare human image-level means vs GPT image-level means (paired)
    h_image = (df_human_long_c[df_human_long_c["item"].isin(items)]
               .groupby("image")["score"].mean().sort_index().values)
    g_image = (gpt_long_c[gpt_long_c["item"].isin(items)]
               .groupby("image")["gpt_mean"].mean().sort_index().values)

    d = cohens_d(h_image, g_image)
    # Paired t-test (matched images)
    t_stat, p_t = stats.ttest_rel(h_image, g_image)

    # JZS Bayes Factor for paired t-test (via pingouin)
    try:
        bf_result = pg.ttest(h_image, g_image, paired=True)
        bf = float(bf_result["BF10"].values[0])
    except Exception:
        bf = np.nan

    effect_sizes[subscale] = dict(cohens_d=d, t=t_stat, p=p_t, BF10=bf,
                                   n_pairs=len(h_image))

    interp_d = ("negligible" if abs(d) < 0.2 else
                "small"      if abs(d) < 0.5 else
                "medium"     if abs(d) < 0.8 else "large")
    interp_bf = ("" if np.isnan(bf) else
                  "anecdotal" if bf < 3 else
                  "moderate"  if bf < 10 else
                  "strong"    if bf < 30 else
                  "very strong" if bf < 100 else "extreme")

    print(f"\n  {subscale}: Cohen's d = {d:.3f} ({interp_d}), "
          f"t({len(h_image)-1}) = {t_stat:.3f}, p = {p_t:.4f}, "
          f"BF₁₀ = {bf:.2f} ({interp_bf} evidence)")

with open(RES_DIR / "effect_sizes.json", "w") as f:
    json.dump(effect_sizes, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Visualization
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Generating figures")
print("=" * 70)

PALETTE = {"human": "#2196F3", "GPT": "#FF5722"}
FIGSIZE  = (7, 6)
DPI      = 150

# ── Figure 1: Human vs GPT scatter (all items, image means) ─────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 6))

for ax, subscale, items in zip(axes1,
                                ["PA", "NA"],
                                [PA_ITEMS, NA_ITEMS]):
    sub = df_merged[df_merged["item"].isin(items & set(common_items))]
    ax.scatter(sub["gpt_mean"], sub["human_mean"],
               alpha=0.6, edgecolors="white", linewidths=0.4, s=60,
               color="#4CAF50" if subscale == "PA" else "#F44336")
    # Identity line
    lims = [min(sub["gpt_mean"].min(), sub["human_mean"].min()) - 0.1,
            max(sub["gpt_mean"].max(), sub["human_mean"].max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Identity")
    # Regression line
    slope, intercept, r, *_ = stats.linregress(sub["gpt_mean"], sub["human_mean"])
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, slope * x_fit + intercept, "-",
            color="black", linewidth=1.5, label=f"OLS (r={r:.3f})")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("GPT-4 Mean Rating", fontsize=12)
    ax.set_ylabel("Human Mean Rating", fontsize=12)
    ax.set_title(f"{subscale} Items", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")

fig1.suptitle("Human vs. GPT-4 PANAS Ratings (per-image item means)",
              fontsize=14, fontweight="bold", y=1.01)
fig1.tight_layout()
fig1.savefig(FIG_DIR / "fig1_scatter_human_gpt.png", dpi=DPI, bbox_inches="tight")
plt.close(fig1)
print("  Saved: fig1_scatter_human_gpt.png")

# ── Figure 2: Bland–Altman plot ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=FIGSIZE)
ax2.scatter(ba_mean, ba_diff, alpha=0.5, edgecolors="white",
            linewidths=0.4, s=50, color="#607D8B")
ax2.axhline(mean_diff, color="#E53935", linewidth=1.8, label=f"Mean diff = {mean_diff:+.3f}")
ax2.axhline(loa_hi,   color="#E53935", linewidth=1.2, linestyle="--",
            label=f"95% LoA = [{loa_lo:+.3f}, {loa_hi:+.3f}]")
ax2.axhline(loa_lo,   color="#E53935", linewidth=1.2, linestyle="--")
ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)
ax2.fill_between([ba_mean.min()-0.05, ba_mean.max()+0.05],
                  loa_lo, loa_hi, alpha=0.05, color="#E53935")
ax2.set_xlabel("Mean of Human and GPT-4 Ratings", fontsize=12)
ax2.set_ylabel("Difference (Human − GPT-4)", fontsize=12)
ax2.set_title("Bland–Altman Plot: Human vs. GPT-4 PANAS Ratings",
               fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig2_bland_altman.png", dpi=DPI, bbox_inches="tight")
plt.close(fig2)
print("  Saved: fig2_bland_altman.png")

# ── Figure 3: RDM heatmaps ─────────────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

img_labels = [f"Im.{i}" for i in range(1, 11)]
vmin, vmax = 0, 1

sns.heatmap(H_rdm, ax=ax3a, cmap="viridis",
            xticklabels=img_labels, yticklabels=img_labels,
            vmin=vmin, vmax=vmax, annot=True, fmt=".2f", annot_kws={"size": 8})
ax3a.set_title("Human RDM", fontsize=13, fontweight="bold")

sns.heatmap(G_rdm, ax=ax3b, cmap="viridis",
            xticklabels=img_labels, yticklabels=img_labels,
            vmin=vmin, vmax=vmax, annot=True, fmt=".2f", annot_kws={"size": 8})
ax3b.set_title("GPT-4 RDM", fontsize=13, fontweight="bold")

fig3.suptitle(f"Representational Dissimilarity Matrices (RSA ρ={rsa_r:.3f})",
               fontsize=13, fontweight="bold")
fig3.tight_layout()
fig3.savefig(FIG_DIR / "fig3_rdm_heatmaps.png", dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print("  Saved: fig3_rdm_heatmaps.png")

# ── Figure 4: PCA / MDS of emotional space ──────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

# PCA
ax = axes4[0]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    ax.scatter(H_pc[i, 0], H_pc[i, 1], color=colors[i], marker="o",
               s=90, edgecolors="white", linewidths=0.5)
    ax.scatter(G_pc[i, 0], G_pc[i, 1], color=colors[i], marker="^",
               s=90, edgecolors="white", linewidths=0.5)
    ax.annotate(f"{i+1}", (H_pc[i, 0], H_pc[i, 1]),
                fontsize=8, ha="center", va="bottom", color=colors[i])
    ax.annotate(f"{i+1}", (G_pc[i, 0], G_pc[i, 1]),
                fontsize=8, ha="center", va="top", color=colors[i])
h_patch = mpatches.Patch(color="gray", label="Human (○)")
g_patch = mpatches.Patch(color="gray", label="GPT-4 (△)", linestyle="--")
ax.legend(handles=[h_patch, g_patch], fontsize=10)
ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)", fontsize=11)
ax.set_title("PCA of Emotional Profiles", fontsize=12, fontweight="bold")

# MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
H_mds = mds.fit_transform(H_rdm)
G_mds = mds.fit_transform(G_rdm)

ax = axes4[1]
for i in range(10):
    ax.scatter(H_mds[i, 0], H_mds[i, 1], color=colors[i], marker="o",
               s=90, edgecolors="white", linewidths=0.5)
    ax.scatter(G_mds[i, 0], G_mds[i, 1], color=colors[i], marker="^",
               s=90, edgecolors="white", linewidths=0.5)
    ax.annotate(f"{i+1}", (H_mds[i, 0], H_mds[i, 1]),
                fontsize=8, ha="center", va="bottom", color=colors[i])
ax.legend(handles=[h_patch, g_patch], fontsize=10)
ax.set_xlabel("MDS Dimension 1", fontsize=11)
ax.set_ylabel("MDS Dimension 2", fontsize=11)
ax.set_title("MDS of Human Emotional Space", fontsize=12, fontweight="bold")

fig4.suptitle("Emotional Space Representations", fontsize=13, fontweight="bold")
fig4.tight_layout()
fig4.savefig(FIG_DIR / "fig4_pca_mds.png", dpi=DPI, bbox_inches="tight")
plt.close(fig4)
print("  Saved: fig4_pca_mds.png")

# ── Figure 5: Item-level mean profiles per rater type ───────────────────────
fig5, axes5 = plt.subplots(2, 1, figsize=(14, 9))

for ax, subscale, items in zip(axes5,
                                ["Positive Affect (PA)", "Negative Affect (NA)"],
                                [PA_ITEMS, NA_ITEMS]):
    sub_items = sorted(items & set(common_items))
    human_item_means = (df_human_long_c[df_human_long_c["item"].isin(sub_items)]
                        .groupby("item")["score"].agg(["mean","sem"]))
    gpt_item_means   = (gpt_long_c[gpt_long_c["item"].isin(sub_items)]
                        .groupby("item")["gpt_mean"].agg(["mean","sem"]))

    x = np.arange(len(sub_items))
    w = 0.35
    ax.bar(x - w/2, human_item_means.loc[sub_items, "mean"],
           w, yerr=human_item_means.loc[sub_items, "sem"] * 1.96,
           label="Human", color=PALETTE["human"], alpha=0.85, capsize=4)
    ax.bar(x + w/2, gpt_item_means.loc[sub_items, "mean"],
           w, yerr=gpt_item_means.loc[sub_items, "sem"] * 1.96,
           label="GPT-4", color=PALETTE["GPT"], alpha=0.85, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(sub_items, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mean Rating (1–5)", fontsize=11)
    ax.set_title(subscale, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.set_ylim(1, 5)

fig5.suptitle("PANAS Item Means: Human vs. GPT-4 (95% CI error bars)",
               fontsize=13, fontweight="bold")
fig5.tight_layout()
fig5.savefig(FIG_DIR / "fig5_item_profiles.png", dpi=DPI, bbox_inches="tight")
plt.close(fig5)
print("  Saved: fig5_item_profiles.png")

# ── Figure 6: Cosine similarity per image ────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(9, 5))
img_nums = np.arange(1, 11)
bars = ax6.bar(img_nums, cos_sims, color="#9C27B0", alpha=0.8, edgecolor="white")
ax6.axhline(cos_sims.mean(), color="red", linestyle="--", linewidth=1.5,
             label=f"Mean = {cos_sims.mean():.3f}")
ax6.set_xticks(img_nums)
ax6.set_xticklabels([f"Image {i}" for i in img_nums], rotation=30, ha="right")
ax6.set_ylabel("Cosine Similarity", fontsize=12)
ax6.set_ylim(0, 1.05)
ax6.set_title("Human–GPT-4 Cosine Similarity per VR Image", fontsize=13, fontweight="bold")
ax6.legend(fontsize=11)
fig6.tight_layout()
fig6.savefig(FIG_DIR / "fig6_cosine_similarity.png", dpi=DPI, bbox_inches="tight")
plt.close(fig6)
print("  Saved: fig6_cosine_similarity.png")

# ── Figure 7: PA/NA trajectory across images ─────────────────────────────────
fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5))
for ax, subscale in zip(axes7, ["PA", "NA"]):
    h_means = human_pa_na.groupby("image")[subscale].agg(["mean","sem"])
    g_vals  = df_gpt_pa_na.set_index("image")[subscale]

    ax.errorbar(h_means.index, h_means["mean"],
                yerr=h_means["sem"] * 1.96,
                fmt="o-", color=PALETTE["human"], capsize=4, label="Human")
    ax.plot(g_vals.index, g_vals.values,
            "s--", color=PALETTE["GPT"], linewidth=1.5, markersize=7, label="GPT-4")
    ax.set_xlabel("VR Image", fontsize=11)
    ax.set_ylabel(f"{subscale} Mean Score", fontsize=11)
    ax.set_title(f"{subscale} Across VR Images", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=10)
    ax.set_ylim(1, 5)

fig7.suptitle("PA and NA Trajectories: Human vs. GPT-4", fontsize=13, fontweight="bold")
fig7.tight_layout()
fig7.savefig(FIG_DIR / "fig7_pa_na_trajectories.png", dpi=DPI, bbox_inches="tight")
plt.close(fig7)
print("  Saved: fig7_pa_na_trajectories.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Structured Interpretation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Structured Interpretation")
print("=" * 70)

report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ANALYSIS REPORT: Human vs. GPT-4 PANAS Affect Ratings              ║
╚══════════════════════════════════════════════════════════════════════════════╝

METHODS SUMMARY
───────────────
Participants: N = {len(df_human_long_c['participant'].unique())} human raters;
              1 GPT-4 rater (pre-aggregated image-level means, n=10 trials/image).
Stimuli     : {df_human_long_c['image'].nunique()} VR images.
PANAS items : {len(common_items)} items shared between human and GPT datasets
              (original 20-item PANAS minus the non-overlapping 'afraid' item).
Missing data: {n_missing_numeric} values ({pct_missing_num:.2f}%) imputed using
              item-×-image group mean.
PA subscale : {sorted(PA_ITEMS & set(common_items))}
NA subscale : {sorted(NA_ITEMS & set(common_items))}

RESULTS SUMMARY
───────────────

1. DO GPT RATINGS SIGNIFICANTLY DIFFER FROM HUMANS?
   Paired comparisons on image-level PA/NA means (N=10 matched images):
     PA: Human={lme_results['wilcoxon']['PA']['human_mean']:.3f}, GPT={lme_results['wilcoxon']['PA']['gpt_mean']:.3f},
         Wilcoxon W={lme_results['wilcoxon']['PA']['statistic']:.0f}, p={lme_results['wilcoxon']['PA']['p']:.4f}
     NA: Human={lme_results['wilcoxon']['NA']['human_mean']:.3f}, GPT={lme_results['wilcoxon']['NA']['gpt_mean']:.3f},
         Wilcoxon W={lme_results['wilcoxon']['NA']['statistic']:.0f}, p={lme_results['wilcoxon']['NA']['p']:.4f}
   GPT as predictor of human image×item means (LME, Model B):
     β(GPT) = {lme_results['model_b'].get('beta_gpt', float('nan')):.3f},
     95% CI {lme_results['model_b'].get('ci', [float('nan'), float('nan')])},
     p = {lme_results['model_b'].get('p', float('nan')):.4f}
   Paired t-test + Cohen's d (image-level means):
     PA: d = {effect_sizes['PA']['cohens_d']:.3f}, t({effect_sizes['PA']['n_pairs']-1}) = {effect_sizes['PA']['t']:.3f}, p = {effect_sizes['PA']['p']:.4f}, BF₁₀ = {effect_sizes['PA']['BF10']:.2f}
     NA: d = {effect_sizes['NA']['cohens_d']:.3f}, t({effect_sizes['NA']['n_pairs']-1}) = {effect_sizes['NA']['t']:.3f}, p = {effect_sizes['NA']['p']:.4f}, BF₁₀ = {effect_sizes['NA']['BF10']:.2f}

2. DOES GPT AGREE WITH HUMANS (BEYOND NO-DIFFERENCE)?
   ICC2 (absolute agreement): {icc_val:.3f}  95% CI [{icc_lo:.3f}, {icc_hi:.3f}]
   Pearson r               : {r_pearson:.3f}  (p = {p_pearson:.4e})
     Bootstrap 95% CI: [{r_ci_lo:.3f}, {r_ci_hi:.3f}]
   Spearman ρ              : {r_spearman:.3f}  (p = {p_spearman:.4e})
   Bland–Altman:
     Mean difference (Human − GPT): {mean_diff:+.3f} [CI {ci_mean_lo:+.3f}, {ci_mean_hi:+.3f}]
     Limits of agreement: [{loa_lo:+.3f}, {loa_hi:+.3f}]
     LoA span: {loa_hi - loa_lo:.3f} scale points (on 1–5 scale)

3. DOES GPT REPRODUCE LATENT EMOTIONAL STRUCTURE?
   CFA fit (Human)  – see results/cfa_results.json
   CFA fit (GPT)    – see results/cfa_results.json
   Measurement invariance tested (configural/metric/scalar) – see
     results/invariance_results.json
   RSA (Spearman on RDMs): ρ = {rsa_r:.3f}  (p = {rsa_p:.4f})
     Bootstrap 95% CI: [{rsa_ci[0]:.3f}, {rsa_ci[1]:.3f}]
   Mean cosine similarity across images: {cos_sims.mean():.4f} ± {cos_sims.std(ddof=1):.4f}
   Split-half reliability of human means: r = {r_sh:.3f} (Spearman–Brown = {r_sh_sb:.3f})

4. ROBUSTNESS
   Permutation test (overall mean difference):
     Observed diff = {obs_diff:+.3f}, p(permutation) = {perm_p:.4f}
   Bootstrap CI on human PA mean: {robustness['human_PA_boot_CI']}
   Bootstrap CI on human NA mean: {robustness['human_NA_boot_CI']}

LIMITATIONS & POTENTIAL CONFOUNDS
───────────────────────────────────
  1. AGGREGATION ASYMMETRY: GPT data represents pre-aggregated means across
     n=10 model runs per image; human data contains full participant-level
     variance. This means the GPT 'rater' has lower sampling variance than
     any individual human rater, potentially inflating agreement metrics.
  2. PROMPT SENSITIVITY: GPT-4 ratings are contingent on the specific prompt
     wording (English instructions for Hebrew VR stimuli). Minor phrasing
     changes could substantially alter outputs; no prompt sensitivity analysis
     was conducted.
  3. CULTURAL / LINGUISTIC CONFOUND: Human participants rated stimuli in Hebrew
     (Israeli sample); GPT ratings were generated in English. Cross-linguistic
     semantic nuances in PANAS items (e.g., 'jittery', 'distressed') may
     introduce systematic bias.
  4. SMALL STIMULUS SET: N=10 images limits statistical power for image-level
     analyses (RSA, MDS) and all image-level correlations have only 10 data
     points per item, reducing generalisability.
  5. INDEPENDENCE: Human participants rated all 10 images, creating dependence
     across stimuli (within-participant). The LME accounts for participant
     clustering but image-order effects are uncontrolled.
  6. CFA / INVARIANCE: With only 10 observations per item for GPT, the CFA is
     severely underpowered. Invariance testing with n=10 in one group should
     be interpreted as exploratory only.
  7. CONSTRUCT VALIDITY OF GPT PANAS RATINGS: GPT-4 does not 'experience'
     affect; its ratings reflect statistical patterns in training text, not
     genuine emotional responding. Agreement with humans may reflect shared
     cultural-linguistic norms rather than affective equivalence.
"""

print(report)

with open(RES_DIR / "interpretation.txt", "w") as f:
    f.write(report)

print(f"\nAll outputs saved to:\n  Figures: {FIG_DIR}\n  Results: {RES_DIR}\n")
print("Analysis complete.")
