"""Paired image bootstrap: ICC(2,1)(model) - ICC(2,1)(no-image baseline), with the
same resampled images for both, so the comparison is not confounded by which
environments happen to be drawn. Writes <out>/paired_baseline_image_bootstrap.csv."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import panas_lib as L

ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = np.random.default_rng(2026)
A = L.impute_cell_mean(L.human_array(L.load_human_raw()[0])); H = np.nanmean(A, 0)
idx = rng.integers(0, L.N_IMAGES, (a.B, L.N_IMAGES))
base = np.array([L.icc2(H[i].ravel(), L.item_mean_baseline(H[i]).ravel())[0] for i in idx])
rows = []
for m, g in L.load_runs_long().groupby("model"):
    G = L.runs_array(g).mean(0)
    mod = np.array([L.icc2(H[i].ravel(), G[i].ravel())[0] for i in idx])
    d = mod - base
    rows.append({"model": m, "ICC_minus_baseline": L.icc2(H.ravel(), G.ravel())[0] - L.icc2(H.ravel(), L.item_mean_baseline(H).ravel())[0],
                 "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p_le_0": float((d <= 0).mean())})
out = pd.DataFrame(rows).sort_values("ICC_minus_baseline", ascending=False)
out.to_csv(Path(a.out) / "paired_baseline_image_bootstrap.csv", index=False)
print(out.round(3).to_string(index=False))
print("models whose 95% CI excludes 0 (above baseline):", int((out.lo > 0).sum()), "of", len(out))
