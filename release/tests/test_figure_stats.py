import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import panas_lib as L
import figure_stats as F


def test_icc_batch_matches_reference():
    rng = np.random.default_rng(0)
    h = rng.normal(3, 1, (5, 200)); g = h + rng.normal(0.3, 0.5, (5, 200))
    ref = [L.icc2(h[i], g[i])[0] for i in range(5)]
    assert np.allclose(F.icc21_batch(h, g), ref)


def test_within_item_r_ignores_item_offsets():
    rng = np.random.default_rng(1)
    H = rng.uniform(1, 5, (10, 20))
    item_offset = rng.uniform(-1.5, 1.5, 20)
    G = H + item_offset                      # same scene pattern, wrong item levels
    pooled, mean_r = F.within_item_r(H, G)
    assert pooled > 0.999 and mean_r > 0.999
    assert F.icc21_batch(H.ravel()[None], G.ravel()[None])[0] < 0.95


def test_item_error_sign_negative_when_model_lower():
    A = np.full((4, 10, 20), 3.0)
    runs = {"m": np.full((10, 10, 20), 2.0)}
    df, summ = F.item_errors(A, runs)
    assert (df["signed_error"] < 0).all() and (summ["share_models_below_zero"] == 1).all()
    assert set(summ.loc[summ["adapted"], "item"]) == set(F.ADAPTED_ITEMS)


def test_exclusion_helper_drops_exactly_the_aliases():
    df = pd.DataFrame({"model": ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash",
                                 "gemini-3.8-flash", "gpt-4.1"]})
    kept = L.drop_excluded(df)["model"].tolist()
    assert kept == ["gemini-3.6-flash", "gemini-3.8-flash", "gpt-4.1"]


def test_environment_interval_narrows_with_more_environments():
    rng = np.random.default_rng(2)
    widths = []
    for n_env in (10, 40):
        H = rng.uniform(1, 5, (n_env, 20)); G = H + rng.normal(0, 0.6, H.shape)
        idx = rng.integers(0, n_env, (500, n_env))
        icc = F.icc21_batch(H[idx].reshape(500, -1), G[idx].reshape(500, -1))
        widths.append(np.percentile(icc, 97.5) - np.percentile(icc, 2.5))
    assert widths[1] < widths[0]
