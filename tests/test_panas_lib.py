import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import panas_lib as L


def test_icc_matches_pingouin():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 1, 200)
    y = x + rng.normal(0.2, 0.5, 200)
    df = pd.DataFrame({"t": np.tile(np.arange(200), 2), "r": np.repeat(["a", "b"], 200),
                       "s": np.concatenate([x, y])})
    ref = pg.intraclass_corr(df, targets="t", raters="r", ratings="s").set_index("Type")["ICC"]
    # pingouin >= 0.6 labels ICC2/ICC2k as ICC(A,1)/ICC(A,k)
    ref = ref.rename({"ICC(A,1)": "ICC2", "ICC(A,k)": "ICC2k"})
    i21, i2k = L.icc2(x, y)
    assert abs(i21 - ref["ICC2"]) < 1e-9 and abs(i2k - ref["ICC2k"]) < 1e-9


def test_perfect_and_offset():
    H = np.random.default_rng(1).uniform(1, 5, (10, 20))
    m = L.agreement_metrics(H, H)
    assert m["ICC2_1"] > 0.999 and m["MAE"] == 0 and m["RSA_correlation"] > 0.999
    m2 = L.agreement_metrics(H, H + 0.5)
    assert abs(m2["BA_bias"] + 0.5) < 1e-9 and abs(m2["MAE"] - 0.5) < 1e-9
    assert m2["pearson"] > 0.999 and m2["ICC2_1"] < m["ICC2_1"]


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "data" / "human.csv").exists(),
                    reason="raw human data (data/human.csv) is not public")
def test_human_array_complete():
    df, acc = L.load_human_raw()
    A = L.human_array(df)
    assert A.shape == (acc["final_N"], 10, 20)


def test_item_mapping_follows_ben_zur():
    assert L.HEBREW_ITEM_MAP["עצבני"] == "nervous"
    assert L.HEBREW_ITEM_MAP["מתח"] == "jittery"
    assert L.HEBREW_ITEM_MAP["לחוץ"] == "distressed"
    assert sorted(L.HEBREW_ITEM_MAP.values()) == sorted(L.PANAS_ITEMS)


def test_within_item_r_ignores_item_offsets():
    rng = np.random.default_rng(3)
    H = rng.uniform(1, 5, (10, 20))
    G = H + rng.normal(0, 2, 20)            # item-specific offsets only
    assert L.agreement_metrics(H, G)["r_within_item"] > 0.999
