import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vlm_prompt as vp

GOOD = {i: 3.0 for i in vp.PANAS_ITEMS}


def test_parse_fenced_and_trailing_comma():
    raw = "```json\n" + json.dumps(GOOD)[:-1] + ",}\n```"
    assert vp.parse_response(raw)["alert"] == 3.0


def test_parse_with_surrounding_text():
    assert vp.parse_response("Sure!\n" + json.dumps(GOOD) + "\nDone")["afraid"] == 3.0


def test_parse_no_json():
    with pytest.raises(ValueError):
        vp.parse_response("no json here")


def test_validate_ok_and_case_insensitive():
    scores, why = vp.validate_scores({k.upper(): v for k, v in GOOD.items()})
    assert scores and why == ""


def test_validate_missing_nonnumeric_range():
    bad = dict(GOOD); bad.pop("proud")
    assert vp.validate_scores(bad)[0] is None
    bad = dict(GOOD); bad["proud"] = "high"
    assert vp.validate_scores(bad)[0] is None
    bad = dict(GOOD); bad["proud"] = 5.5
    assert vp.validate_scores(bad)[0] is None


def _fake_registry(monkeypatch, tmp_path, responder):
    monkeypatch.setattr(vp, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(vp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(vp, "RUNS_LONG", tmp_path / "runs_long.csv")
    monkeypatch.setattr(vp, "CONFIG_FILE", tmp_path / "runs" / "model_config.json")
    (tmp_path / "runs").mkdir()
    spec = vp.Model("fake", "fake-1")
    monkeypatch.setitem(vp.ALL_MODELS, "fake", spec)
    monkeypatch.setitem(vp.PROVIDER_DISPATCH, "fake", responder)
    monkeypatch.setattr(vp.time, "sleep", lambda *_: None)
    vp._dropped_params.clear()


def test_collect_resume_and_fallback(monkeypatch, tmp_path):
    calls = {"n": 0}

    def responder(spec, path, prompt, opts):
        calls["n"] += 1
        if "temperature" in opts:
            raise RuntimeError("400 Unsupported value: 'temperature' is not supported")
        text = "not json" if calls["n"] % 4 == 0 else json.dumps(GOOD)
        return dict(text=text, returned_model="fake-1-x", finish_reason="stop",
                    in_tokens=1, out_tokens=1)

    _fake_registry(monkeypatch, tmp_path, responder)
    photo = vp.list_photos()[0]
    n = vp.collect_image("fake", vp.ALL_MODELS["fake"], "main", 1, photo, 5,
                         "p", "h", 0.7, 0)
    assert n == 5
    log = vp.read_log(vp.run_file("fake", "main"))
    ok = [r for r in log if r["status"] == "ok"]
    assert sorted(r["run"] for r in ok) == [1, 2, 3, 4, 5]
    assert all(r["temperature_applied"] is None and r["temperature_requested"] == 0.7
               for r in log)
    assert any(r["status"] == "invalid" for r in log)      # kept, not averaged
    # resume: nothing more to do
    before = len(log)
    vp.collect_image("fake", vp.ALL_MODELS["fake"], "main", 1, photo, 5, "p", "h", 0.7, 0)
    assert len(vp.read_log(vp.run_file("fake", "main"))) == before


def test_build(monkeypatch, tmp_path):
    def responder(spec, path, prompt, opts):
        return dict(text=json.dumps(GOOD), returned_model="m", finish_reason="stop",
                    in_tokens=1, out_tokens=1)

    _fake_registry(monkeypatch, tmp_path, responder)
    photos = vp.list_photos()
    for i, p in enumerate(photos[:2], start=1):
        vp.collect_image("fake", vp.ALL_MODELS["fake"], "main", i, p, 3, "p", "h", None, 0)
    vp.build(10)
    import pandas as pd
    df = pd.read_csv(tmp_path / "runs_long.csv")
    assert len(df) == 2 * 3 * 20
    legacy = pd.read_csv(tmp_path / "fake.csv", index_col=0)
    assert legacy.shape == (20, 2) and (legacy.values == 3.0).all()
