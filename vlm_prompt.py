"""
VLM PANAS Prompting Script
==========================
Prompts each photo in photos/ with the PANAS prompt in figures/prompt.txt
across multiple VLMs and KEEPS EVERY RUN.

Outputs
-------
  data/runs/<model_key>.jsonl      one line per API call (raw text, parsed scores,
                                   returned model id, temperature, tokens, ...)
  data/runs/<model_key>__<tag>.jsonl  same, for a tagged condition (e.g. t0 control)
  data/runs/model_config.json      per-model settings found by --probe
  data/runs/prompt.txt             prompt actually sent (sha256 stored in each line)
  data/runs_long.csv               model, tag, image, run, item, score   (--build)
  data/<model_key>.csv             legacy 20 items x 10 images means      (--build)

Workflow
--------
  python vlm_prompt.py --probe --models gpt-4.1          # 1 call: detect rejected params
  python vlm_prompt.py --models gpt-4.1 --runs 10        # collect (resumable)
  python vlm_prompt.py --models gpt-4.1 --runs 10 --tag t0 --temperature 0
  python vlm_prompt.py --build                           # runs_long.csv + legacy means
  python vlm_prompt.py --dry-run                         # show remaining calls only

Each (model, tag, image) is drawn until `--runs` VALID responses exist (all 20
items numeric, in 1-5) or MAX_ATTEMPTS_FACTOR * runs calls were made. Failed or
invalid calls are logged, never silently averaged.

Environment variables
---------------------
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPINFRA_API_KEY
"""

import argparse
import base64
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE per line). Never overrides variables that
    are already set in the environment. .env is gitignored."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)


load_dotenv()
PHOTOS_DIR = ROOT / "photos"
PROMPT_FILE = ROOT / "figures" / "prompt.txt"
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
CONFIG_FILE = RUNS_DIR / "model_config.json"
RUNS_LONG = DATA_DIR / "runs_long.csv"

PANAS_ITEMS = [
    "interested", "excited", "strong", "enthusiastic", "proud",
    "alert", "inspired", "determined", "attentive", "active",
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid",
]

DEFAULT_TEMPERATURE = 0.7
MAX_ATTEMPTS_FACTOR = 1.5      # at most 1.5 * runs calls per image
MAX_RETRIES = 3                # transient-error retries per call


# ── Model registry ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Model:
    provider: str                 # openai | anthropic | google | deepinfra
    model_id: str
    max_tokens: int = 1024        # large for models that think before answering
    temperature: float | None = DEFAULT_TEMPERATURE   # None = never send
    note: str = ""


def _m(provider, model_id, max_tokens=1024, **kw):
    return Model(provider, model_id, max_tokens, **kw)


ALL_MODELS: dict[str, Model] = {
    # ── OpenAI: GPT-4.1 ──────────────────────────────────────────────────
    "gpt-4.1":        _m("openai", "gpt-4.1"),
    "gpt-4.1-mini":   _m("openai", "gpt-4.1-mini"),
    "gpt-4.1-nano":   _m("openai", "gpt-4.1-nano"),
    # ── OpenAI: GPT-5.x (reasoning) ──────────────────────────────────────
    "gpt-5.2":        _m("openai", "gpt-5.2", 4096),
    "gpt-5.4":        _m("openai", "gpt-5.4", 4096),
    "gpt-5.4-mini":   _m("openai", "gpt-5.4-mini", 4096),
    "gpt-5.4-nano":   _m("openai", "gpt-5.4-nano", 4096),
    "gpt-5.5":        _m("openai", "gpt-5.5", 4096),
    "gpt-5.6-sol":    _m("openai", "gpt-5.6-sol", 4096),
    "gpt-5.6-terra":  _m("openai", "gpt-5.6-terra", 4096),
    "gpt-5.6-luna":   _m("openai", "gpt-5.6-luna", 4096),
    # ── OpenAI: GPT-6 (verify IDs with --probe; Astra rejects temperature)
    "gpt-6-astra":    _m("openai", "gpt-6-astra", 4096, temperature=None,
                         note="documented: no custom temperature/top_p"),
    "gpt-6-sol":      _m("openai", "gpt-6-sol", 4096),
    "gpt-6.1-sol":    _m("openai", "gpt-6.1-sol", 4096),
    "gpt-6-luna":     _m("openai", "gpt-6-luna", 4096),
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-haiku-4-5":  _m("anthropic", "claude-haiku-4-5-20251001"),
    "claude-sonnet-4-6": _m("anthropic", "claude-sonnet-4-6"),
    "claude-opus-4-6":   _m("anthropic", "claude-opus-4-6"),
    "claude-haiku-5-5":  _m("anthropic", "claude-haiku-5-5", 4096),
    "claude-sonnet-5-5": _m("anthropic", "claude-sonnet-5-5", 4096),
    "claude-opus-5-5":   _m("anthropic", "claude-opus-5-5", 4096),
    # "claude-fable-5-1":  _m("anthropic", "claude-fable-5-1", 8192),
    # ── Google ───────────────────────────────────────────────────────────
    "gemini-2.5-flash":      _m("google", "gemini-2.5-flash", 8192),
    "gemini-2.5-pro":        _m("google", "gemini-2.5-pro", 8192),
    "gemini-3.1-flash-lite": _m("google", "gemini-3.1-flash-lite", 8192),
    "gemini-3.1-pro":        _m("google", "gemini-3.1-pro-preview", 8192),
    "gemini-3.5-flash-lite": _m("google", "gemini-3.5-flash-lite", 8192),
    "gemini-3.5-flash":      _m("google", "gemini-3.5-flash", 8192),
    "gemini-3.6-flash":      _m("google", "gemini-3.6-flash", 8192),
    "gemini-3.7-flash":      _m("google", "gemini-3.7-flash", 8192),
    "gemini-3.8-flash":      _m("google", "gemini-3.8-flash", 8192),
    # listed by the API; confirm it is callable (earlier sources said restricted)
    "gemini-4-argon":        _m("google", "gemini-4-argon", 8192),
    # ── DeepInfra (OpenAI-compatible) ────────────────────────────────────
    "llama-4-scout":          _m("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    "qwen3-vl-235b":          _m("deepinfra", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
    # Replacements for Llama-3.2 / Llama-4-Maverick (retired on DeepInfra);
    # vision input to be confirmed by --probe.
    "qwen3-vl-30b-a3b":       _m("deepinfra", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "qwen3.5-397b-a17b":      _m("deepinfra", "Qwen/Qwen3.5-397B-A17B", 4096),
    "qwen3.8-27b":            _m("deepinfra", "Qwen/Qwen3.8-27B", 4096),
    "mistral-small-3.2-24b":  _m("deepinfra", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    "kimi-k3":                _m("deepinfra", "moonshotai/Kimi-K3", 4096),
    "kimi-k2.6":              _m("deepinfra", "moonshotai/Kimi-K2.6", 4096),
    "gemma-4-31b":            _m("deepinfra", "google/gemma-4-31B-it"),
    "gemma-4-26b-a4b":        _m("deepinfra", "google/gemma-4-26B-A4B-it"),
    "glm-5.3-flash":          _m("deepinfra", "zai-org/GLM-5.3-Flash", 4096),
    "mimo-v2.6-pro":          _m("deepinfra", "XiaomiMiMo/MiMo-V2.6-Pro", 4096),
    "mimo-v2.6-flash":        _m("deepinfra", "XiaomiMiMo/MiMo-V2.6-Flash", 4096),
    "deepseek-v4.1-flash":    _m("deepinfra", "deepseek-ai/DeepSeek-V4.1-Flash", 4096),
}

# Models from the original submission that can still be re-run.
LEGACY_MODELS = [
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5.2", "gpt-5.4",
    "gpt-5.4-mini", "gpt-5.4-nano", "claude-haiku-4-5", "claude-sonnet-4-6",
    "claude-opus-4-6", "gemini-2.5-flash", "gemini-2.5-pro",
    "llama-4-scout", "qwen3-vl-235b",
]
# Retired on DeepInfra before the re-run (old means kept in data/legacy_2026-03):
RETIRED_MODELS = ["llama-3.2-11b-vision", "llama-3.2-90b-vision",
                  "llama-4-maverick-fp8", "llama-4-maverick-turbo"]


# ── Helpers ────────────────────────────────────────────────────────────────
def load_prompt() -> str:
    """Prompt text exactly as stored (wording unchanged); parser tolerates its
    trailing comma, so the prompt is NOT modified."""
    return PROMPT_FILE.read_text().strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def list_photos() -> list[Path]:
    photos = sorted(PHOTOS_DIR.glob("*.jpg"),
                    key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    if not photos:
        raise FileNotFoundError(f"No .jpg files found in {PHOTOS_DIR}")
    return photos


def image_info(path: Path) -> dict:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    data = path.read_bytes()
    return {"file": path.name, "sha256": sha256_bytes(data),
            "width": w, "height": h, "bytes": len(data)}


def parse_response(raw: str) -> dict:
    """Extract the JSON object from a model response string."""
    raw = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("no JSON object in response")
    body = re.sub(r",\s*\}", "}", m.group(0))
    return json.loads(body)


def validate_scores(parsed: dict) -> tuple[dict | None, str]:
    """Return ({item: float}, '') if all 20 items are numeric in 1-5."""
    lower = {str(k).lower(): v for k, v in parsed.items()}
    out = {}
    for item in PANAS_ITEMS:
        if item not in lower:
            return None, f"missing item '{item}'"
        try:
            val = float(lower[item])
        except (TypeError, ValueError):
            return None, f"non-numeric '{item}'"
        if not 1.0 <= val <= 5.0:
            return None, f"out of range '{item}'={val}"
        out[item] = val
    return out, ""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Provider calls ─────────────────────────────────────────────────────────
# Each returns dict(text, returned_model, finish_reason, in_tokens, out_tokens).
# `opts` holds optional sampling params ("temperature", "seed"); a call may be
# retried by `call_with_fallback` with rejected params removed.
def call_openai_compatible(client, spec: Model, image_path: Path, prompt: str,
                           opts: dict) -> dict:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    kwargs = dict(
        model=spec.model_id,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}]}],
        max_completion_tokens=spec.max_tokens,
    )
    kwargs.update(opts)
    r = client.chat.completions.create(**kwargs)
    choice = r.choices[0]
    return dict(
        text=choice.message.content or "",
        returned_model=getattr(r, "model", None),
        finish_reason=choice.finish_reason,
        in_tokens=getattr(r.usage, "prompt_tokens", None),
        out_tokens=getattr(r.usage, "completion_tokens", None),
    )


def call_openai(spec, image_path, prompt, opts):
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return call_openai_compatible(client, spec, image_path, prompt, opts)


def call_deepinfra(spec, image_path, prompt, opts):
    import openai
    client = openai.OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"],
                           base_url="https://api.deepinfra.com/v1/openai")
    return call_openai_compatible(client, spec, image_path, prompt, opts)


def call_anthropic(spec, image_path, prompt, opts):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    kwargs = dict(
        model=spec.model_id,
        max_tokens=spec.max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                          "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt}]}],
    )
    # The SDK no longer exposes sampling params; pass temperature in the raw body.
    # Models that reject it (Claude 4.7+/5.x) fall back via call_with_fallback.
    if "temperature" in opts:
        kwargs["extra_body"] = {"temperature": opts["temperature"]}
    r = client.messages.create(**kwargs)
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return dict(text=text, returned_model=r.model, finish_reason=r.stop_reason,
                in_tokens=r.usage.input_tokens, out_tokens=r.usage.output_tokens)


def call_google(spec, image_path, prompt, opts):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    cfg = dict(max_output_tokens=spec.max_tokens)
    cfg.update(opts)
    r = client.models.generate_content(
        model=spec.model_id,
        contents=[types.Part.from_bytes(data=image_path.read_bytes(),
                                        mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(**cfg),
    )
    um = r.usage_metadata
    cand = r.candidates[0] if r.candidates else None
    return dict(text=r.text or "",
                returned_model=getattr(r, "model_version", None),
                finish_reason=str(cand.finish_reason) if cand else None,
                in_tokens=getattr(um, "prompt_token_count", None),
                out_tokens=getattr(um, "candidates_token_count", None))


PROVIDER_DISPATCH = {"openai": call_openai, "anthropic": call_anthropic,
                     "google": call_google, "deepinfra": call_deepinfra}

# Optional sampling params that may be rejected by some models.
OPTIONAL_PARAMS = ("temperature", "seed")

_state_lock = threading.Lock()
_dropped_params: dict[str, set] = {}      # model_key -> params the API rejected


def call_with_fallback(model_key: str, spec: Model, image_path: Path,
                       prompt: str, opts: dict) -> tuple[dict, dict]:
    """Call the provider; if the API rejects an optional param, drop it (and
    remember that for this model). Returns (result, params_actually_sent)."""
    fn = PROVIDER_DISPATCH[spec.provider]
    with _state_lock:
        dropped = set(_dropped_params.get(model_key, set()))
    sent = {k: v for k, v in opts.items() if k not in dropped}
    for _ in range(len(OPTIONAL_PARAMS) + 1):
        try:
            return fn(spec, image_path, prompt, sent), sent
        except Exception as exc:                     # noqa: BLE001
            msg = str(exc).lower()
            rejected = [p for p in OPTIONAL_PARAMS if p in sent and p in msg
                        and any(w in msg for w in
                                ("unsupported", "not support", "invalid",
                                 "deprecated", "does not", "only", "not allowed",
                                 "400"))]
            if not rejected:
                raise
            for p in rejected:
                sent.pop(p)
            with _state_lock:
                _dropped_params.setdefault(model_key, set()).update(rejected)
    raise RuntimeError("parameter fallback exhausted")


# ── Run log (append-only JSONL) ────────────────────────────────────────────
_write_lock = threading.Lock()


def run_file(model_key: str, tag: str) -> Path:
    name = model_key if tag == "main" else f"{model_key}__{tag}"
    return RUNS_DIR / f"{name}.jsonl"


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_log(path: Path, rec: dict) -> None:
    with _write_lock, path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def attempt_call(model_key, spec, photo, prompt, opts):
    """One API call with transient-error retries. Returns (result, sent, error)."""
    err = None
    for n in range(MAX_RETRIES):
        try:
            res, sent = call_with_fallback(model_key, spec, photo, prompt, opts)
            return res, sent, None
        except Exception as exc:                     # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** (n + 1))
    return None, opts, err


def log_record(model_key, spec, tag, img_idx, info, call_idx, run, status,
               res, sent, error, latency, prompt_hash, reason=""):
    return {
        "model_key": model_key, "provider": spec.provider,
        "model_id": spec.model_id,
        "returned_model": (res or {}).get("returned_model"),
        "tag": tag, "image": img_idx, "image_file": info["file"],
        "image_sha256": info["sha256"],
        "image_px": [info["width"], info["height"]],
        "call_idx": call_idx, "run": run, "status": status, "reason": reason,
        "timestamp": now_utc(),
        "temperature_requested": sent.get("temperature_requested"),
        "temperature_applied": sent.get("temperature"),
        "seed": sent.get("seed"), "max_tokens": spec.max_tokens,
        "prompt_sha256": prompt_hash,
        "raw_text": (res or {}).get("text"),
        "finish_reason": (res or {}).get("finish_reason"),
        "in_tokens": (res or {}).get("in_tokens"),
        "out_tokens": (res or {}).get("out_tokens"),
        "latency_s": round(latency, 2), "error": error,
    }


def collect_image(model_key, spec, tag, img_idx, photo, runs, prompt,
                  prompt_hash, temperature, delay):
    path = run_file(model_key, tag)
    info = image_info(photo)
    prior = [r for r in read_log(path) if r["image"] == img_idx]
    n_ok = sum(1 for r in prior if r["status"] == "ok")
    call_idx = max((r["call_idx"] for r in prior), default=0)
    max_calls = int(runs * MAX_ATTEMPTS_FACTOR)      # budget per invocation
    session_calls = 0
    while n_ok < runs and session_calls < max_calls:
        call_idx += 1
        session_calls += 1
        opts = {}
        if temperature is not None:
            opts["temperature"] = temperature
        opts["seed"] = 1000 * img_idx + call_idx      # distinct per call
        t0 = time.time()
        res, sent, error = attempt_call(model_key, spec, photo, prompt, opts)
        sent = dict(sent)
        sent["temperature_requested"] = temperature
        status, reason, scores = "error", error or "", None
        if res is not None:
            try:
                scores, reason = validate_scores(parse_response(res["text"]))
                status = "ok" if scores else "invalid"
            except Exception as exc:                 # noqa: BLE001
                status, reason = "invalid", f"parse: {exc}"
        run = None
        if status == "ok":
            n_ok += 1
            run = n_ok
        rec = log_record(model_key, spec, tag, img_idx, info, call_idx, run,
                         status, res, sent, error, time.time() - t0,
                         prompt_hash, reason)
        rec["scores"] = scores
        append_log(path, rec)
        if status != "ok":
            print(f"    [{status}] {model_key} img{img_idx} call{call_idx}: {reason[:120]}")
        if delay:
            time.sleep(delay)
    if n_ok < runs:
        print(f"  [short] {model_key} img{img_idx}: {n_ok}/{runs} valid runs")
    return n_ok


def remaining_calls(model_key, tag, runs, photos):
    log = read_log(run_file(model_key, tag))
    out = {}
    for i in range(1, len(photos) + 1):
        ok = sum(1 for r in log if r["image"] == i and r["status"] == "ok")
        out[i] = max(0, runs - ok)
    return out


def snapshot_prompt() -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt()
    (RUNS_DIR / "prompt.txt").write_text(prompt)
    return prompt


def collect_model(model_key, runs, tag, temperature, workers, delay):
    spec = ALL_MODELS[model_key]
    prompt = snapshot_prompt()
    prompt_hash = sha256_bytes(prompt.encode())
    photos = list_photos()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(collect_image, model_key, spec, tag, i, p, runs,
                          prompt, prompt_hash, temperature, delay)
                for i, p in enumerate(photos, start=1)]
        counts = [f.result() for f in futs]
    print(f"  {model_key} [{tag}]: valid runs per image = {counts}")


# ── Probe ──────────────────────────────────────────────────────────────────
def probe_model(model_key, temperature) -> dict:
    """One call on image 1 to see which sampling params the API accepts."""
    spec = ALL_MODELS[model_key]
    prompt = snapshot_prompt()
    photo = list_photos()[0]
    opts = {"seed": 1}
    if temperature is not None:
        opts["temperature"] = temperature
    cfg = {"provider": spec.provider, "model_id": spec.model_id,
           "max_tokens": spec.max_tokens,
           "temperature_requested": temperature, "probed_at": now_utc()}
    t0 = time.time()
    res, sent, error = attempt_call(model_key, spec, photo, prompt, opts)
    if res is None:
        cfg.update(status="unavailable", error=error)
    else:
        scores, reason = (None, "")
        try:
            scores, reason = validate_scores(parse_response(res["text"]))
        except Exception as exc:                     # noqa: BLE001
            reason = f"parse: {exc}"
        cfg.update(status="ok" if scores else "invalid", error=reason or None,
                   returned_model=res["returned_model"],
                   finish_reason=res["finish_reason"],
                   out_tokens=res["out_tokens"],
                   temperature_applied=sent.get("temperature"),
                   seed_applied=sent.get("seed"),
                   latency_s=round(time.time() - t0, 2))
    return cfg


def update_config(model_key, cfg):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        allc = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
        allc[model_key] = cfg
        CONFIG_FILE.write_text(json.dumps(allc, indent=2))


# ── Build derived tables ───────────────────────────────────────────────────
def build(runs: int, models: list[str] | None = None) -> None:
    """runs_long.csv from all logs (first `runs` valid runs per image) and the
    legacy mean CSV per model (main tag only)."""
    rows, summary = [], []
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        for r in read_log(path):
            if r["status"] != "ok" or r["run"] is None or r["run"] > runs:
                continue
            if models and r["model_key"] not in models:
                continue
            for item, val in r["scores"].items():
                rows.append((r["model_key"], r["tag"], r["image"], r["run"],
                             item, val))
    if not rows:
        print("No valid runs found.")
        return
    df = pd.DataFrame(rows, columns=["model", "tag", "image", "run", "item", "score"])
    df.to_csv(RUNS_LONG, index=False)
    print(f"Wrote {RUNS_LONG} ({len(df)} rows)")
    main = df[df["tag"] == "main"]
    for model, g in main.groupby("model"):
        n_runs = g.groupby("image")["run"].nunique()
        piv = (g.groupby(["item", "image"])["score"].mean().unstack("image")
                 .reindex(PANAS_ITEMS))
        piv.columns = [f"image {i}" for i in piv.columns]
        piv.index.name = ""
        piv.to_csv(DATA_DIR / f"{model}.csv")
        summary.append((model, int(n_runs.min()), int(n_runs.max()), g["image"].nunique()))
    print("model, min_runs, max_runs, n_images")
    for s in summary:
        print("  ", *s)


# ── Verification ───────────────────────────────────────────────────────────
def verify_model(model_key: str, tag: str, runs: int) -> dict:
    """Check the log on disk: exactly `runs` valid runs (numbered 1..runs, unique)
    for every image, scores parseable, and how many runs are distinct."""
    n_img = len(list_photos())
    log = read_log(run_file(model_key, tag))
    problems, distinct = [], []
    for i in range(1, n_img + 1):
        ok = [r for r in log if r["image"] == i and r["status"] == "ok"
              and r["run"] is not None and r["run"] <= runs]
        nums = sorted(r["run"] for r in ok)
        if nums != list(range(1, runs + 1)):
            problems.append(f"image {i}: run numbers {nums}")
            continue
        vecs = {tuple(r["scores"][it] for it in PANAS_ITEMS) for r in ok}
        distinct.append(len(vecs))
    statuses = {}
    for r in log:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    returned = sorted({r["returned_model"] for r in log if r.get("returned_model")})
    temps = sorted({str(r["temperature_applied"]) for r in log})
    return {"model": model_key, "tag": tag, "log_lines": len(log),
            "statuses": statuses, "complete": not problems, "problems": problems,
            "distinct_runs_per_image_min": min(distinct) if distinct else 0,
            "returned_models": returned, "temperature_applied": temps,
            "out_tokens_total": sum(r.get("out_tokens") or 0 for r in log)}


# ── Auth check ─────────────────────────────────────────────────────────────
KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY", "deepinfra": "DEEPINFRA_API_KEY"}


def list_provider_models(provider: str) -> set[str]:
    """Authenticated, token-free call: ids of the models the key can see."""
    if provider == "openai":
        import openai
        c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return {m.id for m in c.models.list()}
    if provider == "deepinfra":
        import openai
        c = openai.OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"],
                          base_url="https://api.deepinfra.com/v1/openai")
        return {m.id for m in c.models.list()}
    if provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return {m.id for m in c.models.list(limit=1000)}
    if provider == "google":
        from google import genai
        c = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        return {m.name.removeprefix("models/") for m in c.models.list()}
    raise ValueError(provider)


def check_auth(models: list[str]) -> None:
    """Report per provider whether the key authenticates (never prints keys),
    and which registry model IDs the account can see."""
    for provider, var in KEY_VARS.items():
        keyed = [m for m in models if ALL_MODELS[m].provider == provider]
        if not keyed:
            continue
        if not os.environ.get(var):
            print(f"[{provider}] {var} NOT SET")
            continue
        try:
            ids = list_provider_models(provider)
        except Exception as exc:                     # noqa: BLE001
            print(f"[{provider}] AUTH FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        print(f"[{provider}] auth OK ({len(ids)} models visible)")
        for m in keyed:
            seen = ALL_MODELS[m].model_id in ids
            print(f"    {'found  ' if seen else 'MISSING'} {m:26s} {ALL_MODELS[m].model_id}")


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PANAS VLM prompting (keeps every run)")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                    choices=list(ALL_MODELS))
    ap.add_argument("--legacy-only", action="store_true",
                    help="restrict to the 17 models of the original analysis")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tag", default="main",
                    help="condition label; non-main tags go to <model>__<tag>.jsonl")
    ap.add_argument("--temperature", type=float, default=None,
                    help=f"override (default {DEFAULT_TEMPERATURE} where the model allows)")
    ap.add_argument("--workers", type=int, default=3,
                    help="images processed in parallel per model")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between calls within a worker")
    ap.add_argument("--check-auth", action="store_true",
                    help="verify keys + list which registry model IDs are visible (no tokens used)")
    ap.add_argument("--verify", action="store_true",
                    help="check logs on disk: 10 valid, numbered, distinct runs per image")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--reprobe", action="store_true",
                    help="probe again even if model_config.json already has status ok")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = [m for m in args.models if not args.legacy_only or m in LEGACY_MODELS]

    if args.check_auth:
        check_auth(models)
        return

    if args.verify:
        bad = 0
        for mk in models:
            v = verify_model(mk, args.tag, args.runs)
            flag = "OK  " if v["complete"] else "FAIL"
            bad += not v["complete"]
            print(f"{flag} {mk:26s} lines={v['log_lines']:4d} status={v['statuses']} "
                  f"distinct/img(min)={v['distinct_runs_per_image_min']} "
                  f"T={v['temperature_applied']} model={v['returned_models']}")
            for pr in v["problems"]:
                print("      ", pr)
        raise SystemExit(1 if bad else 0)

    if args.build:
        build(args.runs, None)
        return

    photos = list_photos()
    print(f"Photos: {len(photos)}  Models: {len(models)}  Runs/image: {args.runs}  Tag: {args.tag}")

    if args.probe and not args.dry_run:
        done = (json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {})
        todo = [m for m in models if args.reprobe or done.get(m, {}).get("status") != "ok"]
        print(f"Probing {len(todo)} models (parallel)")

        def _probe(mk):
            spec = ALL_MODELS[mk]
            temp = args.temperature if args.temperature is not None else spec.temperature
            try:
                cfg = probe_model(mk, temp)
            except Exception as exc:                 # noqa: BLE001
                cfg = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
            update_config(mk, cfg)
            print(f"  {mk:26s} {cfg['status']:11s} returned={cfg.get('returned_model')} "
                  f"T_applied={cfg.get('temperature_applied')} err={(cfg.get('error') or '')[:90]}",
                  flush=True)

        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(_probe, todo))
        return

    for model_key in models:
        spec = ALL_MODELS[model_key]
        temp = args.temperature if args.temperature is not None else spec.temperature
        if args.dry_run:
            rem = sum(remaining_calls(model_key, args.tag, args.runs, photos).values())
            print(f"  {model_key:26s} T={temp}  calls still needed >= {rem}")
            continue
        print(f"=== {model_key} ({spec.model_id}) T={temp} ===")
        try:
            if args.probe:
                cfg = probe_model(model_key, temp)
                update_config(model_key, cfg)
                print(f"  probe: {cfg['status']}  returned={cfg.get('returned_model')}  "
                      f"T_applied={cfg.get('temperature_applied')}  err={cfg.get('error')}")
            else:
                collect_model(model_key, args.runs, args.tag, temp,
                              args.workers, args.delay)
        except Exception as exc:                     # noqa: BLE001
            print(f"  [FAILED] {model_key}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
