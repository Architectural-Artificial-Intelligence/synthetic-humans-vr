"""
VLM PANAS Prompting Script
==========================
Prompts each photo in photos/ with the PANAS prompt in figures/prompt.txt
across multiple VLMs. Results are saved as data/<model_id>.csv in the same
format as data/gpt4.csv (20 PANAS items × 10 images, averaged over N runs).

Supported providers
-------------------
  openai   : gpt-4o, gpt-4.1, gpt-4o-mini, gpt-4.1-mini, gpt-4.1-nano,
             gpt-5.1-codex-mini, gpt-5.2, gpt-5.2-chat-latest, gpt-5.2-codex,
             gpt-5.3-codex, gpt-5-mini, gpt-5-nano,
             gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, o4-mini
  anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001
  google   : gemini-2.5-pro, gemini-2.5-flash
  deepinfra: meta-llama/Llama-3.2-11B-Vision-Instruct,
             meta-llama/Llama-3.2-90B-Vision-Instruct,
             meta-llama/Llama-4-Scout-17B-16E-Instruct,
             meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8,
             meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo,
             Qwen/Qwen2.5-VL-72B-Instruct,
             moonshotai/Kimi-K2.5,
             Qwen/Qwen3-VL-235B-A22B-Instruct

Usage
-----
  python vlm_prompt.py                        # run all models
  python vlm_prompt.py --models gpt-4o claude-sonnet-4-6
  python vlm_prompt.py --models gemini-1.5-pro --runs 5

Environment variables
---------------------
  OPENAI_API_KEY      – required for OpenAI models
  ANTHROPIC_API_KEY   – required for Anthropic models
  GOOGLE_API_KEY      – required for Google models
  DEEPINFRA_API_KEY   – required for DeepInfra models
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PHOTOS_DIR = ROOT / "photos"
PROMPT_FILE = ROOT / "figures" / "prompt.txt"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PANAS_ITEMS = [
    "interested", "excited", "strong", "enthusiastic", "proud",
    "alert", "inspired", "determined", "attentive", "active",
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid",
]

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry: (provider, model_id)
ALL_MODELS = {
    # ── OpenAI GPT-4 family ──────────────────────────────────────────────
    "gpt-4.1":               ("openai",    "gpt-4.1"),
    "gpt-4.1-mini":          ("openai",    "gpt-4.1-mini"),
    "gpt-4.1-nano":          ("openai",    "gpt-4.1-nano"),
    # ── OpenAI GPT-5 family ──────────────────────────────────────────────
    "gpt-5.4":               ("openai",    "gpt-5.4"),
    "gpt-5.4-mini":          ("openai",    "gpt-5.4-mini"),
    "gpt-5.4-nano":          ("openai",    "gpt-5.4-nano"),
    "gpt-5-mini":            ("openai",    "gpt-5-mini"),
    "gpt-5-nano":            ("openai",    "gpt-5-nano"),
    "gpt-5.2":               ("openai",    "gpt-5.2"),
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-opus-4-6":       ("anthropic", "claude-opus-4-6"),
    "claude-sonnet-4-6":     ("anthropic", "claude-sonnet-4-6"),
    "claude-haiku-4-5":      ("anthropic", "claude-haiku-4-5-20251001"),
    # ── Google ───────────────────────────────────────────────────────────
    "gemini-2.5-pro":        ("google",    "gemini-2.5-pro"),
    "gemini-2.5-flash":      ("google",    "gemini-2.5-flash"),
    # ── DeepInfra (OpenAI-compatible API) ───────────────────────────────
    "llama-3.2-11b-vision": ("deepinfra", "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "llama-3.2-90b-vision": ("deepinfra", "meta-llama/Llama-3.2-90B-Vision-Instruct"),
    "llama-4-scout":        ("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    "llama-4-maverick-fp8": ("deepinfra", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
    "llama-4-maverick-turbo": ("deepinfra", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo"),
    "qwen2.5-vl-72b":       ("deepinfra", "Qwen/Qwen2.5-VL-72B-Instruct"),
    "kimi-k2.5":            ("deepinfra", "moonshotai/Kimi-K2.5"),
    "qwen3-vl-235b":        ("deepinfra", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
    "NVIDIA-Nemotron-Nano-12B-v2-VL": ("deepinfra", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL"),
    "GLM-4.6Vision":          ("deepinfra", "zai-org/GLM-4.6V"),
}


# ── Helpers ────────────────────────────────────────────────────────────────
def load_prompt() -> str:
    text = PROMPT_FILE.read_text().strip()
    # Fix trailing comma inside JSON (makes it invalid) – remove it
    text = re.sub(r",\s*\}", "}", text)
    return text


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_response(raw: str) -> dict:
    """Extract the JSON object from a model response string."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Remove trailing commas before closing brace
    raw = re.sub(r",\s*\}", "}", raw)
    return json.loads(raw)


# ── Provider call functions ────────────────────────────────────────────────
def call_openai(model_id: str, image_path: Path, prompt: str) -> dict:
    import openai  # pip install openai>=1.0

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = encode_image(image_path)
    # Some newer models (gpt-5.x) use max_completion_tokens; fall back to max_tokens
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_completion_tokens=512,
            temperature=0.7,
        )
    except openai.BadRequestError as e:
        if "max_completion_tokens" in str(e):
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.7,
            )
        else:
            raise
    return parse_response(response.choices[0].message.content)


def call_anthropic(model_id: str, image_path: Path, prompt: str) -> dict:
    import anthropic  # pip install anthropic>=0.25

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64 = encode_image(image_path)
    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return parse_response(response.content[0].text)


def call_google(model_id: str, image_path: Path, prompt: str) -> dict:
    import google.generativeai as genai  # pip install google-generativeai
    import PIL.Image

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_id)
    image = PIL.Image.open(image_path)
    response = model.generate_content([image, prompt])
    return parse_response(response.text)


def call_deepinfra(model_id: str, image_path: Path, prompt: str) -> dict:
    import openai  # pip install openai>=1.0

    client = openai.OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=512,
        temperature=0.7,
    )
    return parse_response(response.choices[0].message.content)


PROVIDER_DISPATCH = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
    "deepinfra": call_deepinfra,
}


# ── Core logic ─────────────────────────────────────────────────────────────
def prompt_model(
    model_key: str,
    runs: int = 10,
    delay: float = 1.0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run the PANAS prompt for every photo, `runs` times each, and return
    a DataFrame with shape (20 items × 10 images) of averaged scores.
    """
    provider, model_id = ALL_MODELS[model_key]
    call_fn = PROVIDER_DISPATCH[provider]
    prompt = load_prompt()

    photos = sorted(PHOTOS_DIR.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not photos:
        raise FileNotFoundError(f"No .jpg files found in {PHOTOS_DIR}")

    out_path = DATA_DIR / f"{model_key}.csv"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists (use --overwrite to re-run)")
        return pd.read_csv(out_path, index_col=0)

    # accumulator: item → {image_label: [scores]}
    accum: dict[str, dict[str, list[float]]] = {
        item: {} for item in PANAS_ITEMS
    }

    for img_idx, photo in enumerate(photos, start=1):
        img_label = f"image {img_idx}"
        for item in PANAS_ITEMS:
            accum[item][img_label] = []

        print(f"  {model_key} | {img_label} ({photo.name})")
        for run in range(1, runs + 1):
            try:
                result = call_fn(model_id, photo, prompt)
                for item in PANAS_ITEMS:
                    key = next(
                        (k for k in result if k.lower() == item.lower()), None
                    )
                    if key is not None:
                        accum[item][img_label].append(float(result[key]))
                    else:
                        print(f"    [warn] missing item '{item}' in run {run}")
            except Exception as exc:
                print(f"    [error] run {run}: {exc}")
            if delay and run < runs:
                time.sleep(delay)

    # Average across runs
    records = {}
    for item in PANAS_ITEMS:
        records[item] = {}
        for img_label in [f"image {i}" for i in range(1, len(photos) + 1)]:
            scores = accum[item].get(img_label, [])
            records[item][img_label] = round(sum(scores) / len(scores), 2) if scores else float("nan")

    df = pd.DataFrame(records).T  # shape: items × images
    df.index.name = ""
    df.to_csv(out_path)
    print(f"  Saved → {out_path}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PANAS VLM prompting script")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_MODELS.keys()),
        choices=list(ALL_MODELS.keys()),
        help="Which models to run (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs per image (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if output CSV already exists",
    )
    args = parser.parse_args()

    prompt = load_prompt()
    print(f"Prompt loaded ({len(prompt)} chars)")
    print(f"Photos: {len(list(PHOTOS_DIR.glob('*.jpg')))} images")
    print(f"Models: {args.models}\n")

    for model_key in args.models:
        print(f"=== {model_key} ===")
        try:
            prompt_model(model_key, runs=args.runs, delay=args.delay, overwrite=args.overwrite)
        except Exception as exc:
            print(f"  [FAILED] {model_key}: {exc}")
        print()


if __name__ == "__main__":
    main()
