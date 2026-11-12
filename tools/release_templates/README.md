# Human and VLM PANAS ratings of ten immersive urban environments — data and code

Dataset and analysis code for *Dortheimer et al. (2026)*, "Human and VLM PANAS ratings of ten
immersive urban environments" (replace with final title/DOI of the paper).
84 participants rated ten 360° urban scenes in VR with the 20-item PANAS (Hebrew version);
45 vision-language models (43 analysed) rated the same panoramas ten times each with the same items.

## Contents
| Path | What |
|---|---|
| `data/human.csv` | De-identified human responses (84 x 10 scenes x 20 items, wide, Hebrew headers). See `CODEBOOK.md`. |
| `data/runs/*.jsonl` | **Raw log of every VLM API call** (raw text, parsed scores, returned model version, temperature, seed, tokens, timestamp, image/prompt SHA-256). `__t0` files = temperature-0 control. |
| `data/runs/model_config.json`, `prompt.txt`, `logs/` | Per-model probe settings, the exact prompt, collection logs. |
| `data/runs_long.csv` | Tidy run-level ratings (model, tag, image, run, item, score), built from the jsonl by `python vlm_prompt.py --build`. |
| `data/<model>.csv` | Per-model 20 items x 10 images run means (derived). |
| `photos/` | The ten panoramas (equirectangular), **faces/plates blurred, EXIF/GPS removed**. The VLMs saw the unblurred originals; their SHA-256 are in `photos/SHA256_originals.txt` and in each jsonl record. |
| `analysis/` | Analysis pipeline (see below). `analysis/scene_info.csv` = scene names, view directions, blur boxes. |
| `vlm_prompt.py` | VLM collector / dataset builder (re-collection needs your own API keys; analysis does not). |
| `results/20261101_103808/` | **Final analysis batch** (43 models): tables T1–T10, figures F1–F12 and Fig 1–6, S1–S3, REPORT.md. |
| `results/gender_20261108_103819/` | Gender analysis; Fig S4, Tables T11–T12 are written into the batch folder. |
| `manuscript/` | Methods (stimuli/procedure) and gender-supplement text. |
| `tests/` | pytest unit tests. |
| `MANIFEST_SHA256.txt` | Checksums of every file. |

## Reproduce
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # tested with Python 3.14; versions pinned in requirements.lock.txt
bash reproduce.sh                    # re-creates tables/figures from the data (minutes; bootstrap B=2000)
pytest tests
```
`reproduce.sh` writes to `reproduced/` and does not overwrite the shipped results.
Pipeline order: `robust_inference.py` → `stability.py` → `paired_baseline.py` → `demographics_table.py`
→ `make_paper_outputs.py` → `figure_stats.py` → `make_main_figures.py` → `gender_analysis.py` → `gender_supplement.py`.
Random seeds are fixed in the scripts (bootstrap seed 2026), so results should match the shipped ones
up to library-version differences.

## Notes and known differences
* **De-identification.** The original export's timestamp was replaced by `P001…` (collection order),
  occupation was coarsened to architecture / engineering / other, and an empty export column dropped.
  Birth year, sex and education are kept. `analysis/demographics_table.py` therefore
  gives coarser occupation rows than the paper's Table T7 (T7 in `results/` is the original).
  All PANAS-based results are unaffected (verified: identical participant x scene x item array).
* **Excluded models.** `gemini-3.5-flash` and `gemini-3.7-flash` are aliases of `…3.6…`/`…3.8…`; raw logs are kept,
  analyses exclude them (`panas_lib.ALIAS_DUPLICATES`).
* **Old study.** The earlier 17-model analysis is not part of this record.
* Code was developed with AI coding assistants (Claude, ChatGPT) and checked by the authors.

## Licence
Data (`data/`, `photos/`, `results/`): CC BY 4.0 — `LICENSE-DATA`. Code (`*.py`, `*.sh`): MIT — `LICENSE-CODE`.
Please cite the Zenodo record (`CITATION.cff`) and the paper.
Ethics: approved by the university ethics committee (approval no. TODO); informed consent obtained.
