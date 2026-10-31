# VLMs as predictors of felt affect in the built environment: code and results

Source code and results for the study comparing 43 vision language models (VLMs) with the PANAS affect that 84 participants reported while viewing ten urban environments in immersive VR.

This branch holds only the latest code (VLM querying and statistical analysis) and the final results. The panoramas and the raw participant responses are not included (see [Data not included](#data-not-included)).

## Contents

| Path | What it is |
|---|---|
| `vlm_prompt.py` | Queries each VLM with the ten panoramas and the PANAS prompt, keeps every call, validates responses, and builds the run-level table |
| `collect_all.sh`, `collect_queue.sh` | Run the collection for all models in parallel, with verification and retries |
| `figures/prompt.txt` | The prompt sent to every model (verbatim) |
| `analysis/panas_lib.py` | Shared loading and metrics: ICC(2,1)/(2,k), MAE, RMSE, within-item correlation, no-image baseline, RSA, Bland–Altman checks |
| `analysis/robust_inference.py` | Agreement with humans, participant/run and image bootstrap intervals, rank intervals, leave-one-environment-out, bias tests, noise ceiling, reliability |
| `analysis/stability.py` | Run-to-run stability, convergence over runs, temperature-0 control |
| `analysis/paired_baseline.py` | Paired image bootstrap of each model against the no-image baseline |
| `tests/` | Unit tests (`pytest`) |
| `data/runs/*.jsonl` | Every VLM call: raw response, parsed scores, returned model version, temperature, seed, tokens, timestamp (`__t0` = temperature-0 control) |
| `data/runs/model_config.json`, `data/runs/prompt.txt` | Per-model settings and the prompt snapshot (its SHA-256 is stored in every call) |
| `data/runs_long.csv` | All valid ratings: model, condition, image, run, item, score (98,000 rows) |
| `results/20261030_v2/` | Final results (below) |

### Results (`results/20261030_v2/`)

| File | Content |
|---|---|
| `human_cell_means.csv` | Human mean and SD for each of the 200 image × item cells (N = 84) |
| `summary/model_comparison.csv` | All agreement indices per model with participant+run and image bootstrap intervals and rank intervals |
| `summary/paired_baseline_image_bootstrap.csv` | Each model's ICC margin over the no-image baseline with paired image-bootstrap intervals |
| `summary/bias_tests.csv` | PA, NA and overall offsets per model (participant-level t-tests, Holm and FDR corrected) |
| `summary/loo_image_metrics.csv` | All indices with each environment left out in turn |
| `summary/bland_altman_checks.csv` | Normality and proportional-bias checks of the differences |
| `summary/individual_level_slopes.csv` | Individual-level regressions of participant ratings on model ratings |
| `summary/robustness_summary.json` | No-image baseline, noise ceiling, split-half reliability, top-5 selection check, leave-one-out ranking stability |
| `summary/reliability_hebrew_panas.json` | Cronbach's α and McDonald's ω for PA and NA |
| `stability/` | Inter-run reliability, convergence by number of runs, ranking concordance, temperature-0 comparison |

## Notes on the data

- **43 distinct models.** The logs contain 45 model keys. `gemini-3.5-flash` and `gemini-3.7-flash` were served by the API as `gemini-3.6-flash` and `gemini-3.8-flash` (see `returned_model`), with identical outputs, and are excluded by the analysis (`ALIAS_DUPLICATES` in `analysis/panas_lib.py`).
- **Item mapping.** Hebrew PANAS items are matched to English items following Ben-Zur (2002). Set `PANAS_ITEM_MAP=legacy` to reproduce the earlier mapping used as a sensitivity check.
- **Collection date.** All main-condition calls were made on 13 October 2026. Providers update models behind unchanged names, so a new collection may differ.

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests            # data-dependent tests skip when inputs are absent

# rebuild the run-level table from the call logs (reproduces data/runs_long.csv exactly)
python vlm_prompt.py --build

# analysis (requires data/human.csv, see below)
cd analysis
python stability.py        --out ../results/<batch>/stability
python robust_inference.py --B 2000 --mixed --out ../results/<batch>/summary
python paired_baseline.py  --B 2000 --out ../results/<batch>/summary
```

New collection requires the ten panoramas in `photos/` (`1.jpg` … `10.jpg`) and API keys in the environment or a `.env` file (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPINFRA_API_KEY`):

```bash
python vlm_prompt.py --check-auth
python vlm_prompt.py --models gemini-4-argon --runs 10
python vlm_prompt.py --verify
python vlm_prompt.py --build
```

## Data not included

- **Panoramas** (`photos/`): available from the authors on request; they will be released with location metadata removed.
- **Individual participant responses** (`data/human.csv`): not public. The aggregate cell means are in `results/20261030_v2/human_cell_means.csv`. The individual-level responses, which the bootstrap and participant-level analyses require, are available from the corresponding author on reasonable request.
