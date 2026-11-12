# Codebook

## data/human.csv (84 rows = participants, 206 columns, UTF-8, English keys and values)
| Column(s) | Meaning |
|---|---|
| `participant_id` | Random-order label P001–P084 (replaces the survey timestamp; order = collection order). |
| `consent` | `agree` = agreed to participate (age ≥18, normal/corrected vision). |
| `birth_year` | Birth year (age = 2025 − birth year). |
| `sex` | `female` (55), `male` (29). |
| `education` | `student` (40), `bachelor` (39), `master` (2), `high_school` (2), `doctorate` (1). |
| `field` | Field of study/occupation, coarsened: `architecture` (71), `engineering` (3), `other` (6); empty = not reported (4). |
| `scene<N>_<item>` (200 columns) | PANAS rating of item `<item>` (English PANAS item, see below) for scene `<N>` = 1…10 (same numbering as `photos/<N>.jpg`). Values: 1 = not at all, 2 = a little, 3 = moderately, 4 = quite a bit, 5 = very much. No missing values. |

(The survey was run in Hebrew; the original Hebrew labels were translated into the English codes above.)

The 20 items (PANAS order): interested, excited, strong, enthusiastic, proud, alert, inspired, determined, attentive, active (PA);
distressed, upset, guilty, scared, hostile, irritable, ashamed, nervous, jittery, afraid (NA).
Parsing is in `analysis/panas_lib.py` (`load_human_raw`, `human_array`).

## data/runs_long.csv
`model` model key · `tag` `main` (T=0.7 runs) or control tag · `image` 1–10 · `run` 1–10 · `item` PANAS item · `score` 1–5 (float).

## data/runs/<model>.jsonl (one JSON object per API call)
`model_key, provider, model_id, returned_model` (version string returned by the API) · `tag, image, image_file, image_sha256, image_px` ·
`call_idx, run, status, reason, timestamp` (UTC) · `temperature_requested, temperature_applied` (null = not settable) ·
`seed, max_tokens, prompt_sha256` · `raw_text` (verbatim response) · parsed scores and token counts (see `vlm_prompt.py`).

## data/<model>.csv
Rows = 20 items, columns `image 1 … image 10`, values = mean over the runs (`python vlm_prompt.py --build`).

## analysis/scene_info.csv
`scene` 1–10 · `short_name` · `yaw, pitch, fov` (deg; rectilinear view used for Fig 1) · `auto_detect` · `blur_boxes` (x0,y0,x1,y1 in 900×600 view pixels).

## results/
`REPORT.md` and `paper/README.md` in the batch folder list every table/figure; tables are CSV, figures PNG/PDF.
