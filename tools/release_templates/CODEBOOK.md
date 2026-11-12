# Codebook

## data/human.csv (84 rows = participants, 206 columns, UTF-8, Hebrew headers)
| Column(s) | Meaning |
|---|---|
| 1 `participant_id` | Random-order label P001–P084 (replaces survey timestamp; order = collection order). |
| 2 consent | "מסכים/ה" (agree) to participate (age ≥18, normal/corrected vision). |
| 3 `שנת לידה` | Birth year (age = 2025 − birth year). |
| 4 `מין` | Sex: `נקבה` female (55), `זכר` male (29). |
| 5 `רמת השכלה:` | Education: `סטודנט` student, `תואר בראשון` BA, `תואר שני` MA, `דוקטורט` PhD, `12 שנות לימוד` high school. |
| 6 `מקצוע:` | Field, coarsened: `אדריכלות` architecture, `הנדסה` engineering, `אחר` other, empty = not reported. |
| 7–206 | 200 PANAS cells = 10 scenes × 20 items. Header text is the Hebrew instruction plus `[עד כמה החלל … ]` (item). Scene index = numeric suffix `.k` of the header (no suffix = scene 1, `.1` = scene 2 …, `.9` = scene 10). Values are Hebrew Likert labels: כלל לא=1, מעט=2, במידה בינונית=3, במידה רבה=4, רבה מאוד=5. Empty = missing. |

The Hebrew-item → English mapping (`HEBREW_ITEM_MAP`) and parsing are in `analysis/panas_lib.py` (`load_human_raw`, `human_array`).
The 20 items (PANAS order): interested, excited, strong, enthusiastic, proud, alert, inspired, determined, attentive, active (PA);
distressed, upset, guilty, scared, hostile, irritable, ashamed, nervous, jittery, afraid (NA).

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
