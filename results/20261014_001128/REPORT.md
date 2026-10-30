# VR-PANAS-VLM revision: analysis report and response to reviewers

Batch `20261014_001128` · generated 2026-10-14 · all numbers reproducible from `data/runs/*.jsonl` and `data/human.csv`

---

## 1. Summary

The review raised five recurring problems: (1) too little information to reproduce the VLM side, (2) only run-averaged VLM outputs, so no stability evidence, (3) vague statistics (ICC form, units, normalisation, RSA metric, CIs, multiple comparisons), (4) only 10 environments and 10 image-level means behind each correlation, and (5) conclusions broader than the evidence. We could not add scenes, so we rebuilt the evidence base instead:

| What changed | Before | Now |
|---|---|---|
| VLM data kept | one mean per item × image | **every call** (4,928 logged; 4,900 valid), raw text, parsed scores, returned model version, temperature, seed, tokens, timestamps |
| Models | 17 | **45** (all re-run; 14 of the original 17 still exist, 4 Llama models were retired by the provider) |
| Runs per image | averaged, not retained | **10 independent runs × 10 images × 45 models**, each verified to have exactly 10 valid, uniquely numbered runs per image |
| Uncertainty | bootstrap over 200 non-independent cells | **participant-level (cluster) bootstrap, B = 2000, plus run resampling**, rank intervals, leave-one-environment-out |
| Stability | none | within-cell SD, inter-run ICC, single-run vs mean-of-10, convergence curve, variance decomposition, T = 0 control, re-collection 7 months later |
| Stats reporting | ICC form and units implicit | ICC(2,1) and ICC(2,k), unit, MAE/RMSE normalisation, three RSA distance metrics, Bland-Altman assumption checks, Holm correction where tests are made |

**Main result under the stronger analysis.** Agreement is real but graded and model-dependent. ICC(2,1) with the human cell means ranges 0.55–0.87 (median 0.79); the median Spearman ρ is 0.85, and 38 of 45 models have ρ ≥ 0.80. Models track the *ordering* of human affect better than its *level*: 41 of 45 models rate negative affect significantly lower than humans (mean offset 0.38 rating points). Run-to-run noise is small (median inter-run ICC 0.978). No model predicts an individual participant as well as the rest of the sample does. These results support "correspondence/alignment at the aggregate level", and they do not support "reproduction" or substitution for individual raters.

---

## 2. What was done

### 2.1 VLM data collection (`vlm_prompt.py`)
- **Models (45):** OpenAI 15, Anthropic 6, Google 10, open-weight via DeepInfra 14. Every model was probed once with the real image before collection; all accepted image input and returned valid scores. IDs, the version string returned by each API, applied temperature and token limits are in Table T1 and `data/runs/model_config.json`.
- **Prompt:** unchanged from the original study (`figures/prompt.txt`, snapshot and SHA-256 in `data/runs/prompt.txt` and every log line). It asks for the *average* rating of 20 typical people for each of the 20 PANAS items as JSON. No structured-output or JSON-mode constraint was used; responses were parsed by a tolerant parser and validated (all 20 items, numeric, 1–5).
- **Image input:** the ten JPEGs, sent unmodified as base64 (3584 × 1792 px, 2:1 aspect ratio, about 2.3 MB each). Providers may downscale internally; this is not under our control.
- **Sampling:** temperature 0.7 where the API accepts it (34 models). Eleven models reject custom temperature and ran at the provider default (GPT-5.5, GPT-5.6 ×3, GPT-6 ×4, Claude Haiku/Sonnet/Opus 5.5). A distinct seed per call (1000 × image + call index) was sent where supported; Anthropic has no seed parameter.
- **Runs:** 10 valid runs per image per model. Each (model, image) was drawn until 10 valid runs existed; failures are logged, never averaged around. 28 of 4,928 calls failed (0.6%), all API errors from one model (GLM-5.3-Flash), all replaced by retries. No response was rejected as invalid.
- **Each model was run as its own call** and verified on disk afterwards (`python vlm_prompt.py --verify`): 45/45 models complete.
- **Excluded by choice:** `claude-fable-5-1` (cost).

### 2.2 Human data
84 participants × 10 environments × 20 PANAS items (Hebrew). Units used throughout: **primary = 200 image × item cells** (human = mean of 84 participants, model = mean of 10 runs); secondary = PA/NA subscale × image (10 points each); participant level for the mixed models and noise ceiling.

### 2.3 Analyses
| Script | Purpose |
|---|---|
| `analysis/panas_lib.py` | shared loading and metrics (ICC(2,1), ICC(2,k), r, ρ, MAE, RMSE, normalised errors, Bland-Altman, RSA) |
| `analysis/robust_inference.py` | participant + run bootstrap CIs, rank intervals, leave-one-image-out, noise ceiling, top-5 selection check, Holm/FDR bias tests, individual-level slopes (OLS participant-clustered and mixed models), Cronbach α / McDonald ω, imputation sensitivity, sample accounting, Bland-Altman checks |
| `analysis/stability.py` | within-cell variability, single-run vs mean-of-10, inter-run reliability, convergence, image-order stability, variance decomposition, T = 0 comparison, ranking concordance |
| `analysis/make_paper_outputs.py` | all 10 tables and 12 figures in `paper/` |
| `analysis/panas_analysis.py` | original per-model analysis (LME, CFA, Wilcoxon, Bayes factors); fixed (see §5) and re-run for all 45 models |

Statistical definitions (to be stated in the Methods):
- **ICC:** two-way random effects, absolute agreement; targets = 200 cells; two "raters" = human mean and model mean. We report ICC(2,1) (single measures) and ICC(2,k) with k = 2. ICC(2,1) is the headline because it penalises absolute offsets.
- **MAE / RMSE:** in rating points, plus normalised versions: ÷ 4 (scale range) and ÷ SD of all human ratings (1.32).
- **RSA:** Spearman correlation between the 45 pairwise image dissimilarities of human and model profiles; reported under correlation, Euclidean and cosine distance.
- **CIs:** percentile intervals from 2,000 replicates that resample participants (and, for the "both" interval, each model's runs).
- **Bland-Altman:** bias and 95% limits of agreement over the 200 cells, with assumption checks (see §4).

---

## 3. Results

### 3.1 Agreement with humans (Table T2, Figure F1)
Top 10 of 45 (full table in `paper/tables/T2_agreement_main`):

| Rank | 95% rank interval | Model | ICC(2,1) [95% CI] | Spearman ρ [95% CI] | RMSE [95% CI] | Bias (human − model) |
|---:|:--|:--|:--|:--|:--|--:|
| 1 | 1–2 | gemini-4-argon | 0.865 [0.821, 0.888] | 0.893 [0.865, 0.905] | 0.433 [0.388, 0.512] | 0.23 |
| 2 | 1–9 | gemma-4-26b-a4b | 0.846 [0.805, 0.865] | 0.886 [0.860, 0.895] | 0.435 [0.399, 0.507] | 0.18 |
| 3 | 2–6 | gpt-5.4 | 0.843 [0.801, 0.866] | 0.869 [0.839, 0.882] | 0.466 [0.426, 0.542] | 0.22 |
| 4 | 3–10 | gpt-5.6-terra | 0.834 [0.786, 0.861] | 0.871 [0.840, 0.886] | 0.467 [0.417, 0.546] | 0.24 |
| 5 | 3–9 | claude-opus-4-6 | 0.833 [0.787, 0.860] | 0.888 [0.859, 0.900] | 0.477 [0.425, 0.556] | 0.27 |
| 6 | 3–13 | kimi-k3 | 0.832 [0.783, 0.860] | 0.896 [0.867, 0.904] | 0.465 [0.415, 0.547] | 0.28 |
| 7 | 3–12 | gpt-4.1-mini | 0.829 [0.784, 0.854] | 0.866 [0.832, 0.881] | 0.502 [0.459, 0.578] | 0.22 |
| 8 | 4–15 | gemma-4-31b | 0.827 [0.786, 0.848] | 0.878 [0.855, 0.882] | 0.466 [0.425, 0.539] | 0.24 |
| 9 | 1–26 | qwen3-vl-235b | 0.827 [0.786, 0.845] | 0.856 [0.827, 0.872] | 0.474 [0.448, 0.536] | −0.20 |
| 10 | 5–13 | gpt-6-astra | 0.823 [0.775, 0.853] | 0.855 [0.823, 0.874] | 0.496 [0.444, 0.575] | 0.25 |

Bottom: gemini-2.5-flash 0.570, gemini-3.5-flash-lite 0.567, qwen3.8-27b 0.555 (all have ρ ≥ 0.76; their low ICC comes from a large positive offset of 0.54–0.63 points).

- 30 of 45 models have ICC ≥ 0.75; none is below 0.50. Median ICC(2,k) is 0.88, median MAE 0.42, median normalised RMSE (÷ range) 0.14.
- Median RSA is 0.865 (correlation), 0.857 (Euclidean), 0.874 (cosine), minimum 0.61: the conclusions do not depend on the distance metric (T3, F4).
- Only the first place is reasonably firm; ranks 2–12 overlap heavily (F2). The 95% rank interval of qwen3-vl-235b spans 1–26 because its bias has the opposite sign to the others.
- By provider (median ICC): open-weight 0.805, OpenAI 0.795, Anthropic 0.768, Google 0.759. Newer is not systematically better: Claude 5.5 models are below their 4.x counterparts (e.g. Sonnet 0.79 → 0.65), as are GPT-6 vs GPT-5.4 (0.84 → 0.69/0.82), while Gemini improved (2.5-Flash 0.57 → 3.8-Flash 0.76). The 11 models that do not accept a custom temperature have a lower median ICC (0.744 vs 0.793).

### 3.2 Systematic offsets (Table T4, Figure F6)
Participant-level bias tests (N = 84, Holm-adjusted across models): humans give higher negative-affect ratings than the VLMs for 41 of 45 models (mean offset 0.38 points), and higher positive-affect ratings for 30 of 45 (mean 0.18). VLMs compress the negative end. This is why ICC(2,1) is lower than Spearman ρ for most models.

### 3.3 Run-to-run stability (Table T5, Figures F7, F8, F12)
- Median inter-run ICC(2,1) = 0.978 (range 0.87–0.997); least stable: gemini-2.5-flash 0.870, mimo-v2.6-flash 0.895, qwen3.8-27b 0.897; most stable: claude-opus-4-6 0.997, gpt-6-astra 0.996, gpt-6.1-sol 0.995.
- Median within-cell SD across runs = 0.10 rating points. Variance decomposition (median over models): item 50%, image × item 45%, image 1.5%, run/residual 2.2%.
- A single run is already close to the 10-run mean (median SD of single-run ICC 0.015). The median gain from averaging 10 vs 1 run is 0.007 ICC (max 0.037); about 90% of that gain is reached with 5 runs. Kendall's W across single-run model rankings = 0.92.
- Image ordering is preserved across runs (median Kendall τ 0.82 for PA, 0.82 for NA; worst model 0.64 / 0.44).
- **T = 0 control** (4 models, 10 runs/image, Table T10): T = 0 is *not* deterministic for 3 of 4 models. Mean cell SD fell from 0.11–0.23 (T = 0.7) to 0.003–0.12; the share of cells with zero SD rose from 1–16% to 97% (Claude Haiku 4.5), 40.5% (GPT-4.1 mini), 39.5% (Gemini 2.5 Flash), 35.5% (Llama 4 Scout). Agreement with humans barely changes (ICC 0.545–0.810 at T = 0 vs 0.570–0.829).
- **Re-collection 7 months later:** 14 models were run in both the original collection (Mar 2026, as reported: averaged over runs) and now. ICC values correlate r = 0.98 across models; median change −0.001, mean absolute change 0.021. The exceptions are Gemini 2.5 Flash (+0.106) and GPT-5.4-nano (+0.051), which may reflect provider-side model updates behind an unpinned alias.

### 3.4 Robustness to the 10 environments and to participant sampling (Table T8, Figures F9, F10)
- **Leave-one-environment-out:** model ranking vs full ranking, Kendall τ min 0.87, mean 0.92 (F9 shows each model's ICC range when any single environment is removed).
- **Human reliability:** split-half (Spearman-Brown) 0.985 [0.979, 0.989]: the 84 participants determine the 200 cell means very precisely.
- **Top-5 selection bias:** choosing the top-5 models by ICC on a random half of participants and evaluating them on the other half loses only 0.008 ICC (0.839 → 0.831). But the identity of the top-5 is unstable: two random halves share on average 2.98 of 5 models.
- **Noise ceiling (individual level):** an average participant correlates r = 0.656 with the mean of the others (0.665 with the mean including themselves). The mean correlation of each model's cell means with individual participants is below that for **all 45 models**. At the individual level the models are therefore weaker predictors than the group mean is. Individual-level OLS slopes (participant-clustered) range 0.67–1.36 (mixed models 0.68–1.38), median R² 0.28.

### 3.5 Human sample and the Hebrew PANAS (Table T6, Figure F11)
Cronbach α (pooled over images) = 0.850 [0.823, 0.872] for PA and 0.930 [0.919, 0.940] for NA; McDonald ω = 0.841 and 0.928. Per image, α ranges 0.66–0.88 (PA) and 0.79–0.91 (NA). The current data file has 84 rows, none blank, **0% missing PANAS cells**, one duplicated timestamp.

---

## 4. How each review item was addressed

Status key: **Done** = analysis or documentation produced here · **Text** = requires a manuscript change (no new analysis possible or needed) · **Author input** = needs information only the study team has · **Open** = not done.

### Reviewer 1 (Reject)

| # | Comment | Status | What we did / what remains |
|---|---|---|---|
| 1 | VR vs 2D modality; give presentation format, FOV, VLM inputs | **Text** + partly **Done** | VLM inputs are now fully documented (§2.1: ten 3584 × 1792 JPEGs, unmodified, base64, per-call hash and pixel size logged). The modality difference cannot be removed by analysis; per your decision, address it in wording and limitations. VR presentation details (headset, FOV, freedom to look around, exposure time) need **author input**. |
| 2 | Only 10 environments; ranking 17 models; give CIs for ICC, ρ, MAE, RMSE, RSA | **Done** (CIs) / cannot add scenes | CIs for all metrics (T2, T3, F1) from participant + run bootstrap. Ranking uncertainty shown as rank intervals (F2) and leave-one-environment-out (F9, τ ≥ 0.87). Models increased from 17 to 45. The number of environments stays 10; state it as a limitation. |
| 3 | Averaging hides run variability; give SD across runs and run-level ICC | **Done** | All 10 runs kept. T5/F7/F8: within-cell SD, per-run agreement with humans (mean, SD, range), inter-run ICC(2,1)/(2,k), α, split-half, convergence, variance decomposition. |
| 4 | Unit of analysis (subscales vs 20 items), exact ICC form, normalisation | **Done** (documented) / **Text** | Primary unit = 200 item × image cells; subscale (PA/NA) results are secondary. ICC(2,1) and ICC(2,k) both reported; MAE/RMSE normalised by range (4) and by human SD. Put the definitions of §2.3 in the Methods. |
| 5 | Hebrew PANAS source/validation; internal consistency | **Done** (α, ω) / **Author input** | α = 0.85 (PA), 0.93 (NA); ω = 0.84, 0.93 (T6, F11). The origin and validation of the Hebrew translation (and any back-translation) must come from the authors. |
| 6 | 84 recruited and 84 analysed | **Author input** / **Text** | The data file contains 84 rows with no blank records. The repository manuscript text (`paper.md`) says 85 recruited, 1 excluded, N = 84; the reviewed text apparently said 84 recruited. The excluded record is not in the file, so the exclusion cannot be verified from data. Reconcile the counts against the raw survey export. |
| 7 | Imputation of 1.7% missing; sensitivity analysis | **Done**, with a discrepancy | The current `human.csv` has 0% missing cells, so no imputation occurs and complete-case results are identical (`imputation_sensitivity.json`). Mean imputation cannot change any item × image mean anyway; it would only affect participant-level variance. The manuscript's "1.7% imputed" claim conflicts with this file; check which data version the paper used. |
| 8 | Temper claims on city-scale mapping, design optimisation, population estimates | **Text** | Our evidence supports aggregate-level correspondence in 10 environments. Reframe the applications as future work (see §6). |

### Reviewer 2 (minor revisions)

| Comment | Status | Notes |
|---|---|---|
| Highlight novelty vs literature | **Text** | Distinctive contributions now backed by data: 45 models, run-level stability for every model, participant-level uncertainty, individual-level noise ceiling, and a T = 0 control. |
| Practical/industrial significance | **Text** | Supported only as cautious, aggregate screening use; the noise ceiling and negative-affect offset are the limits to state. |
| Figures/tables labelled and discussed | **Done** (assets) / **Text** | 10 tables and 12 figures with consistent labels; `paper/README.md` maps each to the comments it answers. Numbering and in-text references must be synchronised in the manuscript. |
| Grammar/typos; abbreviations, symbols, units; references/format; conclusion | **Text / Open** | Not touched by the analysis. Reference list not yet checked (see R4.8). |

### Reviewer 3

| # | Comment | Status | What we did / what remains |
|---|---|---|---|
| 1 | Stimulus selection, documentation, Supplementary Table 1, VR presentation | **Author input** / partly **Done** | Technical image facts are recorded (all ten 3584 × 1792 JPEG, 2:1 aspect; hashes in logs). Content characteristics, selection rationale, real vs synthetic, viewpoint, perceptual distinctiveness, and VR protocol need the authors. |
| 2 | Reproduce the prompt; temperatures; why 10 runs; "simulate average" wording | **Done** / **Text** | Prompt is in `figures/prompt.txt` and `data/runs/prompt.txt` (hash logged): reproduce it in the main text. Temperatures in T1 (0.7 for 34 models; provider default for 11 that reject it). Stability evidence at T > 0: T5, F7. T = 0 control (T10, F12) shows T = 0 is not deterministic for 3 of 4 models; the convergence curve (F8) shows why 10 runs are more than enough (about 90% of the gain by 5 runs). The "simulate the average of 20 people" instruction elicits a group-level rating, not the model's own affect: discuss this in the text. |
| 3a | ICC(2,1) vs (2,k) | **Done** | Both reported; (2,1) is the headline because it is the stricter absolute-agreement index; k = 2 refers to the two sources (human mean, model mean). |
| 3b | RSA metric and robustness | **Done** | Correlation distance is primary; Euclidean and cosine reported (T3). Median ρ 0.865 / 0.857 / 0.874: not metric-dependent. |
| 3c | Multiple comparisons | **Done** (partly) | Holm (and FDR) adjustment applied across the 45 models for the bias tests (T4). The ICC/ρ/MAE intervals are descriptive, per-model CIs and are not adjusted. We replace a bare ranking by rank intervals and report that only first place is clearly separated. State this justification in the text. |
| 4 | Sample size justification / power | **Open** | No power analysis was done. Available evidence: split-half reliability 0.985 [0.979, 0.989] and the CIs above show the cell means are precise at N = 84. Recommend stating that it was a convenience sample (84 participants, 84.5% architecture-related occupations, mean age 26.7) and discussing generalisability. A precision-vs-N curve can be added on request. |
| 5 | "Reproduce" vs "correspondence" | **Text** | The data support the softer wording: rank order matches well (ρ median 0.85) but absolute levels do not (negative-affect offset in 41/45 models). Reserve "reproduce" for cases where both match. |
| 6 | Limitation: VR viewing vs real-world environments | **Text** | Add: no multi-sensory input, social context, temporal dynamics, or behavioural interaction; 10 curated scenes. |

### Reviewer 4

| # | Comment | Status | What we did / what remains |
|---|---|---|---|
| 1 | Distinguish VR for humans vs images for VLMs | **Text** | Per your decision, handled in wording. |
| 2 | CIs, sensitivity, leave-one-environment-out | **Done** | T2, F1, F2, F9, T8. |
| 3 | Full VLM setup: versions, prompt, temperature, sampling, resolution/format, API settings, structured output | **Done** (data) / **Text** | T1 + `model_config.json` + raw logs: requested and returned model IDs, temperature requested/applied, max tokens, per-call seed, image px and hash, finish reason, token counts, timestamps (calls made 2026-10-13). No structured-output mode was used (prompt-only JSON); 0 invalid responses, 28 API errors (one model) retried. Put the T1 summary in a supplement. |
| 4 | Run-to-run variability | **Done** | §3.3, T5, F7, F8. |
| 5 | Exact ICC, unit, MAE/RMSE normalisation, Bland-Altman assumptions, RSA metric, CI procedure | **Done** | §2.3. Bland-Altman checks (`summary/bland_altman_checks.csv`): differences are **non-normal for 43/45 models** (Shapiro p < .05) and show **proportional bias in 24/45** (slopes −0.13 to +0.54). Limits of agreement are therefore approximate; say so, and prefer MAE/RMSE as the primary error metrics. |
| 6 | Conclusions broader than evidence | **Text** | See §6. |
| 7 | Hebrew PANAS translation, recruitment, demographics, generalisability | **Done** (demographics, reliability) / **Author input** | Demographics table T7 (N = 84; age M = 26.7, SD = 7.2, range 18–58; 65.5% female; 84.5% architecture-related). Translation procedure and recruitment channel need the authors. |
| 8 | Figure numbering; missing references (Ma and Quan 2026; Yao et al. 2026) | **Open** | Not addressed by the analysis. The reference list and figure numbering must be checked in the manuscript; new figure/table numbers (F1–F12, T1–T10) must be reconciled with the old ones. |

### Reviewer 5

| # | Comment | Status | What we did / what remains |
|---|---|---|---|
| 1 | Only 10 environments; urban-scale claims | **Text** | Cannot add scenes; claims to be restricted (§6). |
| 2 | Modality mismatch | **Text** | See R1.1. |
| 3 | r = 0.98 from 10 means; effective N | **Done** (partly) | Intervals now come from participant-level resampling over the 84 × 200 ratings, and individual-level analyses use all 84 × 10 × 20 ratings (noise ceiling, mixed-model slopes). The number of *environments* is still 10; this limitation remains and should be stated. |
| 4 | VR experiment details (headset, exposure, freedom, order, randomisation, fatigue) | **Author input** | Cannot be derived from the data files. |
| 5 | Model reproducibility (API versions, temperature, seed, resolution, dates) | **Done** | T1, `model_config.json`, jsonl logs. Note that only some providers support seeds (OpenAI, Gemini, DeepInfra; not Anthropic) and seeds are best-effort. |
| 6 | Averaging removes variability | **Done** | §3.3. |
| 7 | Statistical procedures: units, bootstrap, Bland-Altman, normalisation | **Done** | §2.3; code is in `analysis/`, parameters saved in `summary/robustness_summary.json`. |
| 8 | Top-5 selected by ICC then analysed on the same data | **Done** (best available) | Repeated split-half selection (1,000×): optimism 0.008 ICC, but the top-5 set is unstable (mean overlap 2.98/5). This is not an independent validation set; the same 10 environments are used. Recommend not building conclusions on a "top-5" subset; use the full 45-model table and rank intervals. |
| 9 | Cultural validity / Hebrew PANAS | **Done** (α, ω) / **Author input** | See R1.5. |
| 10 | Conclusions exceed evidence | **Text** | See §6. |

---

## 5. Code fixes found along the way (for transparency)
1. `panas_analysis.py`: removed the dead, unpaired bootstrap block; labelled the ICC as ICC(2,1); added MAE/RMSE; the pair-bootstrap over cells now carries a warning that cells are not independent.
2. `panas_analysis.py`: **the "measurement invariance" section fitted the same single-group model three times**, so its configural/metric/scalar rows were identical. It is now marked "not performed"; **do not report invariance results from earlier outputs**.
3. `vlm_prompt.py` (original): default `--runs 1` despite a docstring saying 10; Anthropic and Google calls ran with unset temperature while OpenAI/DeepInfra used 0.7; no failure tracking; max tokens too small for reasoning models. All fixed in the rewrite.
4. The project `.venv` pointed at a Python version that no longer matched its packages; repaired.

---

## 6. Suggested wording changes

**Replace** "VLMs reproduce the relational geometry of human affective experience" with "VLM ratings show strong correspondence with the relational structure of human affective ratings across ten environments". **Replace** claims of "city-scale affective mapping, design optimisation, population-level estimation" with: "These results suggest that VLMs may eventually support rapid, low-cost screening of likely group-level affective responses to environments; testing this requires larger and more varied stimulus sets, other sensory modalities, and field validation."

**Limitations paragraph (draft):** Humans experienced immersive VR scenes, whereas VLMs received a single 2D image of each scene; the comparison is therefore between two different input modalities. Only ten curated environments were used, with 84 participants (a convenience sample, mostly architecture-related, mean age 26.7). Real environments additionally involve sound, movement, social context, and temporal dynamics. Agreement was assessed at the group level: no model predicted individual participants as well as the rest of the sample did, and nearly all models underestimated negative affect (mean offset 0.38 points). VLM outputs vary across runs and with provider-side model updates; we report run-level variability and re-collection stability for this reason.

**Methods paragraph (draft, VLM protocol):** "Each model rated each of the ten images ten independent times with an unchanged prompt (see Supplement), using the provider API with temperature 0.7 where permitted (11 models do not accept a custom temperature and ran at the provider default). Images were sent as 3584 × 1792 px JPEGs without resizing. We recorded every call, including the model version returned by the API, and kept only valid responses (20 numeric items in 1–5); 28 of 4,928 calls (0.6%) failed and were repeated."

---

## 7. Remaining items and caveats

**Need information from the authors**
- VR protocol (headset, FOV, viewing freedom, exposure duration, image order, randomisation, fatigue control).
- Stimulus documentation (selection rationale, real vs synthetic, viewpoint, content characteristics) for Supplementary Table 1.
- Hebrew PANAS source and validation; recruitment channel and procedure; confirmation of the 85/84 counts.
- Which data version the paper's "1.7% imputed" refers to.

**Not done**
- Power analysis (only precision-based evidence; a precision-vs-N curve is easy to add).
- Reference list and figure numbering check, including Ma and Quan (2026) and Yao et al. (2026).
- `claude-fable-5-1` not evaluated (cost); T = 0 control run on 4 of 45 models only.

**Caveats on the analyses**
- **Selection of models:** 4 originally analysed Llama models no longer exist; their old numbers cannot be re-run. 11 models ran at the provider default temperature. Some model IDs (e.g. `gemini-4-argon`, `gemini-3.1-pro-preview`) are newly released or previews and may change.
- **CIs:** percentile bootstrap intervals for a bounded index like ICC sit asymmetrically around the point estimate (the point lies near the upper end), because resampling adds noise to the human means. Consider BCa intervals if a reviewer asks.
- **Bland-Altman:** limits of agreement are approximate (non-normal differences, proportional bias).
- **Model drift:** results can change as providers update models; the re-collection check suggests this is small for most models but 2 of 14 moved by more than 0.05 ICC.
- **Environment sampling:** none of the new analyses can show that results generalise beyond these 10 scenes.

---

## 8. File map and reproduction

```
results/20261014_001128/
  REPORT.md                  this report
  summary/                   robust_inference outputs (model_comparison.csv, bias_tests.csv, loo_image_metrics.csv,
                             individual_level_slopes.csv, bland_altman_checks.csv, reliability_hebrew_panas.json,
                             imputation_sensitivity.json, sample_accounting.json, robustness_summary.json, forest_icc.png)
  stability/                 stability_by_model.csv, convergence_by_k.csv, model_rank_across_runs.csv,
                             ranking_concordance.json, temperature_control_t0_vs_main.csv
  paper/tables/ T1-T10       .csv / .tex / .md
  paper/figures/ F1-F12      .png / .pdf
  paper/README.md            table/figure to reviewer-comment map
results/20261014_001136/<model>/   per-model legacy analysis (45 models)
data/runs/<model>.jsonl            raw calls (+ __t0 control files); model_config.json; prompt.txt; logs/
data/runs_long.csv                 98,000 rows (model, tag, image, run, item, score)
data/legacy_2026-03/               original averaged results (superseded)
```

```bash
python vlm_prompt.py --check-auth                  # keys and model availability
python vlm_prompt.py --probe                       # one call per model
./collect_queue.sh                                 # all models, each as its own call, verified
python vlm_prompt.py --verify                      # 10 valid, numbered runs per image per model
python vlm_prompt.py --build                       # runs_long.csv + mean CSVs
python vlm_prompt.py --models gpt-4.1-mini --runs 10 --tag t0 --temperature 0   # T = 0 control
cd analysis
python stability.py --out ../results/<batch>/stability
python robust_inference.py --B 2000 --mixed --out ../results/<batch>/summary
python make_paper_outputs.py --batch ../results/<batch>
python -m pytest ../tests
```
