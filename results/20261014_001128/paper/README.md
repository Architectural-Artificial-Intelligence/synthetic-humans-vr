# Paper outputs (batch 20261014_001128)

Regenerate: `python analysis/make_paper_outputs.py --batch results/20261014_001128`
Unit of analysis: 200 image x item cells (human = mean of 84 participants, model = mean of 10 runs).
CIs: participant + run bootstrap, B = 2000, seed 2026. ICC = ICC(2,1)/(2,k), two-way random, absolute agreement.

| File | Content | Reviewer comments |
|---|---|---|
| T1_model_reproducibility | requested/returned model ID, temperature, max tokens, calls, tokens, dates | R1.3, R3.2, R4.3, R5 |
| T2_agreement_main | ICC(2,1)/(2,k), rho, MAE, RMSE, normalised errors, bias, rank intervals, with CIs | R1.2, R1.4, R3.3, R4.2, R4.5 |
| T3_rsa_distance_metrics | RSA under correlation / Euclidean / cosine distance | R3.3b |
| T4_bias_tests | participant-level PA/NA/overall bias, Holm-adjusted | R3.3c, R3.5 |
| T5_run_stability | within-cell SD, inter-run ICC, split-half, single-run vs mean-of-10 | R1.3, R3.2, R4.4, R5 |
| T6_human_sample_reliability | N accounting, missing data, Cronbach alpha / McDonald omega | R1.5, R1.6, R1.7, R5 |
| T7_demographics | participant demographics | R4.7 |
| T8_robustness_summary | leave-one-image-out, noise ceiling, top-5 selection check | R4.2, R5 |
| T9_noise_ceiling_by_model | model vs individual participants | R5 |
| T10_temperature_control | T=0.7 vs T=0 on 4 models | R3.2 |
| F1_forest_icc | ICC with CIs, 45 models | R1.2 |
| F2_rank_intervals | ranking uncertainty | R5 |
| F3_scatter_bland_altman | item-level scatter + Bland-Altman | R1.4 |
| F4_rsa_rdm | RDMs and RSA | R3.3b |
| F5_pa_na_profiles | PA/NA by environment | - |
| F6_bias_pa_na | PA vs NA offsets | R3.5 |
| F7_run_stability | single-run vs mean-of-10; inter-run ICC | R1.3, R4.4 |
| F8_convergence | agreement vs number of runs | R3.2 |
| F9_leave_one_image_out | leave-one-environment-out sensitivity | R4.2 |
| F10_noise_ceiling | model vs individual participants | R5 |
| F11_hebrew_panas_alpha | alpha per environment | R1.5 |
| F12_temperature_control | T=0 vs T=0.7 | R3.2 |
