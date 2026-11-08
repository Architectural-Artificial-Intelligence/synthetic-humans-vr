# Female vs male PANAS ratings of the VR scenes

Generated 2026-11-08 10:43 by `analysis/gender_analysis.py` (seed 2026, B = 2000 bootstraps, 5000 permutations). N = 84: 55 female, 29 male; 10 images x 20 items, no missing ratings. 43 VLMs (main batch).

Signs: differences are **female minus male**. Positive = women rate higher.

## 0. Confounds
| variable                | female                | male                  | test                                               |
|:------------------------|:----------------------|:----------------------|:---------------------------------------------------|
| age (years)             | 25.6 (6.5), median 24 | 28.7 (8.2), median 26 | Welch t = -1.76, p = 0.084; Mann-Whitney p = 0.002 |
| architecture occupation | 50/55 (91%)           | 21/29 (72%)           | Fisher exact OR = 3.81, p = 0.053                  |
| current student         | 25/55 (45%)           | 15/29 (52%)           | Fisher exact OR = 0.78, p = 0.649                  |

## 1. Primary test: person-level PA and NA means
One score per participant (mean over 10 images x 10 items), so the test unit is the person.

| scale   |   mean_F |   sd_F |   mean_M |   sd_M |   diff_F_minus_M |   welch_t |   welch_p |   welch_p_holm |   mwu_p |   perm_p |   hedges_g |   g_lo |   g_hi |
|:--------|---------:|-------:|---------:|-------:|-----------------:|----------:|----------:|---------------:|--------:|---------:|-----------:|-------:|-------:|
| PA      |    2.647 |  0.419 |    2.649 |  0.442 |           -0.001 |    -0.014 |     0.989 |          0.989 |   0.962 |    0.988 |     -0.003 | -0.464 |  0.458 |
| NA      |    2.046 |  0.471 |    1.810 |  0.349 |            0.236 |     2.601 |     0.011 |          0.023 |   0.022 |    0.018 |      0.540 |  0.132 |  0.970 |

Sensitivity: with 55 vs 29, a two-sided Welch-type test at alpha = .05 detects d >= 0.65 with 80% power (d >= 0.75 at 90%).

### 1b. Mixed-model and ordinal robustness checks (female coefficient)
Scale level, participant random PA/NA intercepts, image fixed, + age:
| scale   |   beta_female |    se |   ci_lo |   ci_hi |     p |
|:--------|--------------:|------:|--------:|--------:|------:|
| PA      |         0.034 | 0.099 |  -0.160 |   0.228 | 0.730 |
| NA      |         0.234 | 0.102 |   0.033 |   0.434 | 0.022 |

Same + architecture occupation:
| scale   |   beta_female |    se |   ci_lo |   ci_hi |     p |
|:--------|--------------:|------:|--------:|--------:|------:|
| PA      |         0.017 | 0.100 |  -0.179 |   0.213 | 0.864 |
| NA      |         0.215 | 0.103 |   0.013 |   0.418 | 0.037 |

Item level, 200 cell fixed effects, random participant + participant x image, + age:
| scale   |   beta_female |    se |   ci_lo |   ci_hi |     p |
|:--------|--------------:|------:|--------:|--------:|------:|
| PA      |         0.034 | 0.099 |  -0.160 |   0.228 | 0.730 |
| NA      |         0.234 | 0.102 |   0.033 |   0.434 | 0.022 |

Ordinal (proportional-odds) GEE, clustered by participant, + age, image, item (OR > 1 = women higher):
| scale   |   logOR_female |    se |    OR |   OR_lo |   OR_hi |     p |
|:--------|---------------:|------:|------:|--------:|--------:|------:|
| PA      |          0.055 | 0.146 | 1.056 |   0.793 |   1.407 | 0.707 |
| NA      |          0.515 | 0.201 | 1.673 |   1.128 |   2.483 | 0.011 |

## 2. Does the gender gap depend on the scene?
Split-plot ANOVA on PA / NA image scores (gender between, image within); Greenhouse-Geisser epsilon from the image factor applied to the within effects:
| scale   | effect         |       F |   df1 |   df2 |   p_unc |    p_GG |     eps |   np2 |
|:--------|:---------------|--------:|------:|------:|--------:|--------:|--------:|------:|
| PA      | gender         |   0.000 |     1 |    82 |   0.989 | nan     | nan     | 0.000 |
| PA      | image          |  74.198 |     9 |   738 |   0.000 |   0.000 |   0.627 | 0.475 |
| PA      | gender x image |   0.995 |     9 |   738 |   0.442 |   0.425 |   0.627 | 0.012 |
| NA      | gender         |   5.634 |     1 |    82 |   0.020 | nan     | nan     | 0.064 |
| NA      | image          | 134.221 |     9 |   738 |   0.000 |   0.000 |   0.663 | 0.621 |
| NA      | gender x image |   3.709 |     9 |   738 |   0.000 |   0.001 |   0.663 | 0.043 |

Per-image differences (Welch, BH-FDR within scale):
| scale   |   image |   mean_F |   mean_M |   diff |   hedges_g |     p |   p_fdr |
|:--------|--------:|---------:|---------:|-------:|-----------:|------:|--------:|
| PA      |       1 |    3.125 |    3.117 |  0.008 |      0.012 | 0.957 |   0.978 |
| PA      |       2 |    2.667 |    2.817 | -0.150 |     -0.199 | 0.408 |   0.718 |
| PA      |       3 |    2.196 |    2.200 | -0.004 |     -0.007 | 0.978 |   0.978 |
| PA      |       4 |    2.449 |    2.307 |  0.142 |      0.220 | 0.307 |   0.718 |
| PA      |       5 |    2.175 |    2.276 | -0.101 |     -0.173 | 0.422 |   0.718 |
| PA      |       6 |    3.078 |    3.186 | -0.108 |     -0.141 | 0.520 |   0.743 |
| PA      |       7 |    3.638 |    3.441 |  0.197 |      0.247 | 0.267 |   0.718 |
| PA      |       8 |    2.769 |    2.903 | -0.134 |     -0.185 | 0.431 |   0.718 |
| PA      |       9 |    2.064 |    2.041 |  0.022 |      0.037 | 0.870 |   0.978 |
| PA      |      10 |    2.311 |    2.197 |  0.114 |      0.209 | 0.318 |   0.718 |
| NA      |       1 |    1.605 |    1.486 |  0.119 |      0.182 | 0.343 |   0.503 |
| NA      |       2 |    1.735 |    1.566 |  0.169 |      0.235 | 0.276 |   0.503 |
| NA      |       3 |    2.755 |    2.431 |  0.324 |      0.411 | 0.070 |   0.176 |
| NA      |       4 |    1.598 |    1.466 |  0.133 |      0.205 | 0.352 |   0.503 |
| NA      |       5 |    2.722 |    2.410 |  0.311 |      0.406 | 0.054 |   0.176 |
| NA      |       6 |    1.369 |    1.334 |  0.035 |      0.064 | 0.779 |   0.902 |
| NA      |       7 |    1.331 |    1.310 |  0.021 |      0.051 | 0.812 |   0.902 |
| NA      |       8 |    1.300 |    1.297 |  0.003 |      0.009 | 0.968 |   0.968 |
| NA      |       9 |    3.307 |    2.597 |  0.711 |      0.866 | 0.000 |   0.002 |
| NA      |      10 |    2.742 |    2.207 |  0.535 |      0.655 | 0.002 |   0.012 |

## 3. Equivalence and Bayes factors
TOST with smallest effect of interest d = 0.5 and d = 0.3 (converted to rating points with the pooled SD). Equivalence is supported when tost_p < .05 (i.e. the 90% CI lies inside the bounds). JZS Bayes factor, Cauchy prior r = 0.707.

| scale   |   diff_ci90_lo |   diff_ci90_hi |   tost_bound_raw_d0.5 |   tost_p_d0.5 |   tost_bound_raw_d0.3 |   tost_p_d0.3 |   BF10 |   BF01 |
|:--------|---------------:|---------------:|----------------------:|--------------:|----------------------:|--------------:|-------:|-------:|
| PA      |         -0.168 |          0.165 |                 0.213 |         0.019 |                 0.128 |         0.104 |  0.238 |  4.209 |
| NA      |          0.085 |          0.387 |                 0.217 |         0.584 |                 0.130 |         0.877 |  2.604 |  0.384 |

## 4. Profile similarity (female vs male 10 x 20 mean profiles)
Null distribution: gender labels permuted (55/29 kept). For similarity metrics p is the share of random splits that are *less* similar than the real F/M split; a small p means the genders differ more than random halves of the sample.

| metric          |   observed_F_vs_M |   null_mean |   null_2.5 |   null_97.5 |   perm_p |
|:----------------|------------------:|------------:|-----------:|------------:|---------:|
| ICC2_1          |             0.919 |       0.960 |      0.935 |       0.974 |    0.004 |
| pearson         |             0.935 |       0.967 |      0.953 |       0.977 |    0.000 |
| spearman        |             0.935 |       0.960 |      0.946 |       0.971 |    0.002 |
| r_within_item   |             0.941 |       0.959 |      0.944 |       0.971 |    0.015 |
| RSA_correlation |             0.948 |       0.951 |      0.904 |       0.983 |    0.398 |
| MAE             |             0.249 |       0.185 |      0.148 |       0.242 |    0.018 |
| PA_bias         |            -0.001 |      -0.002 |     -0.187 |       0.186 |    0.987 |
| NA_bias         |             0.236 |      -0.001 |     -0.195 |       0.190 |    0.019 |

Cell-wise Welch screen (200 cells): 32 with p < .05 uncorrected (about 10 expected by chance), 9 significant after BH-FDR.

## 5. Do the VLMs agree more with women or with men?
Each model's run-mean profile vs the female and male mean profiles. Women are subsampled to n = 29 so both reference means have equal reliability; CIs come from a participant bootstrap of both groups (n = 29 each). For ICC / r a positive difference means the model agrees more with women; for MAE a positive difference means it is *further* from women.

Pooled over models:
| metric        |   mean_diff_F_minus_M |   boot_lo |   boot_hi |   boot_p |   n_models_diff_pos |   n_models |
|:--------------|----------------------:|----------:|----------:|---------:|--------------------:|-----------:|
| ICC2_1        |                -0.106 |    -0.171 |    -0.016 |    0.013 |                   0 |         43 |
| pearson       |                -0.079 |    -0.120 |    -0.017 |    0.013 |                   1 |         43 |
| r_within_item |                -0.041 |    -0.086 |     0.040 |    0.454 |                   1 |         43 |
| MAE           |                 0.129 |     0.020 |     0.216 |    0.022 |                  43 |         43 |
| PA_bias       |                -0.001 |    -0.216 |     0.219 |    0.993 |                   0 |         43 |
| NA_bias       |                 0.234 |     0.020 |     0.436 |    0.030 |                  43 |         43 |

Per model, ICC(2,1) (sorted):
| model                 |   vs_female_all55 |   vs_male |   diff_F_minus_M_equal_n |   boot_lo |   boot_hi |   boot_p |   p_holm |
|:----------------------|------------------:|----------:|-------------------------:|----------:|----------:|---------:|---------:|
| gpt-6-luna            |             0.609 |     0.778 |                   -0.175 |    -0.256 |    -0.063 |    0.001 |    0.022 |
| claude-sonnet-5-5     |             0.584 |     0.746 |                   -0.168 |    -0.253 |    -0.051 |    0.003 |    0.123 |
| gpt-6-sol             |             0.613 |     0.766 |                   -0.159 |    -0.244 |    -0.041 |    0.005 |    0.175 |
| claude-opus-5-5       |             0.677 |     0.822 |                   -0.151 |    -0.236 |    -0.033 |    0.009 |    0.261 |
| claude-haiku-5-5      |             0.639 |     0.783 |                   -0.150 |    -0.236 |    -0.035 |    0.008 |    0.240 |
| glm-5.3-flash         |             0.741 |     0.876 |                   -0.143 |    -0.211 |    -0.046 |    0.002 |    0.084 |
| gpt-5.4-nano          |             0.645 |     0.781 |                   -0.143 |    -0.218 |    -0.048 |    0.004 |    0.156 |
| gpt-5.2               |             0.725 |     0.855 |                   -0.139 |    -0.204 |    -0.044 |    0.003 |    0.123 |
| gpt-5.6-sol           |             0.733 |     0.857 |                   -0.132 |    -0.204 |    -0.033 |    0.006 |    0.198 |
| gpt-5.6-luna          |             0.671 |     0.797 |                   -0.132 |    -0.205 |    -0.038 |    0.005 |    0.175 |
| gpt-4.1               |             0.730 |     0.856 |                   -0.132 |    -0.204 |    -0.037 |    0.006 |    0.198 |
| mistral-small-3.2-24b |             0.736 |     0.858 |                   -0.131 |    -0.199 |    -0.038 |    0.004 |    0.156 |
| gpt-6.1-sol           |             0.757 |     0.873 |                   -0.125 |    -0.193 |    -0.031 |    0.004 |    0.156 |
| gpt-5.4-mini          |             0.749 |     0.864 |                   -0.124 |    -0.187 |    -0.033 |    0.004 |    0.156 |
| kimi-k2.6             |             0.625 |     0.742 |                   -0.123 |    -0.213 |    -0.007 |    0.038 |    0.740 |
| gpt-6-astra           |             0.760 |     0.874 |                   -0.123 |    -0.192 |    -0.027 |    0.007 |    0.217 |
| qwen3-vl-30b-a3b      |             0.729 |     0.840 |                   -0.118 |    -0.193 |    -0.022 |    0.014 |    0.392 |
| mimo-v2.6-pro         |             0.648 |     0.759 |                   -0.116 |    -0.201 |    -0.006 |    0.040 |    0.740 |
| gpt-5.5               |             0.734 |     0.842 |                   -0.116 |    -0.190 |    -0.012 |    0.023 |    0.575 |
| deepseek-v4.1-flash   |             0.729 |     0.835 |                   -0.114 |    -0.186 |    -0.018 |    0.022 |    0.572 |
| gemini-3.8-flash      |             0.706 |     0.811 |                   -0.112 |    -0.197 |    -0.002 |    0.047 |    0.799 |
| mimo-v2.6-flash       |             0.728 |     0.831 |                   -0.110 |    -0.185 |    -0.006 |    0.037 |    0.740 |
| gpt-5.6-terra         |             0.766 |     0.867 |                   -0.109 |    -0.174 |    -0.017 |    0.016 |    0.432 |
| claude-opus-4-6       |             0.778 |     0.875 |                   -0.105 |    -0.173 |    -0.014 |    0.025 |    0.576 |
| claude-sonnet-4-6     |             0.739 |     0.832 |                   -0.100 |    -0.180 |     0.005 |    0.068 |    1.000 |
| gpt-4.1-mini          |             0.775 |     0.867 |                   -0.099 |    -0.168 |    -0.011 |    0.024 |    0.576 |
| gemini-3.1-flash-lite |             0.687 |     0.777 |                   -0.098 |    -0.187 |     0.017 |    0.108 |    1.000 |
| gemini-3.5-flash-lite |             0.522 |     0.615 |                   -0.097 |    -0.188 |     0.018 |    0.103 |    1.000 |
| claude-haiku-4-5      |             0.753 |     0.840 |                   -0.095 |    -0.158 |    -0.011 |    0.030 |    0.630 |
| gemini-4-argon        |             0.810 |     0.896 |                   -0.095 |    -0.155 |    -0.011 |    0.025 |    0.576 |
| qwen3.8-27b           |             0.501 |     0.587 |                   -0.091 |    -0.179 |     0.020 |    0.112 |    1.000 |
| gemini-3.6-flash      |             0.750 |     0.828 |                   -0.086 |    -0.161 |     0.014 |    0.097 |    1.000 |
| gpt-5.4               |             0.792 |     0.869 |                   -0.085 |    -0.145 |    -0.001 |    0.048 |    0.799 |
| kimi-k3               |             0.781 |     0.856 |                   -0.085 |    -0.156 |     0.014 |    0.096 |    1.000 |
| gemini-3.1-pro        |             0.727 |     0.799 |                   -0.079 |    -0.158 |     0.025 |    0.141 |    1.000 |
| qwen3.5-397b-a17b     |             0.754 |     0.823 |                   -0.077 |    -0.147 |     0.019 |    0.134 |    1.000 |
| llama-4-scout         |             0.761 |     0.817 |                   -0.068 |    -0.117 |     0.005 |    0.073 |    1.000 |
| gemma-4-31b           |             0.782 |     0.836 |                   -0.064 |    -0.127 |     0.020 |    0.170 |    1.000 |
| gpt-4.1-nano          |             0.701 |     0.751 |                   -0.059 |    -0.113 |     0.018 |    0.156 |    1.000 |
| gemma-4-26b-a4b       |             0.805 |     0.847 |                   -0.052 |    -0.111 |     0.032 |    0.297 |    1.000 |
| qwen3-vl-235b         |             0.792 |     0.826 |                   -0.044 |    -0.101 |     0.038 |    0.308 |    1.000 |
| gemini-2.5-flash      |             0.539 |     0.565 |                   -0.030 |    -0.111 |     0.069 |    0.607 |    1.000 |
| gemini-2.5-pro        |             0.703 |     0.714 |                   -0.017 |    -0.088 |     0.076 |    0.831 |    1.000 |

All metrics per model: `vlm.csv`.

## 6. Dispersion and rater consistency
| measure                                    |   female |   male |   stat |     p |
|:-------------------------------------------|---------:|-------:|-------:|------:|
| PA person-mean SD (Brown-Forsythe)         |    0.419 |  0.442 |  0.002 | 0.965 |
| NA person-mean SD (Brown-Forsythe)         |    0.471 |  0.349 |  3.059 | 0.084 |
| mean pairwise inter-rater r over 200 cells |    0.453 |  0.422 |  0.031 | 0.466 |

## 7. Item-level DIF screen
Item rating ~ female + rest-of-scale score + image, cluster-robust by participant; BH-FDR over 20 items. A non-zero female coefficient means women rate this item differently from men at the same overall PA/NA level. Formal multi-group CFA invariance is not feasible with 29 men.

| scale   | item         |   beta_female |    se |     p |   p_fdr |
|:--------|:-------------|--------------:|------:|------:|--------:|
| PA      | interested   |         0.012 | 0.109 | 0.914 |   0.914 |
| PA      | excited      |         0.054 | 0.088 | 0.538 |   0.718 |
| PA      | strong       |        -0.262 | 0.159 | 0.099 |   0.330 |
| PA      | enthusiastic |        -0.081 | 0.086 | 0.344 |   0.625 |
| PA      | proud        |        -0.202 | 0.082 | 0.015 |   0.097 |
| PA      | alert        |         0.236 | 0.116 | 0.041 |   0.206 |
| PA      | inspired     |         0.076 | 0.123 | 0.536 |   0.718 |
| PA      | determined   |         0.028 | 0.134 | 0.836 |   0.880 |
| PA      | attentive    |         0.200 | 0.134 | 0.136 |   0.388 |
| PA      | active       |        -0.057 | 0.106 | 0.587 |   0.733 |
| NA      | distressed   |         0.116 | 0.067 | 0.085 |   0.330 |
| NA      | upset        |         0.088 | 0.067 | 0.184 |   0.428 |
| NA      | guilty       |        -0.032 | 0.076 | 0.670 |   0.788 |
| NA      | scared       |         0.169 | 0.065 | 0.009 |   0.094 |
| NA      | hostile      |         0.064 | 0.085 | 0.457 |   0.718 |
| NA      | irritable    |        -0.060 | 0.083 | 0.467 |   0.718 |
| NA      | ashamed      |        -0.114 | 0.091 | 0.214 |   0.428 |
| NA      | nervous      |        -0.108 | 0.085 | 0.204 |   0.428 |
| NA      | jittery      |        -0.018 | 0.076 | 0.813 |   0.880 |
| NA      | afraid       |         0.206 | 0.057 | 0.000 |   0.007 |

## 8. Sensitivity: architecture participants only
| scale   |   mean_F |   sd_F |   mean_M |   sd_M |   diff_F_minus_M |   welch_t |   welch_p |   welch_p_holm |   mwu_p |   perm_p |   hedges_g |   g_lo |   g_hi |   tost_p_d0.5 |   BF10 |
|:--------|---------:|-------:|---------:|-------:|-----------------:|----------:|----------:|---------------:|--------:|---------:|-----------:|-------:|-------:|--------------:|-------:|
| PA      |    2.649 |  0.437 |    2.638 |  0.353 |            0.011 |     0.115 |     0.909 |          0.909 |   0.985 |    0.912 |      0.027 | -0.437 |  0.518 |         0.027 |  0.265 |
| NA      |    2.069 |  0.476 |    1.823 |  0.327 |            0.247 |     2.512 |     0.015 |          0.030 |   0.038 |    0.035 |      0.557 |  0.134 |  1.028 |         0.610 |  1.827 |
