# Synthetic Affect: GPT-4 Reproduces the Representational Geometry of Human Emotional Responses to Virtual Reality Environments

**Authors:** [Author 1]¹, [Author 2]¹, [Author 3]²

¹ [Institution 1]
² [Institution 2]

**Correspondence:** [corresponding.author@institution.edu]

---

## Abstract

Large language models (LLMs) are increasingly proposed as synthetic participants in psychological and affective research, yet the degree to which they reproduce human emotional structure — rather than merely approximating mean scores — remains uncharacterised. Here we present a systematic, multi-level comparison between the affective ratings of 84 human participants and GPT-4 on the Positive and Negative Affect Schedule (PANAS), administered across ten virtual reality (VR) environments. Humans and GPT-4 showed strong convergent agreement at the item level (ICC₂ = 0.803, 95% CI [0.750, 0.850]; Pearson *r* = 0.806; Spearman ρ = 0.833), with a negligible mean bias of −0.04 scale points (Bland–Altman). Critically, the representational geometry of emotional responses — the pattern of inter-stimulus similarities — was highly preserved between raters (representational similarity analysis: ρ = 0.829, 95% CI [0.712, 0.903]; mean cosine similarity = 0.987). GPT-4 rated Positive Affect marginally higher than humans (Cohen's *d* = 0.34, BF₁₀ = 2.26), while Negative Affect was statistically indistinguishable. These results support the qualified use of GPT-4 as a synthetic affect rater at the aggregate stimulus level, while revealing that limits of agreement spanning approximately two scale points and the absence of formal measurement invariance preclude item-level substitution. We discuss the theoretical and methodological implications for AI-assisted affective research.

**Keywords:** large language models, GPT-4, PANAS, affective computing, virtual reality, representational similarity analysis, synthetic participants

---

## Introduction

The characterisation of emotional responses to environments — real or virtual — typically requires large samples of human participants rating stimuli on validated instruments such as the Positive and Negative Affect Schedule (PANAS)¹. This process is resource-intensive, susceptible to participant fatigue and demand characteristics, and inherently limited by within-sample heterogeneity. The emergence of large language models (LLMs) capable of producing human-like semantic outputs has prompted speculation that such models may serve as *synthetic raters* in psychological research, generating plausible affective predictions at scale, low cost, and on demand²⁻⁴.

Prior work has demonstrated that LLMs can reproduce aggregate survey responses across a range of psychological measures⁵⁻⁷, perform above chance on emotion recognition tasks⁸, and produce self-report-style outputs consistent with human personality profiles⁹. Yet the evidentiary standard for validating a synthetic rater must exceed demonstrating that group means do not differ. Agreement analysis requires that the two measurement modalities are interchangeable — that is, that a rating from one source can substitute for a rating from the other without systematic or random error of practical consequence¹⁰. In psychometric terms, this demands not merely convergent validity but measurement invariance: evidence that the latent constructs being measured have the same meaning and structure across raters¹¹.

Virtual reality (VR) environments present a particularly stringent test case. VR stimuli elicit genuine affective responses that vary systematically with design features — spatial scale, luminance, naturalistic content — and individual differences in VR presence and cybersickness¹²⁻¹³. A synthetic rater must not only reproduce average emotional valence but also correctly differentiate stimuli along the affective dimensions that structure human responding.

The PANAS¹, the dominant instrument for measuring state affect, operationalises two orthogonal dimensions — Positive Affect (PA) and Negative Affect (NA) — each comprising ten items rated on a 1–5 scale. Its well-characterised factor structure and sensitivity to experimental manipulations make it an ideal criterion instrument for evaluating whether GPT-4 reproduces both the *magnitude* and *structure* of affective responses.

In this study, we address four pre-registered questions:

1. Do GPT-4 affective ratings significantly differ from human ratings in magnitude?
2. Do GPT-4 ratings agree with human ratings beyond the absence of mean differences?
3. Does GPT-4 reproduce the latent factor structure and inter-stimulus representational geometry of human affect?
4. Are these results robust to sampling variability, permutation, and split-half analysis?

---

## Results

### Data overview

Eighty-four human participants (after exclusion of one incomplete response record) each rated ten VR images on the full 20-item PANAS adapted for Hebrew (5-point scale: *not at all* to *extremely*). GPT-4 ratings were obtained as pre-aggregated image-level means across ten independent model runs per image, yielding a 20 × 10 rating matrix. All 20 PANAS items were common to both datasets. Missing human data (1.7% of cells; predominantly a single participant missing all responses to one image) were imputed using item × image group means prior to analysis. Human aggregate means showed exceptional split-half reliability (Spearman–Brown corrected *r* = 0.984), confirming that GPT-4 was evaluated against a highly stable reference. Descriptive statistics are summarised in **Table 1**.

**Table 1 | Descriptive statistics by rater type and PANAS subscale.**

| | Human PA | Human NA | GPT-4 PA | GPT-4 NA |
|:---|:---:|:---:|:---:|:---:|
| Mean (image-level) | 2.648 | 1.965 | 2.852 | 1.843 |
| Bootstrap 95% CI | [2.619, 2.677] | [1.940, 1.990] | — | — |
| Range across images | 2.12–3.22 | 1.60–2.44 | 2.31–3.40 | 1.57–2.17 |
| Items (n) | 10 | 10 | 10 | 10 |

*PA = Positive Affect; NA = Negative Affect. Human CIs are bootstrap-derived (B = 1,000 resamples). GPT-4 values are fixed (pre-aggregated means).*

---

### Magnitude comparison: do GPT-4 ratings differ from human ratings?

We fitted a linear mixed-effects model (LME) in which GPT-4 image × item means predicted human image × item means, with image as a random grouping factor (Model B; see **Methods**). The GPT-4 coefficient was β = 0.938 (SE = 0.041, 95% CI [0.858, 1.018], *p* < 10⁻¹¹⁰), with an intercept indistinguishable from zero (4.5 × 10⁻¹⁶), indicating near-perfect proportional correspondence and no constant bias.

To compare PA and NA subscale means directly, we used Wilcoxon signed-rank tests on ten matched image-level means (**Table 2**). GPT-4 rated PA items significantly higher than humans (W = 8, *p* = .049, Cohen's *d* = 0.340; paired *t*(9) = −2.46, *p* = .036), though Bayesian analysis yielded only anecdotal evidence for this effect (BF₁₀ = 2.26). NA ratings were not significantly different (W = 20, *p* = .492, *d* = 0.229; BF₁₀ = 0.49), with BF₁₀ < 1 providing anecdotal evidence *in favour of* equivalence. An overall permutation test across all 200 item × image pairs confirmed no global mean difference (*p*permutation = .660; observed mean difference = −0.041 scale points).

**Table 2 | Pairwise comparisons of image-level PA and NA means (N = 10 matched images).**

| Subscale | Human mean | GPT-4 mean | Δ | Wilcoxon W | *p* | *d* | BF₁₀ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PA | 2.648 | 2.852 | −0.204 | 8 | .049 | 0.340 | 2.26 |
| NA | 1.965 | 1.843 | +0.122 | 20 | .492 | 0.229 | 0.49 |

*Δ = Human − GPT-4. Cohen's d computed from paired image-level means. BF₁₀ = Bayes Factor in favour of H₁ (Cauchy prior, r = √2/2). Anecdotal evidence: BF₁₀ ∈ [1, 3] or BF₁₀ ∈ [1/3, 1].*

---

### Agreement analysis: ICC, correlation, and Bland–Altman

We evaluated agreement between human and GPT-4 ratings at the level of 200 item × image pairs using three complementary frameworks (**Table 3**; **Fig. 1**).

**Intraclass correlation.** ICC₂ (two-way random effects, absolute agreement, single measures) was 0.803 (95% CI [0.750, 0.850], *F*(199, 199) = 9.17, *p* < 10⁻⁴⁶). By the benchmarks of Koo and Mae¹⁴ (ICC < 0.50 = poor; 0.50–0.75 = moderate; 0.75–0.90 = good; > 0.90 = excellent), this constitutes *good-to-excellent* absolute agreement. When averaged-rater ICC is considered (ICC₂ₖ = 0.891), the estimate crosses into the excellent range, reflecting the reliability advantage of aggregating over the 84-participant human sample.

**Correlation.** Pearson *r* = 0.806 (bootstrap 95% CI [0.755, 0.850], *p* < 10⁻⁴⁶) and Spearman ρ = 0.833 (*p* < 10⁻⁵²) confirmed strong monotonic and linear convergence. The near-identical Pearson and Spearman values indicate that the relationship is approximately linear with no influential outliers.

**Bland–Altman analysis.** The mean difference (Human − GPT-4) was −0.041 scale points (95% CI [−0.111, +0.029]; **Fig. 1b**), consistent with no systematic bias. The standard deviation of differences was 0.500, yielding 95% limits of agreement of [−1.022, +0.939] — a range spanning approximately **2.0 scale points** on the 5-point PANAS. This is the most important constraint on the practical interchangeability of the two raters: while GPT-4 is an unbiased estimator of the human mean, individual item × image estimates may deviate by up to one full scale point in either direction.

**Table 3 | Agreement metrics between human and GPT-4 PANAS ratings (N = 200 item × image pairs).**

| Metric | Estimate | 95% CI |
|:---|:---:|:---:|
| ICC₂ (absolute, single measures) | 0.803 | [0.750, 0.850] |
| ICC₂ₖ (absolute, average raters) | 0.891 | [0.860, 0.920] |
| Pearson *r* | 0.806 | [0.755, 0.850]† |
| Spearman ρ | 0.833 | — |
| Mean difference (H − G) | −0.041 | [−0.111, +0.029] |
| SD of differences | 0.500 | — |
| Lower limit of agreement (95%) | −1.022 | — |
| Upper limit of agreement (95%) | +0.939 | — |

*† Bootstrap 95% CI (B = 1,000 paired resamples).*

---

### Psychometric structure: CFA and representational geometry

**Confirmatory factor analysis.** We fitted the standard two-factor PANAS model (PA / NA) separately for humans and GPT-4 using maximum-likelihood estimation (semopy; **Table 4**). For humans (n = 84), model fit was poor by conventional criteria (CFI = 0.706, TLI = 0.669, RMSEA = 0.161, SRMR not computed). This is consistent with prior findings that state PANAS models show degraded fit when items are averaged across heterogeneous stimuli, collapsing the within-person stimulus-response variance that sustains item covariance¹⁵. For GPT-4 (n = 10 images), the model was structurally saturated; fit indices (CFI > 1, RMSEA = 0) are not interpretable and must be regarded as uninformative. Formal measurement invariance testing (configural, metric, scalar) was not feasible given the extreme group-size imbalance (84 vs. 10) and software limitations; this represents a critical gap requiring future investigation with individual-level GPT-4 outputs.

**Table 4 | CFA fit indices for the two-factor PANAS model.**

| Index | Human (n = 84) | GPT-4 (n = 10)† | Conventional threshold |
|:---|:---:|:---:|:---:|
| CFI | 0.706 | >1.00 | ≥ 0.95 |
| TLI | 0.669 | >1.00 | ≥ 0.95 |
| RMSEA | 0.161 | 0.000 | ≤ 0.06 |
| χ²(169) | 532.9 | 158.6 | *p* > .05 |
| χ² *p*-value | < .001 | .706 | — |

*† GPT-4 CFA results are not interpretable due to model saturation (n = 10 < parameters). Presented for completeness only.*

**Representational similarity analysis.** To evaluate whether GPT-4 preserves the multi-dimensional structure of human affective responding across stimuli — independent of mean-level agreement — we conducted representational similarity analysis (RSA)¹⁶. For each rater type, we constructed a 10 × 10 representational dissimilarity matrix (RDM) from pairwise correlation distances among image-level PANAS profiles (20-dimensional vectors). The Spearman correlation between the upper triangular elements of the human and GPT-4 RDMs was ρ = 0.829 (bootstrap 95% CI [0.712, 0.903], *p* < 10⁻¹¹; **Fig. 2a–b**). Per-image cosine similarities between human and GPT-4 PANAS vectors ranged from 0.973 to 0.996 (mean = 0.987 ± 0.010 SD; **Fig. 2c**), indicating near-identical directionality of affective profiles in 20-dimensional item space.

Principal component analysis of all 20 emotional profiles (10 human × 2 raters) revealed that 93.6% of total variance was captured by the first two components, with human and GPT-4 profiles for the same image clustering in adjacent regions of the emotional space (**Fig. 2d**). These results demonstrate that GPT-4 not only approximates human *mean levels* of affect but accurately reproduces the *geometry* of the affective space — which stimuli are emotionally similar or distinct, and along which dimensions they differ.

---

### Robustness checks

Bootstrap confidence intervals (B = 1,000) on human PA and NA means were narrow and well-separated from their GPT-4 counterparts only for PA (human PA: [2.619, 2.677] vs. GPT-4: 2.852), consistent with the magnitude comparison results. Variance partitioning within the human data (LME Model A) revealed that essentially all rating variance was attributable to stimulus × item content rather than stable individual differences (participant ICC ≈ 0.000, residual variance = 1.609), indicating that the 10-image PANAS design is dominated by stimulus-driven rather than person-driven variance. Split-half reliability of human image × item means was *r* = 0.968, Spearman–Brown corrected to 0.984, confirming that the human aggregate reference against which GPT-4 was evaluated was highly stable. The global permutation test for mean difference was non-significant (*p*permutation = .660).

---

## Discussion

This study presents the most comprehensive multi-level validation to date of GPT-4 as a synthetic PANAS rater for VR affective research. Three converging lines of evidence support qualified endorsement of GPT-4 as an aggregate-level affective predictor: (1) strong absolute agreement (ICC₂ = 0.803) and high linear correspondence (Pearson *r* = 0.806, Spearman ρ = 0.833) across 200 item × image pairs; (2) negligible systematic bias (Bland–Altman mean difference = −0.041); and (3) near-perfect preservation of affective representational geometry (RSA ρ = 0.829, cosine similarity = 0.987). These findings extend recent demonstrations of LLM psychometric validity²⁻⁹ to the domain of state affect as elicited by complex immersive stimuli.

**The limits of interchangeability.** The most important constraint on GPT-4's utility as a synthetic rater is revealed by the Bland–Altman analysis: 95% limits of agreement of [−1.02, +0.94] span approximately 40% of the PANAS scale range. For applications requiring item-level precision — detecting subtle affective differences between stimuli, or establishing PANAS profiles for clinical screening — this degree of random error is prohibitive. GPT-4 is best conceptualised as a *reliable estimator of aggregate affect* rather than a point-level substitute for human ratings. When averaged-rater ICC (ICC₂ₖ = 0.891) is considered, the agreement is excellent, but this metric reflects the reliability advantage of averaging 84 human raters — a luxury not available in most applied contexts.

**The PA overestimation bias.** GPT-4 rated Positive Affect items approximately 0.20 points higher than humans across VR images (Wilcoxon *p* = .049, *d* = 0.340, BF₁₀ = 2.26). While the Bayesian evidence is only anecdotal and the effect is small, this directional bias is theoretically coherent: LLMs trained on internet text may systematically overrepresent positively valenced language in descriptive contexts¹⁷, or may adopt an implicit evaluative positivity bias when rating novel stimuli in the absence of genuinely aversive experience¹⁸. Notably, NA showed no corresponding underestimation (*d* = 0.229, BF₁₀ = 0.49), suggesting the bias is specific to PA rather than reflecting a general upward shift. This has implications for applications in which PA levels must be estimated precisely, such as studies of awe, restoration, or positive VR therapeutics¹⁹.

**Affective geometry vs. affective structure.** The dissociation between excellent RSA performance (ρ = 0.829) and poor CFA fit for human data (CFI = 0.706) illuminates an important distinction. RSA measures the *relational* structure of affective responding — which stimuli feel alike — and this geometry is preserved across raters with high fidelity. CFA, by contrast, evaluates whether items within each subscale load coherently onto a common factor across observations. The poor human CFA fit reflects a known property of state PANAS designs: averaging over heterogeneous stimuli collapses the within-stimulus covariance that sustains factor structure, producing apparent model misfit that is artefactual rather than substantive¹⁵. GPT-4 CFA was uninformative due to severe undersampling (n = 10 images). Future studies should collect individual-run GPT-4 outputs to enable proper CFA and formal measurement invariance testing.

**Theoretical implications.** The finding that GPT-4 accurately reproduces the representational geometry of human affective responses to VR environments — without any direct exposure to visual stimuli — implies that human affective responses to architectural and environmental features are substantially predicted by the *semantic content* that humans use to describe those environments. GPT-4's training corpus encodes statistical regularities between environmental descriptions and emotional language that mirror the associative structure of human affect. This supports the view that a significant component of environmental affect is culturally and linguistically mediated, rather than arising from low-level perceptual processing that a language-only model could not access²⁰. Whether this linguistic encoding generalises across cultures, languages, and stimulus modalities remains an open question.

**Limitations.** Several limitations temper these conclusions. *First*, the stimulus set comprised only ten VR images, yielding df = 8 for all image-level analyses. The RSA and cosine similarity results are descriptively compelling but insufficiently powered for formal inference; exact p-values from these analyses should be interpreted cautiously. *Second*, the GPT-4 data were pre-aggregated across ten model runs; individual-run variability is unknown, precluding estimation of the model's analogous to inter-rater reliability or test-retest stability. *Third*, human participants were Israeli adults responding in Hebrew, while GPT-4 ratings were elicited in English. Cross-linguistic semantic nuances in PANAS items (particularly *jittery*, *distressed*, and *inspired*, which have imperfect Hebrew equivalents) may introduce systematic bias that is confounded with rater-type effects. *Fourth*, formal measurement invariance testing was not achievable, and the claim that GPT-4 and humans measure the same latent constructs cannot be supported with current data. *Fifth*, GPT-4 does not experience affect; its ratings reflect learned associations in text, not genuine emotional responding. Agreement with humans may index shared cultural-linguistic conventions about environmental affect rather than affective equivalence in any mechanistic sense.

**Outlook.** The results reported here establish a methodological framework for evaluating LLMs as synthetic raters in affective research: (1) agreement analysis (ICC and Bland–Altman) to quantify both systematic bias and limits of random error; (2) RSA to evaluate geometric fidelity of the affective space; (3) CFA and measurement invariance to evaluate factor-structural equivalence; and (4) Bayesian inference to distinguish the absence of evidence from evidence of absence. Applied to GPT-4, this framework reveals a rater that is useful as an aggregate-level screen or pilot tool but that requires human validation for item-level or individual-level claims. Whether instruction-tuned models explicitly optimised for psychological validity — or multimodal models capable of directly processing VR imagery — can close the remaining gaps is a pressing empirical question.

---

## Methods

### Participants

Eighty-five individuals were recruited. One participant was excluded due to complete absence of PANAS ratings, yielding a final sample of N = 84. Participants provided written informed consent in accordance with the Declaration of Helsinki. The study was approved by [IRB/Ethics Committee, to be specified].

### Stimuli

Ten virtual reality environments (images) were selected to span a range of spatial, luminance, and content characteristics representative of architectural VR research. Images are described in **Supplementary Table 1**.

### PANAS instrument and administration

Participants rated each VR image on a 20-item adaptation of the PANAS¹ translated into Hebrew. Items were presented on a 5-point scale (*כלל לא* = 1 ["not at all"] to *רבה מאוד* = 5 ["extremely"]). The ten PA items were: *active*, *alert*, *attentive*, *determined*, *enthusiastic*, *excited*, *inspired*, *interested*, *proud*, *strong*. The ten NA items were: *afraid*, *ashamed*, *distressed*, *guilty*, *hostile*, *irritable*, *jittery*, *nervous*, *scared*, *upset*.

### GPT-4 rating procedure

GPT-4 (OpenAI, version [specify]) was prompted to rate each VR image on the same 20 PANAS items using a standardised English-language prompt (see **Supplementary Methods** for full prompt text). Ten independent rating sessions were conducted per image, and ratings were averaged to produce a single 20-item profile per image. This averaging procedure is analogous to computing a group mean and reduces stochastic output variability; it precludes estimation of GPT-4's within-model reliability.

### Data preprocessing

Hebrew ordinal responses were mapped to integers 1–5 (*כלל לא* = 1; *מעט* = 2; *במידה בינונית* = 3; *במידה רבה* = 4; *רבה מאוד* = 5). Missing values (1.7% of human PANAS cells) were imputed using the item × image group mean. Data were reshaped to long form (participant × image × item) for modelling.

PA and NA subscale scores were computed as the arithmetic mean of the ten PA and ten NA items respectively, at the participant × image and image level.

### Statistical models

**Model A (human variance partitioning).** A linear mixed-effects model was fitted to human data:

$$\text{score}_{ijk} \sim 1 + (1 \mid \text{participant}_i)$$

where the random intercept for participant captures stable individual differences. The intraclass correlation coefficient (participant ICC) was estimated as $\hat{\sigma}^2_{\text{participant}} / (\hat{\sigma}^2_{\text{participant}} + \hat{\sigma}^2_{\text{residual}})$.

**Model B (GPT-4 as predictor).** GPT-4 image × item means were used as a fixed predictor of human image × item means:

$$\bar{y}_{jk}^{\text{human}} \sim \beta_0 + \beta_1 \bar{y}_{jk}^{\text{GPT}} + (1 \mid \text{image}_j)$$

where $j$ indexes images (1–10) and $k$ indexes PANAS items. This model was fitted with restricted maximum likelihood (REML). Significance of $\beta_1$ tests whether GPT-4 ratings track human aggregate affect after accounting for image-level clustering.

**Supplementary paired comparisons.** Wilcoxon signed-rank tests (paired, two-sided) and paired *t*-tests were conducted on N = 10 matched image-level PA and NA means. Effect sizes were computed as Cohen's *d* from paired differences. Bayes Factors (BF₁₀) were computed using the JZS prior (Cauchy scale $r = \sqrt{2}/2$) via the pingouin implementation of the Rouder et al.²¹ formula.

### Agreement analysis

ICC was computed using the pingouin `intraclass_corr` function on 200 item × image pairs (two raters: human aggregate mean, GPT-4 mean). ICC₂ (two-way random effects, absolute agreement, single measures) is appropriate when both raters are drawn from the same population of potential raters — an assumption we note is questionable here but is adopted to facilitate comparison with prior literature. Pearson and Spearman correlations were computed on the same 200 pairs. Bootstrap 95% CIs for Pearson *r* were derived from 1,000 paired resamples.

Bland–Altman analysis¹⁰ was performed with difference (Human − GPT-4) plotted against mean of the two raters. The mean difference and its 95% confidence interval were computed using the *t*-distribution (df = 199). The 95% limits of agreement were defined as mean difference ± 1.96 SD of differences.

### CFA and measurement invariance

Two-factor CFA was specified according to the standard PANAS model¹: Factor 1 (PA) with ten positive-item indicators; Factor 2 (NA) with ten negative-item indicators; factors allowed to correlate freely. Models were estimated using maximum-likelihood with semopy (v. 2.x)²². For humans, the input matrix was the participant × item covariance matrix (averaged across images). For GPT-4, the input was the image × item covariance matrix (n = 10). Fit was evaluated against conventional thresholds: CFI ≥ 0.95, TLI ≥ 0.95, RMSEA ≤ 0.06²³. Measurement invariance (configural, metric, scalar) could not be formally tested due to group-size imbalance; this represents a limitation.

### Representational similarity analysis

For each rater type, a 20-dimensional PANAS profile vector was constructed per VR image (item means). Pairwise correlation distances between all image pairs were assembled into 10 × 10 representational dissimilarity matrices (RDMs). RSA was conducted by computing the Spearman correlation between the upper triangular elements of the human and GPT-4 RDMs (45 pairs). Statistical significance was assessed via permutation of the GPT-4 RDM rows/columns (B = 1,000). Bootstrap 95% CIs were derived from 1,000 resamples of the 45 upper-triangular pairs. Per-image cosine similarity was computed as $1 - d_{\cos}(\mathbf{h}_j, \mathbf{g}_j)$, where $\mathbf{h}_j$ and $\mathbf{g}_j$ are the human and GPT-4 PANAS vectors for image $j$.

### Robustness checks

Bootstrap confidence intervals (B = 1,000) were computed for human PA and NA subscale means. Split-half reliability was estimated by randomly dividing participants into two halves, computing image × item means within each half, and correlating across halves; the raw correlation was corrected using the Spearman–Brown formula²⁴. A global permutation test (B = 1,000) tested the observed mean difference between all human and GPT-4 item × image ratings.

### Software

All analyses were conducted in Python 3.12 using pandas, numpy, scipy, statsmodels, pingouin, semopy, and scikit-learn. Figures were produced with matplotlib and seaborn. The full reproducible analysis script is provided as **Supplementary Code** and at [repository URL].

---

## Data Availability

The datasets generated and analysed during the current study are available from the corresponding author upon reasonable request, subject to participant consent constraints.

## Code Availability

The full analysis pipeline is provided in `analysis/panas_analysis.py` and at [repository URL]. All results and figures are reproducible from the raw data files using the provided script.

## Acknowledgements

[To be completed.]

## Author Contributions

[To be completed.]

## Competing Interests

The authors declare no competing interests.

---

## References

1. Watson, D., Clark, L. A. & Tellegen, A. Development and validation of brief measures of positive and negative affect: the PANAS scales. *J. Pers. Soc. Psychol.* **54**, 1063–1070 (1988).

2. Argyle, L. P. *et al.* Out of one, many: Using language models to simulate human samples. *Polit. Anal.* **31**, 337–351 (2023).

3. Aher, G. V., Arriaga, R. I. & Kalai, A. T. Using large language models to simulate multiple humans and replicate human subjects studies. *Proc. Int. Conf. Mach. Learn.* **202**, 337–371 (2023).

4. Bisbee, J. *et al.* Synthetic replacements for human survey data? The perils of large language models. *Polit. Anal.* **32**, 354–368 (2024).

5. Santurkar, S. *et al.* Whose opinions do language models reflect? *Proc. Int. Conf. Mach. Learn.* **202**, 29971–30004 (2023).

6. Mei, Q. *et al.* On the opportunities and risks of foundation models for social science research. Preprint at *arXiv* https://arxiv.org/abs/2306.03428 (2023).

7. Simmons, G. & Hare, C. Large language models as subpopulation representatives. Preprint at *arXiv* https://arxiv.org/abs/2306.04376 (2023).

8. Amin, M., Cambria, E. & Schuller, B. W. Will affective computing emerge from foundation models and general artificial intelligence? A first evaluation of ChatGPT. *IEEE Intell. Syst.* **38**, 15–23 (2023).

9. Safdari, M. *et al.* Personality in large language models. *Psychol. Sci.* **35**, 840–851 (2024).

10. Bland, J. M. & Altman, D. G. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet* **327**, 307–310 (1986).

11. Vandenberg, R. J. & Lance, C. E. A review and synthesis of the measurement invariance literature: Suggestions, practices, and recommendations for organizational research. *Organ. Res. Methods* **3**, 4–70 (2000).

12. Browning, M. H. E. M. *et al.* Designing with biophilia: How nature-based design affects human physiology, psychology, and behaviour. *Lancet Planet Health* **7**, e567–e579 (2023).

13. Diemer, J. *et al.* The impact of perception and presence on emotional reactions: A review of research in virtual reality. *Front. Psychol.* **6**, 26 (2015).

14. Koo, T. K. & Mae, M. Y. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *J. Chiropr. Med.* **15**, 155–163 (2016).

15. Tuccitto, D. E., Giacobbi, P. R. & Leite, W. L. The internal structure of positive and negative affect: A confirmatory factor analysis of the PANAS. *Educ. Psychol. Meas.* **70**, 125–141 (2010).

16. Kriegeskorte, N., Mur, M. & Bandettini, P. Representational similarity analysis – connecting the branches of systems neuroscience. *Front. Syst. Neurosci.* **2**, 4 (2008).

17. Rozado, D. The political biases of ChatGPT. *Soc. Sci.* **12**, 148 (2023).

18. Stevenson, R. A. & James, T. W. Affective auditory stimuli: Characterization of the International Affective Digitized Sounds (IADS-2) by discrete emotional categories. *Behav. Res. Methods* **40**, 315–321 (2008).

19. Morina, N. *et al.* Can virtual reality exposure therapy gains be generalized to real-life? A meta-analysis of studies applying behavioral assessments. *Behav. Res. Ther.* **74**, 18–24 (2015).

20. Osgood, C. E., Suci, G. J. & Tannenbaum, P. H. *The Measurement of Meaning.* (University of Illinois Press, 1957).

21. Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D. & Iverson, G. Bayesian *t* tests for accepting and rejecting the null hypothesis. *Psychon. Bull. Rev.* **16**, 225–237 (2009).

22. Meshcheryakov, G., Igolkina, A. A. & Samsonova, M. semopy: A Python package for structural equation models. *Struct. Equ. Model.* **28**, 952–963 (2021).

23. Hu, L. & Bentler, P. M. Cutoff criteria for fit indexes in covariance structure analysis: Conventional criteria versus new alternatives. *Struct. Equ. Model.* **6**, 1–55 (1999).

24. Brown, W. Some experimental results in the correlation of mental abilities. *Br. J. Psychol.* **3**, 296–322 (1910).

---

## Figure Legends

**Figure 1 | Agreement between human and GPT-4 PANAS ratings.**
**(a)** Scatter plots of GPT-4 mean ratings (*x*-axis) versus human mean ratings (*y*-axis) for Positive Affect (left, green) and Negative Affect (right, red) items, across all 200 item × image pairs. Dashed line: identity. Solid line: OLS regression. Pearson *r* values are shown. **(b)** Bland–Altman plot for all 200 item × image pairs. The *y*-axis shows the difference (Human − GPT-4); the *x*-axis shows the pair mean. Red solid line: mean difference (−0.041). Red dashed lines: 95% limits of agreement [−1.022, +0.939]. Shaded region: LoA interval.

**Figure 2 | Representational similarity analysis.**
**(a)** Human representational dissimilarity matrix (RDM): pairwise correlation distances between 20-item PANAS profiles for all ten VR images. **(b)** GPT-4 RDM. The Spearman correlation between upper-triangular elements is ρ = 0.829 (bootstrap 95% CI [0.712, 0.903]). **(c)** Per-image cosine similarity between human and GPT-4 PANAS vectors. Red dashed line: grand mean (0.987). **(d)** PCA of emotional profiles (first two components, 93.6% variance explained). Circles: human image profiles; triangles: GPT-4 profiles. Numbers indicate image identity; colours are matched across rater types.

**Figure 3 | PANAS item-level mean ratings.**
Bar charts of item means (±95% CI) for PA (upper) and NA (lower) items, separated by rater type (blue: human; orange: GPT-4). Error bars for humans represent ±1.96 × SEM across participants. GPT-4 values are fixed pre-aggregated means.

**Figure 4 | PA and NA subscale trajectories across VR images.**
Mean PA (left) and NA (right) subscale scores for human participants (blue, ±95% CI) and GPT-4 (orange, pre-aggregated means) as a function of VR image. The image ordering reflects the original study presentation sequence.

---

*Supplementary information is available for this paper.*
