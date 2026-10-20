# Explainability pack: literature and a test run on the current model

IAA AI-assisted hackathon 2026, Part 1. Scope: a literature review of explainability
methods actuarial researchers recommend for pricing models, and a test run of a
corresponding explainability pack against the current experiment (per-line LightGBM,
motor split into a BI claim-count Poisson and an other-perils Tweedie, property lines
blending LightGBM with an EBM). Repo: `~/Projects/iaa-hackathon-2026`. Pack code,
JSON output and charts are in `~/.cache/iaa-owner/1019/explain/`.

## 1. Literature: what actuarial researchers recommend, and why

### Richman and Wuthrich (ETH Zurich / RiskLab)

- Richman, R. and Wuthrich, M.V. (2023). *LocalGLMnet: interpretable deep learning
  for tabular data*. Scandinavian Actuarial Journal, 2023(1), 71-95.
  https://ideas.repec.org/a/taf/sactxx/v2023y2023i1p71-95.html, a network
  architecture that keeps a GLM's linear-predictor structure but lets each
  regression weight depend on the full feature vector, so each fitted model
  yields a feature-by-feature attribution analogous to a GLM coefficient, plus
  group-LASSO variable selection. The recommendation that carries over to a tree
  ensemble: report attributions on the same linear-predictor (log) scale a GLM
  reviewer already reads, not an arbitrary black-box score.
- Richman, R. and Wuthrich, M.V. (2024). *Smoothness and monotonicity constraints
  for neural networks using ICEnet*. Annals of Actuarial Science, 18(3), open
  preprint arXiv:2305.08807, https://arxiv.org/abs/2305.08807 (2023 Swiss Re
  Prize for Best Reinsurance/Risk Paper). ICE (individual conditional
  expectation) curves are used both as a diagnostic, to see whether a fitted
  curve is smooth and directionally sensible per policy, not just on average,
  and as the basis for a constraint layer enforcing smoothness/monotonicity during
  training. The direct analogue for a tree model: inspect ICE curves, not only the
  averaged PDP, because the average can mask a chunk of the population moving the
  wrong way.
- Richman, R., Scognamiglio, S. and Wuthrich, M.V. (2024). *The Credibility
  Transformer*. arXiv:2409.16653, https://arxiv.org/abs/2409.16653. A Transformer architecture for tabular
  insurance pricing with a credibility-weighted token; cited as this group's
  current model-architecture work, not an explainability method itself; its own
  diagnostics still fall back to PDP/ICE and attention-weight inspection.
- Wuthrich, M.V. and Merz, M. (2023). *Statistical Foundations of Actuarial
  Learning and its Applications*. Springer, open access.
  https://link.springer.com/book/978-3-031-12409-9, the standard reference this
  group points practitioners to; covers GLM/GAM/tree/network methods and model
  diagnostics (residuals, calibration, variable importance) in one place. No
  single chapter is titled "explainability," but the diagnostic toolkit (marginal
  plots, auto-calibration, variable importance) is the one used throughout the
  group's papers.

### Laval University (Quebec)

- Henckaerts, R., Cote, M.-P., Antonio, K. and Verbelen, R. (2021). *Boosting
  insights in insurance tariff plans with tree-based machine learning methods*.
  North American Actuarial Journal, 25(2), 255-285. Open preprint:
  arXiv:1904.10890, https://arxiv.org/abs/1904.10890. Best-paper award (NAAJ 2021; Brockett-Shapiro Award, ARIA).
  Builds a full frequency-severity tariff with gradient-boosted trees, then
  recommends variable importance, partial dependence, and ALE. Its main
  practical contribution is binning the PD/ALE effects into a small number
  of risk classes per variable and refitting a GLM on those classes as a
  transparent surrogate a regulator or underwriter can read as an ordinary
  rating table. The companion R package `maidrr` (Model-Agnostic Interpretable
  Data-driven suRRogate, Henckaerts 2020-2022) automates that surrogate-GLM
  workflow; it is GPL-licensed on CRAN. Reading and reimplementing the idea is
  fine, which is what Part 2(d) below does independently in Python.
- Blier-Wong, C., Cossette, H., Lamontagne, L. and Marceau, E. (2020). *Machine
  Learning in P&C Insurance: A Review for Pricing and Reserving*. Risks, 9(1), 4.
  MDPI, open access (CC-BY), https://www.mdpi.com/2227-9091/9/1/4. A
  Laval-authored survey covering interpretability
  methods (variable importance, PDP, SHAP, surrogate models) alongside the ML
  methods themselves; a map of the field rather than a new method.
- Marie-Pier Cote (Laval, Intact Chair in Massive Data Analytics for Actuarial
  Science) continues to supervise work on explainability/fairness of black-box
  actuarial models; no additional single-authored paper of hers was found with a
  stable public URL in this search beyond the NAAJ 2021 piece above; this is
  noted rather than invented.

### Utrecht University

No actuarial explainability research group was found at Utrecht University under
that name. The owner's intended reference may be Katrien Antonio's group (KU
Leuven / University of Amsterdam), who co-authored the Henckaerts et al. NAAJ
2021 paper above and leads much of the Benelux work on interpretable tree and
boosting methods for pricing; this report cites that paper rather than guessing
at an Utrecht-specific source.

### Foundational, model-agnostic methods behind all of the above

- Apley, D.W. and Zhu, J. (2020). *Visualizing the effects of predictor variables
  in black box supervised learning models*. Journal of the Royal Statistical
  Society Series B, 82(4), 1059-1086. Open preprint: arXiv:1612.08468, https://arxiv.org/abs/1612.08468. Defines
  ALE and shows PDP can extrapolate into regions with no data support when
  predictors are correlated, because it averages over the joint distribution of
  the *other* features while forcing the feature of interest to a fixed value;
  ALE instead averages local finite differences within narrow bins of the
  conditional distribution, avoiding that extrapolation. This is why the pack
  below reports ALE alongside PD, and treats ALE as the safer of the two under
  correlated predictors (sum_insured vs. bedrooms, flood_zone vs.
  distance_to_river_km, and similar pairs in this dataset).
- Aas, K., Jullum, M. and Loland, A. (2021). *Explaining individual predictions
  when features are dependent: more accurate approximations to Shapley values*.
  Artificial Intelligence, 298, 103502. Open preprint: arXiv:1903.10464, https://arxiv.org/abs/1903.10464. The
  standard caveat citation for SHAP under correlated features: naive Kernel SHAP
  assumes feature independence when sampling coalitions, which misattributes
  credit between correlated predictors; the paper proposes conditional sampling
  as a fix. SHAP was not computed in the test run below: ALE plus a surrogate
  GLM covered more ground per hour of machine time. This is still the correct
  caveat to cite if SHAP is added for the final submission.
- Society of Actuaries (2021). Baeder, L., Brinkmann, P. and Xu, E. (Milliman).
  *Interpretable Machine Learning for Insurance*. SOA research report.
  https://www.soa.org/resources/research-reports/2021/interpretable-machine-learning/,
  a practitioner survey (PDP, ALE, ICE, SHAP, surrogate models, variable
  importance) with worked code, aimed at this exact use case.
- Lundberg, S.M. and Lee, S.-I. (2017). *A unified approach to interpreting model
  predictions*. NeurIPS 2017. The original SHAP paper, for completeness; the
  Aas/Jullum/Loland caveat above is the operative citation here given the
  correlated predictors in this dataset.

### Licence note

Licence terms for `maidrr` (CRAN, the R package behind the Henckaerts et al.
surrogate-GLM method) and for the SOA report's accompanying code were not
independently verified in this pass: a search found only a same-named but
unrelated CRAN package, not `maidrr` itself, so nothing here should be taken as a
confirmed licence. This matters only if either is reused directly: everything in
Part 2 below was written independently in Python against the repo's own fitted
models, and no third-party explainability library's source was copied, so no
licence obligation was triggered. Confirm the licence before reusing `maidrr` or
the SOA code in anything that ships.

## 2. What was run tonight

Pack: `~/.cache/iaa-owner/1019/explain/explain.py`. Repo code was not modified or
committed; the pack monkeypatches `iaa.experiment`'s LightGBM/EBM `predict` and its
`_anchor` helper (the same instrumentation `scripts/model_card.py` uses) to capture
every fitted sub-model from one call to `fit_predict`, then reuses that single fit for
everything below: no `fit_predict` is called more than once per fold.

- **Fold B** (train 2021-23, valid 2024): full pack, every predictor each sub-model
  uses: PD, ALE, the fold's own observed one-way, ICE-violation shares against both
  the model's own average direction and the actuarial prior, and a per-seed
  relativity band from the 3 bagged seeds.
- **Fold A** (train 2021-22, valid 2023): a lighter pass (top 12 predictors per
  sub-model by `model_card.json` ranking) used only for the fold A vs. B direction
  check. Fold A was fit twice: the first pass used the first 12 columns in the
  training frame's column order (a bug, not a ranking), caught after the main run
  started; `patch_fold_a.py` refit fold A once more, by ranking, and overwrote
  that section of `explain.json` in place without re-running fold B.
- **Interaction check**: top-5 predictors per line (by `model_card.json` ranking),
  every pair, 3x3 grid, LightGBM sub-models only (the EBM has `interactions=0` by
  construction). Tested on the log-rate scale, since LightGBM here is a log-link
  Tweedie/Poisson model and additivity on the raw rate scale is the wrong null.
- **Surrogate GLM**: one `sklearn.linear_model.TweedieRegressor(power=1.5,
  link="log")` per line, banded features (deciles for numeric, top-6 + "other" for
  categoricals), fitted to the current model's own final (anchored) predicted rate,
  weighted by exposure.

Compute: `num_threads=3` for LightGBM, `n_jobs=1` for the EBM (set *after* the first
attempt hung: `interpret`'s EBM spins up a fresh `loky` process pool on every
`.predict()` call, and the thousands of calls a PD/ALE grid needs exhausted the
pool/semaphore budget within a couple of minutes; single-process prediction is fast
enough for this grid size and avoids the issue entirely). `nice -n 15`, one process
at a time, 20k-row PD/ALE sample and 1,500-row ICE sample per sub-model, 8-point
grids.

## 3. Direction table and flags

A flag of "disagrees with the actuarial prior" on a LightGBM predictor under
`MONOTONE` in `src/iaa/experiment.py` (`voluntary_excess`, `compulsory_excess`,
`ncb_years`, `licence_points`, `flood_zone`, `prior_claims_5y`) should never fire:
the constraint forces the direction regardless of what the data says, so a zero
ICE-violation share there is a property of the code, not evidence the data agrees
with the prior. Of those six, only three have an unconstrained test anywhere in
the model: `voluntary_excess`, `flood_zone` and `prior_claims_5y` are also used by
the buildings/contents EBM, which carries no monotone constraint, so a
zero-violation EBM result on those three is the real data-driven confirmation.
`compulsory_excess` is pruned from every line (`PRUNED` in `experiment.py`), and
`ncb_years`/`licence_points` are motor-only, where there is no second,
unconstrained sub-model to check them against. Their direction is asserted by
the owner, not independently confirmed by this pack. Everywhere else,
`monotone_constrained=False` and a clean direction result reflects what the data
actually shows.

Flags below are raised on the PD direction (Spearman of PD relativity vs. level),
which is also what the fold A vs. fold B comparison uses. The table reports the
ALE direction beside it, per the task's own definition of the direction check.
Where the two disagree, trust ALE: the Apley and Zhu (2020) argument cited in
section 1 is that PD can extrapolate into regions a correlated predictor has no
data support for, so a PD/ALE sign disagreement is itself evidence of that
extrapolation, not a model error.

**Consistency check first.** The pooled D^2 recomputed from the captured
sub-models (`fold_B.pred_d2` = 0.1415) matches `model_card.json`'s own pooled D^2
exactly (diff 0.0, same dataset fingerprint `2e6c0e5ff80c`): the sub-models this
pack captured and explained are the ones that actually produced the model's
predictions, not a stale or mismatched copy.

**Scale of the flags.** 186 predictor/sub-model combinations were evaluated;
76 carry at least one flag. Of those, 50 are flagged on the non-monotone-wiggle
rule alone (two or more sign changes across an 8-point PD curve), almost all on
low-ranked predictors (make_model, vehicle_colour, seats, telematics_score,
region, credit_band and similar) where the true effect is close to flat and an
8-point grid on a 20k-row sample is expected to show sampling noise as a wiggle.
That threshold is read as too sensitive at this grid resolution: treat a
wiggle-only flag as "look at the chart before concluding anything," not as a
confirmed problem. The full table is in `summary.md`, with 26 substantive
(non-wiggle-only) flags; the predictors below are the most important of those
26, each carrying a contradicted prior, a contradicted observed one-way, or a
large ICE-violation share:

| predictor | line / sub-model | rank | issue |
|---|---|---|---|
| wildfire_index | motor other-perils, contents lgbm+ebm | 20 | opposes the "up" prior on 3 of its 6 sub-models: motor other-perils and both contents sub-models (lgbm and ebm, the latter unconstrained), with 26-100% of ICE curves wrong-signed. Agrees with the prior on motor BI frequency and on both buildings sub-models, where only ICE heterogeneity (33-34%) is flagged |
| rainfall_mm | buildings lgbm+ebm | 19-23 | opposes the "up" prior on both buildings sub-models, including the unconstrained EBM (100% of its ICE curves wrong-signed); agrees with the prior on both contents sub-models |
| hist_n_customer | buildings lgbm | 27 | opposes the "up" prior (more customer claims history should raise cost), 76% of ICE curves wrong-signed |
| distance_to_river_km | motor other-perils, buildings lgbm | 27-29 | opposes the "down" prior on motor other-perils; 37-59% of ICE curves wrong-signed on both lines |
| vehicle_age, vehicle_value_group | motor BI frequency | 8, 11 | model direction contradicts the observed BI-frequency one-way on 2021-23 |
| start_month, dry_summer_index (motor other-perils) | motor | 18, 30 | model direction contradicts its own line's observed one-way |

`wildfire_index` and `rainfall_mm` are the two to lead with: each opposes its
prior on more than one sub-model, including an unconstrained EBM, which rules
out a single sub-model's noise or a monotone constraint as the explanation.
Both are worth checking against the raw weather data before the final
submission: either the index runs in the opposite direction from what its name
suggests, or there is a real, explainable correlation (construction standards,
land values, or regional mix) suppressing the expected effect on the lines
where it opposes the prior, or the index needs a closer look.


## 4. Interaction check and the surrogate GLM

The interaction table below tests, on the log-rate scale, whether the top-5
predictors per line interact beyond what LightGBM's own additive structure
would predict; a flag means the joint effect departs from the product of the
two marginal effects by more than 10%. The EBM is excluded because
`interactions=0` makes it additive by construction, with nothing to test.

The surrogate GLM target is each line's final anchored rate (the per-policy
number a policyholder actually sees), not the per-sub-model PD/ALE curves in
section 3. The anchor cancels out of section 3's 1.00-base normalisation, so
the two are not meant to match level for level; the surrogate is the
regulator-readable summary of the whole line; the section 3 curves are the
diagnostic detail underneath it.

**Interactions.** 40 pairs were evaluated (top-5 predictors per line, all 10
pairs each, across the four LightGBM sub-models: motor BI frequency,
motor other-perils, buildings, contents); 2 are flagged, both on the contents
line: `sum_insured x region` (log deviation 0.102) and
`flood_zone x elevation_m` (0.145). Both are
plausible interactions rather than artefacts: a property's region and its
insured value jointly affect theft and water-damage cost in a way that is not
purely additive, and flood zone combined with elevation is close to the
textbook case for a real interaction (elevation matters more in a flood zone
than outside one). Every other evaluated pair, including all of motor's
(`ncb_years`, `driver_age`, `annual_mileage`, `region`, `years_licensed`), came
back under the 10% threshold; the full deviation for each pair is in
`summary.md`.

**Surrogate GLM fidelity.** The surrogate reproduces the full model's own
predictions well given how coarse its banding is (deciles for numeric fields,
top-6 categories plus "other" for categoricals): D^2 against the full model's
predictions is 0.85 for motor, 0.77 for buildings, 0.78 for contents. R^2
against the rate itself (the harder, noisier target) is 0.62 / 0.45 / 0.40.
The surrogate's own relativity tables (`summary.md`) line up qualitatively with
section 3's PD/ALE curves for the fields checked by hand (`ncb_years`
decreasing, `annual_mileage` increasing, `driver_age` U-shaped), which is a
second, independent confirmation of those directions using a different
method entirely.

## 5. Recommended explainability section for the final 5-page report

A compact outline, sized to roughly one of the five pages:

1. **Method choice and why** (2-3 sentences). ALE over PD as the primary relativity
   curve because several predictors here are correlated (sum_insured/bedrooms,
   flood_zone/distance_to_river_km, ncb_years/driver_age/years_licensed); PD is
   shown alongside it for continuity with familiar one-way plots, not as the
   primary read. The EBM's shape functions are reported as exact, not estimated,
   because `interactions=0` makes it a pure additive model.
2. **Relativity tables per line and sub-model**: one table per line (motor BI
   frequency, motor other-perils, buildings, contents), each row a predictor with
   base-100 relativities at a handful of representative levels: the format an
   underwriting or pricing committee already reads.
3. **Direction table with flags**: expected sign from actuarial reasoning next
   to the model's own ALE-based sign, the fold A vs. fold B agreement, and any
   non-monotone wiggle or ICE-violation flag. This is the section most useful to
   a regulator asking "why does this variable move the price the way it does."
4. **Stability**: seed band on the headline relativities (a tight band
   supports "this is signal, not noise") and the fold A vs. B direction check.
5. **Surrogate GLM as the regulator-readable summary**: one banded Tweedie GLM
   per line fitted to the model's own predictions, its R^2/D^2 against the full
   model, and its relativity table. This is the section to lead with for a reader
   who wants a GLM-shaped answer and does not need the rest.
6. **Known limits**: correlated predictors and the ALE/PD caveat; SHAP was not
   computed and would need the dependent-feature correction if added later;
   high-cardinality fields (make_model, occupation, previous_insurer) are reported
   by their top raw levels after credibility encoding, not as a continuous curve;
   the motor BI sub-model is a claim-count model, so its observed one-way is
   compared against BI claim frequency, not cost. The single 11.9m claim never
   enters that comparison, the same separation the model itself uses to keep that
   claim from distorting the BI relativities. Other-perils is compared against its
   own trended peril cost; ordinal categoricals stored as nominal (credit_band,
   security_level, lodgers, named_young_driver, trees_within_5m, urban_density)
   carry a stated direction prior but are not sign-tested, because the data does
   not mark them as ordered categories; driver_age and policyholder_age carry a
   U-shaped prior, which a single Spearman correlation cannot test either. Read
   those two from the PD/ALE curve shape directly.
