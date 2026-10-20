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

