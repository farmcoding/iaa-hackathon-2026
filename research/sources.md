# Research sources and findings

Gathered 2026-10-18 by research agents working from public sources; summarised and
checked against the data where possible. Items marked unverified came from secondary
summaries.

## K: Kaggle competition practice

- K1. Match the training loss to the metric: Tweedie objective for a Tweedie-scored
  pure premium. scikit-learn Tweedie insurance example
  (https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html);
  glum French MTPL tutorial.
- K2. Frequency x severity can beat a single Tweedie model when drivers differ by
  component (CAS loss-cost modelling material).
- K3. Training Tweedie power need not equal the scoring power; tune it on validation
  D² at 1.5 (general practice; the mismatch tactic is an inference).
- K7. High-cardinality categoricals: out-of-fold target encoding or native categorical
  handling; never fit encodings on the validation fold.
- K9. Typical GBM settings for sparse claims data: low learning rate, large
  min_data_in_leaf, feature subsampling, early stopping.
- K10. Seed bagging is a cheap variance reduction (confirmed here: single-seed sd 0.0026).
- K11. Blend 2 to 4 diverse models with weights fitted on out-of-fold predictions;
  full stacking is out of proportion for a solo 4-day entry.
- K13. Time-forward validation, not random K-fold, for a next-year forecast.
- K14. TabPFN is not usable at 2.4M rows; RealMLP/TabM are possible blend members but
  unlikely to beat tuned GBMs alone (NeurIPS 2024 RealMLP; TabPFN-2.5 report).
- Strategy: trust CV over the public leaderboard; with one final submission choose the
  most robust validated blend; keep a submission log; withholding strong submissions
  gains nothing (ties go to the earlier submission); don't probe the leaderboard.

## J: Jurisdiction, bodily injury and large losses

- J1. The terminology reads as Great Britain (excess, no-claims bonus, penalty points,
  comparison sites, listed buildings, clay-and-trees subsidence, 12 regions). This is an
  inference; the flood zones 1-5 don't match the Environment Agency scheme.
- J2. Road Traffic Act 1988: third-party injury cover is unlimited, which explains an
  11.9m CU claim. Ogden discount rate -0.75% (2017), -0.25% (2019), +0.5% (Jan 2025);
  periodical payment orders create long-tail, index-linked large losses; whiplash
  tariff from 31 May 2021 lowered small injury claims.
- J3. Capping: choose the threshold with mean-excess and Hill plots, model the capped
  cost, reload the excess by exposure or with a large-loss frequency model; GPD above
  the threshold. Public UK data can only inform shape (tail index, PPO propensity),
  never amounts, because the currency is fictional.
- J4. Single valuation date: report-delay patterns for IBNR counts, mature-cohort
  severity for immature years (Bornhuetter-Ferguson logic), case-reserve adequacy by
  comparing cohorts (IFoA "Triangle-free reserving"; Antonio & Plat 2014; Wüthrich 2018).
- J5. Buildings perils: freeze-thaw drives escape of water; dry summers on shrinkable
  clay with nearby trees drive subsidence (2022 heatwave); flood relates to zone,
  elevation and distance to river.

## R: Ron Richman and co-authors

Verified joint work (2020-2026):

| Co-authors | Paper | Link |
|---|---|---|
| Wüthrich | LocalGLMnet: interpretable deep learning for tabular data (2021, Brian Hey Prize) | arXiv 2107.11059 |
| Wüthrich | Nagging predictors (Risks 2020) | Risks 8(3) |
| Wüthrich | ICEnet: smoothness and monotonicity for neural nets (Annals of Actuarial Science 2024) | arXiv 2305.08807 |
| Wüthrich | LASSO regularisation within LocalGLMnet (2022) | |
| Scognamiglio, Wüthrich | The Credibility Transformer (2024) | arXiv 2409.16653 |
| Padayachy, Scognamiglio, Wüthrich | In-context learning enhanced Credibility Transformer (2025, rev. 2026) | arXiv 2509.08122 |
| Lindholm, Tsanakas, Wüthrich | Multi-task network for discrimination-free insurance prices (ASTIN Bulletin 2022) | arXiv 2207.02799 |
| Wüthrich, Avanzi, Lindholm, Maggi, Mayer, Schelldorfer, Scognamiglio | AI Tools for Actuaries (lecture notes, 2025) | SSRN 5162304 |
| (solo) | Scaling laws, tabular data and actuarial ratemaking models (2026) | arXiv 2609.03106 |

Corrections to common attributions: CANN is Wüthrich & Merz ("Yes, we CANN!", ASTIN
Bulletin 2019), not Richman. Isotonic recalibration and the balance property are
Wüthrich & Ziegel (Scand. Actuarial J. 2024, arXiv 2301.02692), not Richman. No joint
pricing paper with Balona, Ferrario or Merz was found.

- R1. Credibility Transformer: a credibility token blends a portfolio prior with the
  risk's own representation inside attention (Bühlmann-Straub logic). On freMTPL2freq
  the ensemble improves out-of-sample Poisson deviance only slightly (23.717 vs 23.726
  x 10^-2). No public code.
- R2. The in-context-learning extension handles unseen categorical levels (for example
  a new vehicle model) from a context batch of similar risks without retraining; it
  speaks directly to Part 2. No public code.
- R3. CANN: GLM in a skip connection, network learns the residual. Re-implemented here
  as GLM linear predictor = LightGBM init_score.
- R4. Practical CPU analogues: credibility-shrunk categorical encodings and offset
  transfer from the Part 1 model (CANN used as a credibility prior).
- R5. Wüthrich & Ziegel: isotonic recalibration restores auto-calibration; a log-link
  GBM does not satisfy the balance property by construction, which is why the level
  anchoring step exists.
- R6. Nagging predictors: portfolio-level stability after about 20 seeds, policy level
  sometimes hundreds.
- R7. Validation framing: Murphy decomposition of deviance into uncertainty,
  discrimination and miscalibration (Fissler, Lorentzen, Mayer, arXiv 2202.12780;
  Brauer, Menzel, Wüthrich, arXiv 2510.04556, model monitoring with Gini and
  deviance decomposition).
- Code: no official repos for LocalGLMnet, ICEnet, Credibility Transformer. The
  ActuarialDataScience tutorials repo has no licence file, so its code is not reused.
  TabPFN v2 carries a custom licence and is out of scope. LocalGLMnet, ICEnet and the
  Credibility Transformer need PyTorch and exceed the 5-minute budget; cited as
  rationale, not run.

Part 2 plan from this work: (a) offset transfer, the Part 1 model's log prediction as
init_score for a small residual GBM on the new market; (b) Bühlmann-Straub shrinkage of
new-market segment factors toward developed-market factors, weight e/(e+k);
(c) shrunk encodings that fall back to the mean for unseen levels.

## S: Standards and governance

See `docs/governance.md`.
