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

Pending (research agent still running).

## S: Standards and governance

See `docs/governance.md`.
