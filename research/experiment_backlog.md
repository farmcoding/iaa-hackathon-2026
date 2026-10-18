# Experiment backlog

Ranked by expected gain for the effort. The loop agent takes the highest open item.
Status: open / tried (see results.tsv) / parked. Sources are in research/sources.md.

Baseline (commit afcb387): per-line LightGBM Tweedie 1.5, 3-seed bag, rate target with
exposure weight, level anchored. CV 0.1056 (A 0.0900, B 0.1212), public LB 0.1060.
Weak spots: motor D² is low in fold A (0.019); contents is low in fold A (0.032).
Best kept (2026-10-18): training power 1.1 + stronger regularisation, CV 0.1201.

| # | Idea | Why | Source | Status |
|---|---|---|---|---|
| 1 | Tune the training Tweedie power (1.3, 1.5, 1.7), scoring stays at 1.5 | Training power need not equal scoring power; cheap | Kaggle research K3 | tried: 1.3 kept (0.1104), 1.1 kept (0.1128); per-line powers discarded (0.1127) |
| 2 | Stronger regularisation: min_data_in_leaf 2000, num_leaves 15, lr 0.03 with more trees | 7% of rows have claims; small leaves overfit noise | K9 | kept (0.1201) |
| 3 | Frequency x severity by peril group, weather features driving weather perils | Different drivers per peril; 2025 weather is supplied | Actuarial practice; jurisdiction research §5 | open |
| 4 | Cap motor bodily injury per row at a threshold (try 100k, 250k) in training, reload the excess by exposure | One 11.9m claim dominates motor deviance | Jurisdiction research §3 | tried: +0.0008, inside noise; retry with excess reload by exposure |
| 5 | Monotone constraints: voluntary/compulsory excess (down), ncb_years (down), licence_points (up), flood_zone (up), prior_claims_5y (up) | Stabilises relativities; transfers to Part 2 | Actuarial practice | tried: +0.0005, inside noise; human decision pending (adopt for governance) |
| 6 | Drop high-cardinality IDs-like categoricals (make_model, occupation, previous_insurer, campaign_code, branch_id) or replace with out-of-fold target encoding | Native categoricals can overfit thousands of levels | K7 | open |
| 7 | Weather interactions: freezing_days x escape-of-water exposure (basement, property_age); dry_summer_index x clay soil x trees_within_5m for subsidence | UK peril mechanics | Jurisdiction research §5 | open |
| 8 | GLM baseline (glum Tweedie, main effects) blended with the GBM, weights fitted on fold B | Diversity; GLM is the actuarial reference | K1, K11 | open |
| 9 | Direct Tweedie GBM + per-peril freq-sev blend | Ensemble of the two structures | K2, K11 | open |
| 10 | CatBoost-style ordered target statistics via sklearn TargetEncoder (cv inside train) | Safer high-cardinality handling | K7 | open |
| 11 | Premium on-levelling: written_premium divided by line-year average | written_premium drifts train to test | Data scan 2026-10-18 | open |
| 12 | Customer-level features across lines: total sum insured, multi-line flag x credit band | Cross-line risk signal | Data model | open |
| 13 | Find time-drifting features: adversarial classifier year 2021-22 vs 2024 on features; drop or de-trend the top drifters (tenure, renewal_count, ages, written_premium) and check whether A/E holds without anchoring | Model under-predicts out of time by 9-15% without anchoring; drift handling is also the Part 2 skill | Assumptions log 2026-10-18 | open |
| 14 | Isotonic auto-calibration: fit sklearn IsotonicRegression (increasing, exposure-weighted) of actual rate on predicted rate using an inner time split (train on all but the last training year, calibrate on the last), apply to valid, then re-anchor the level | Restores auto-calibration (each price cohort pays its own way); also the formal version of the level anchoring | Wüthrich & Ziegel 2024 (R5) | open |
| 15 | CANN-lite: glum Tweedie GLM (main effects, offset log exposure) whose linear predictor is LightGBM init_score; GBM learns the residual. Predict as exp(raw_score + glm_eta); pass the offset to any early-stopping set too | GLM base keeps relativities explainable; GBM adds interactions; reusable as the Part 2 transfer mechanism | Wüthrich & Merz 2019 "Yes, we CANN!" (R3) | open |
| 16 | Credibility-shrunk encodings for high-cardinality categoricals (make_model, occupation, previous_insurer, region): Bühlmann-Straub shrinkage weight e/(e+k) toward the line mean, fitted out of fold on train only | Practical CPU analogue of entity embeddings and the Credibility Transformer's prior blend; unseen levels fall back to the mean (Part 2) | Richman, Scognamiglio, Wüthrich 2024; Padayachy et al. 2025 (R4) | open |
| 17 | Wider nagging bag: 8-10 seeds with varied bagging/feature fractions and fewer rounds per seed to stay in budget | Nagging predictors stabilise portfolio prices after ~20 nets; our seed sd is 0.0026 | Richman & Wüthrich 2020 (R6) | open |
| 18 | Per-segment true-up: multiplicative factors by line x channel (and line x region) from an inner time split, credibility-weighted | Cheaper cousin of 14; compare, don't stack blindly | R5 | open |
