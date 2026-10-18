# Experiment backlog

Ranked by expected gain for the effort. The loop agent takes the highest open item.
Status: open / tried (see results.tsv) / parked. Sources are in research/sources.md.

Baseline (commit afcb387): per-line LightGBM Tweedie 1.5, 3-seed bag, rate target with
exposure weight, level anchored. CV 0.1056 (A 0.0900, B 0.1212), public LB 0.1060.
Weak spots: motor D² is low in fold A (0.019); contents is low in fold A (0.032).

| # | Idea | Why | Source | Status |
|---|---|---|---|---|
| 1 | Tune the training Tweedie power (1.3, 1.5, 1.7), scoring stays at 1.5 | Training power need not equal scoring power; cheap | Kaggle research K3 | tried: 1.3 kept (0.1104); try 1.1-1.2 and per-line powers |
| 2 | Stronger regularisation: min_data_in_leaf 2000, num_leaves 15, lr 0.03 with more trees | 7% of rows have claims; small leaves overfit noise | K9 | open |
| 3 | Frequency x severity by peril group, weather features driving weather perils | Different drivers per peril; 2025 weather is supplied | Actuarial practice; jurisdiction research §5 | open |
| 4 | Cap motor bodily injury per row at a threshold (try 100k, 250k) in training, reload the excess by exposure | One 11.9m claim dominates motor deviance | Jurisdiction research §3 | open |
| 5 | Monotone constraints: voluntary/compulsory excess (down), ncb_years (down), licence_points (up), flood_zone (up), prior_claims_5y (up) | Stabilises relativities; transfers to Part 2 | Actuarial practice | open |
| 6 | Drop high-cardinality IDs-like categoricals (make_model, occupation, previous_insurer, campaign_code, branch_id) or replace with out-of-fold target encoding | Native categoricals can overfit thousands of levels | K7 | open |
| 7 | Weather interactions: freezing_days x escape-of-water exposure (basement, property_age); dry_summer_index x clay soil x trees_within_5m for subsidence | UK peril mechanics | Jurisdiction research §5 | open |
| 8 | GLM baseline (glum Tweedie, main effects) blended with the GBM, weights fitted on fold B | Diversity; GLM is the actuarial reference | K1, K11 | open |
| 9 | Direct Tweedie GBM + per-peril freq-sev blend | Ensemble of the two structures | K2, K11 | open |
| 10 | CatBoost-style ordered target statistics via sklearn TargetEncoder (cv inside train) | Safer high-cardinality handling | K7 | open |
| 11 | Premium on-levelling: written_premium divided by line-year average | written_premium drifts train to test | Data scan 2026-10-18 | open |
| 12 | Customer-level features across lines: total sum insured, multi-line flag x credit band | Cross-line risk signal | Data model | open |
| 13 | Find time-drifting features: adversarial classifier year 2021-22 vs 2024 on features; drop or de-trend the top drifters (tenure, renewal_count, ages, written_premium) and check whether A/E holds without anchoring | Model under-predicts out of time by 9-15% without anchoring; drift handling is also the Part 2 skill | Assumptions log 2026-10-18 | open |
