# Experiment backlog

The loop agent takes the highest open item in Tiers 1 and 2. Items marked OWNER are
human-only (they change the frozen target or need GPU permission); the agent skips them.
Status: open / tried (see results.tsv) / parked. Sources: research/sources.md.

Best so far (2026-10-18, batch 2 start): motor bodily injury as a Poisson frequency model with
severity from anchoring, other perils Tweedie power 1.1, num_leaves 15, min_data_in_leaf
2000, 3-seed bag, monotone constraints (owner), level anchored. Score 0.1267 (fold A
0.1160, fold B 0.1320). Earlier best: fold A 0.1135, fold B 0.1265; public leaderboard 0.1120 (for the version before the monotone constraints).
Primary metric now weights fold B twice (it tracks the leaderboard; fold A inflated
gains about fourfold).

Where the points are: motor is the largest line by cost and has the weakest within-line
D² (fold B 0.064); one 11.9m bodily injury claim sits in every motor split. Buildings
A/E in fold B is 1.049 because anchoring cannot see a weather-heavy year, while 2025
weather is supplied.

Already tried (batch 1): training power 1.3 and 1.1 (kept), per-line powers (discarded),
stronger regularisation (kept), BI cap without reload (inside noise), monotone
constraints (owner keep).

## Tier 1: expected to move the score

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 1.1 | Motor split: bodily injury as Poisson frequency on BI claim counts x a flat trended BI severity; other motor perils as Tweedie on their cost | BI severity is barely predictable from rating factors; modelling BI cost directly puts one 11.9m claim into every split | Motor D² up in both folds | Advisor review; J2, J3 | kept: score 0.1267 (fold B 0.1320), largest single gain so far |
| 1.2 | Weather perils float: for buildings and contents, model weather perils (storm, flood, subsidence, escape of water) separately with 2025 weather, letting their level move with the weather; anchor only non-weather perils | Host supplied 2025 weather; anchoring hides weather years (buildings A/E 1.049 in fold B) | Buildings A/E nearer 1.00; buildings D² up | Advisor review; J5 | tried: 0.1213, below best; retry with weather features on the peril model only and the level floated by weather index |
| 1.3 | Anchor the level on the most recent one or two on-levelled years instead of all training years | Portfolio mix drifts over time | Per-line A/E nearer 1.00 in both folds | Advisor review | tried: +0.0000, inside noise |
| 1.4 | sample_weight = exposure x completion (`comp_*` columns), so immature, assumption-heavy targets count less | Developed targets for immature periods carry 1/c variance | Fold B and motor up | Advisor review; reserving research | tried: -0.0000, inside noise |

## Tier 2: likely gains, cheap

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 2.1 | Out-of-fold credibility encoding for make_model, occupation, previous_insurer, region (`sklearn.preprocessing.TargetEncoder`, smooth="auto"), compared with native categorical handling | Thousands of levels; shrinkage toward the mean is credibility; unseen levels fall back to the mean (Part 2) | Small gain, better stability | R4; repos research (BSD-3) | open |
| 2.2 | Peril mechanics features: clay soil x trees_within_5m x dry_summer_index (subsidence); freezing_days x basement x property_age (escape of water); flood_zone x elevation x distance_to_river (flood) | UK peril mechanics; large-leaf GBMs miss sparse three-way interactions | Buildings and contents D² up | J5 | open |
| 2.3 | Isotonic auto-calibration on an inner time split, then re-anchor | Each price band pays its own way | Small gain, calibration evidence | R5 (Wüthrich & Ziegel) | open |
| 2.4 | CANN-lite: glum Tweedie GLM (main effects, offset log exposure) linear predictor as LightGBM init_score; predict exp(raw + eta) because predict() does not add init_score back | Explainable GLM base plus GBM residual; also the Part 2 transfer mechanism | Neutral to small gain | R3 (Wüthrich & Merz) | open |
| 2.5 | Wider seed bag (8-10 seeds, varied fractions, fewer rounds per seed) | Nagging predictors; seed sd 0.0026 | Small gain | R6 | open |
| 2.6 | Per-line hyperparameters (leaves, min_data, power) instead of one set | Lines differ in size and tail | Small gain | Loop batch 1 | open |
| 2.7 | GLM + GBM blend, weights fitted on an inner time split | Diversity | Small gain | K11 | open |
| 2.8 | Null-importance feature selection: drop features that do no better than a shuffled target | Noisy categoricals overfit | Stability | Repos research §3 | open |
| 2.9 | Premium on-levelling: written_premium divided by line-year mean | written_premium drifts train to test | Stability | Data scan | open |
| 2.10 | Customer-level cross-line features: total sum insured, lines held x credit band | Cross-line risk signal | Small gain | Data model | open |
| 2.11 | Cap motor BI with excess reload by exposure (retry of the discarded cap, with reload) | First try was inside noise without reload | Small gain | J3 | open |
| 2.12 | Speed: build each line's lgb.Dataset once per fold and reuse across seeds | Binning is a large share of run time | Same score, faster | Advisor review | open |

## Tier 3: OWNER track (human-run; changes the frozen target or the evaluation)

| # | Idea | How it is judged | Source | Status |
|---|---|---|---|---|
| 3.1 | GLM reference run (glum Tweedie, main effects), logged as the actuarial baseline | Ledger row; report baseline | Judges' expectation | open |
| 3.2 | Report-delay model with right truncation for BI claim counts, replacing the count chain ladder | Trend test; as-if 31 Dec 2023 reporting backtest (counts are fully backtestable) | Crevecoeur, Antonio, Verbelen | open |
| 3.3 | Cape Cod and Bornhuetter-Ferguson with the pricing model as a priori (chainladder-python, MPL-2.0) | Trend test; backtest; downstream D² | CAS/Friedland; chainladder-python | open |
| 3.4 | Berquist-Sherman style case-adequacy diagnostic for BI (cross-sectional) | Confirms and sizes the understated-reserve finding | Berquist & Sherman 1977 | open |
| 3.5 | BI severity trend re-selection on capped severity with a separate excess layer; candidates 3.0%, 3.5%, 4.0%, fitted | Trend test | Reserving research §4 | open |
| 3.6 | Fold B scoring scale: score on the reported scale (prediction x row completion vs reported cost) as a check on the developed-scale score | Agreement with leaderboard | Advisor review | open |
| 3.7 | Drift and adversarial validation across years and (Part 2) across markets | Part 2 readiness | Backlog 13 | open |
| 3.8 | Per-line trended constant submission: isolates the level part of the score from the ranking part | Public score vs 0.1120 | Advisor review | awaiting approval |

## Tier 4: GPU track (OWNER; ask before each run with run-time and quota estimate)

| # | Idea | Notes | Status |
|---|---|---|---|
| 4.1 | TabM with a Tweedie loss, adapted from RonRichman/frmtpl-scaling-laws (GPL-2.0) | Richman 2026: TabM scales best of the deep tabular models on motor pricing data. Time one epoch on 100k rows locally (M4 Metal, free) before estimating | open |
| 4.2 | RealMLP (pytabkit, Apache-2.0) as a second neural blend member | Cheaper to tune than TabM | open |

Not pursued: TabPFN (weights not OSI-licensed; 50k-row design), TabICL, Mitra, CARTE
(licences fine, scale mismatch), Credibility Transformer code (CC-BY-NC-ND), SAV
tutorials and AITools4Actuaries notebooks (no licence), micro-level reserving and
DeepTriangle (need payment histories we don't have).
