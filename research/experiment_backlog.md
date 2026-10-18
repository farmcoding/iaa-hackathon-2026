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
| 2.1 | Out-of-fold credibility encoding for make_model, occupation, previous_insurer, region (`sklearn.preprocessing.TargetEncoder`, smooth="auto"), compared with native categorical handling | Thousands of levels; shrinkage toward the mean is credibility; unseen levels fall back to the mean (Part 2) | Small gain, better stability | R4; repos research (BSD-3) | kept: 0.1339 (fold B 0.1362); public 0.1191 |
| 2.2 | Peril mechanics features: clay soil x trees_within_5m x dry_summer_index (subsidence); freezing_days x basement x property_age (escape of water); flood_zone x elevation x distance_to_river (flood) | UK peril mechanics; large-leaf GBMs miss sparse three-way interactions | Buildings and contents D² up | J5 | tried: below best |
| 2.3 | Isotonic auto-calibration on an inner time split, then re-anchor | Each price band pays its own way | Small gain, calibration evidence | R5 (Wüthrich & Ziegel) | tried: 0.1347 vs 0.1339, inside noise |
| 2.4 | CANN-lite: glum Tweedie GLM (main effects, offset log exposure) linear predictor as LightGBM init_score; predict exp(raw + eta) because predict() does not add init_score back | Explainable GLM base plus GBM residual; also the Part 2 transfer mechanism | Neutral to small gain | R3 (Wüthrich & Merz) | crashed twice from an environment bug (driver PATH lacked /usr/sbin, so joblib could not count CPU cores); fixed 2026-10-18, retry |
| 2.5 | Wider seed bag (8-10 seeds, varied fractions, fewer rounds per seed) | Nagging predictors; seed sd 0.0026 | Small gain | R6 | tried: 0.1341, inside noise |
| 2.6 | Per-line hyperparameters (leaves, min_data, power) instead of one set | Lines differ in size and tail | Small gain | Loop batch 1 | tried (min_data scaled by line size): below best |
| 2.7 | GLM + GBM blend, weights fitted on an inner time split | Diversity | Small gain | K11 | open |
| 2.8 | Null-importance feature selection: drop features that do no better than a shuffled target | Noisy categoricals overfit | Stability | Repos research §3 | open |
| 2.9 | Premium on-levelling: written_premium divided by line-year mean | written_premium drifts train to test | Stability | Data scan | tried: below best |
| 2.10 | Customer-level cross-line features: total sum insured, lines held x credit band | Cross-line risk signal | Small gain | Data model | tried (lines held): below best |
| 2.11 | Cap motor BI with excess reload by exposure (retry of the discarded cap, with reload) | First try was inside noise without reload | Small gain | J3 | open |
| 2.13 | CatBoost Tweedie (variance_power 1.1-1.5, monotone constraints, native categoricals) as a bake-off against LightGBM per line, then as a blend member | Different tree-building and categorical handling; native ordered target statistics | Small gain in a blend | Research Part B (approved 5.1) | open |
| 2.14 | XGBoost reg:tweedie as a second challenger and blend member | Community default for this metric | Small gain in a blend | Research Part D (approved 5.2) | open |
| 2.15 | Explainable Boosting Machine (tweedie_deviance power 1.5) per line: score alone and as a blend member | Interpretable model with near-GBM accuracy; governance evidence | Neutral alone, small gain blended | Research Part B (approved 5.4) | open |
| 2.16 | Property lines (buildings, contents) less regularised than motor: per-line num_leaves and min_data_in_leaf, e.g. 31 leaves and min_data 500 for property only | TabM (a neural net) beats LightGBM on buildings and contents on both folds (fold A buildings 0.103 vs 0.084), so the property trees are probably over-regularised; the current settings were tuned on the pooled score | Gain on property lines, motor unchanged | TabM go/no-go 2026-10-18 | open |
| 2.17 | Per-line feature-group exclusion: motor without the vehicle group except the strongest fields (vehicle_age, make_model via the credibility encoder, cover_type); buildings and contents without the customer/marketing group (occupation, marital_status, household_size, email_domain, marketing_consent, document_preference, previous_insurer, campaign_code, branch_id, marketing_segment) | Variable study (reports/features/feature_study.json, fold B group drop tests): motor +0.0058 without the vehicle group; buildings +0.0035 and contents +0.0036 without customer/marketing; motor loses 0.0038 without it, so per line only. The study model used native categoricals, so the vehicle result may be make_model overfitting that the encoder already fixes | Gain on the line concerned; test motor and property as separate iterations | Variable study 2026-10-18 | open |
| 2.18 | Drop features with negative permutation importance on fold B, per line (motor: engine_size_cc, vehicle_use, campaign_code, soil_type, property_age; buildings: channel, occupation, marital_status, household_size, years_at_address, email_domain, campaign_code, region, property_type, start_month; contents: days_quote_to_inception, email_domain, previous_insurer, branch_id, wind_exposure_index, freezing_days, dry_summer_index, property_age) | Shuffling them improved fold B; fewer noise features mean steadier relativities and less drift exposure | Small gain, mainly on the worse fold | Variable study 2026-10-18 | open |
| 2.19 | Written premium as a relativity, not a level: divide by the line-year mean premium (or rank within line-year) before fitting | Strongest variable on every line, and it drifts +0.42 sd in contents test vs train; tree thresholds on the absolute premium misrank 2025 rows. Premium on-levelling by a fixed index was tried and discarded; this uses the data's own year means | Gain on fold B and on the leaderboard, where the drift bites | Variable study 2026-10-18 | open |
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
| 3.8 | Per-line trended constant submission: isolates the level part of the score from the ranking part | Public score; tests 2025 levels | Advisor review | deferred by Karl 2026-10-18; revisit before Thursday's final submission |
| 3.13 | Level backtest without a submission: predict each line's 2023 level from 2021-22 and 2024 from 2021-23 with candidate trend estimators (log-linear all years, last two years, exposure-weighted, severity x frequency split) and score the one-year-ahead error | Absolute error of predicted vs developed line loss cost | Q12 follow-up | done 2026-10-18: current method (config trend on all on-levelled years) mean abs error 1.6% vs 2.9-3.3% for re-fitted trends; caveat, config trends were chosen with these years. Largest miss buildings 2024 at -4.7% on every method (weather year): next, a weather-indexed buildings level for 2025 |

## Tier 4: GPU track (OWNER; ask before each run with run-time and quota estimate)

| # | Idea | Notes | Status |
|---|---|---|---|
| 4.1 | TabM with a Tweedie loss, adapted from RonRichman/frmtpl-scaling-laws (GPL-2.0) | Richman 2026: TabM scales best of the deep tabular models on motor pricing data. Time one epoch on 100k rows locally (M4 Metal, free) before estimating | **go (fold B, 2026-10-18 13:00)**: TabM alone D² 0.1355 vs LightGBM 0.1362 on the same rows; 50/50 blend 0.1397 (+0.0035), flat between weights 0.4 and 0.6. Gain is on property (buildings +0.0086, contents +0.0059); motor -0.0007. 32 minutes on Metal, early stop after 4 epochs. Weight fixed at 0.5 by the owner; fold A is the honest check (`scripts/blend_submission.py`) |
| 4.2 | RealMLP (pytabkit, Apache-2.0) as a second neural blend member | Cheaper to tune than TabM | open |

## Tier 3b: reserving lab (OWNER; Balona & Richman replay, adapted to one valuation date)

The ultimate method is chosen by replay, never by pricing-model D² (changing the
method changes the target itself, so D² across methods is not comparable).

| # | Idea | How it is judged | Source | Status |
|---|---|---|---|---|
| 3.9 | Replay harness: as-if valuations at each quarter-end 2022-2024, censoring claims reported later; each candidate method predicts later-reported claim counts (and their 2024 incurred) by loss-period cohort | AvE score: exposure-weighted Poisson deviance of predicted vs actual later-reported counts, averaged over as-if dates; CDR score adds the change in ultimate between successive as-if dates (stability). Balona & Richman: prefer AvE where the ultimate is volatile (motor BI), CDR where stable | Balona & Richman 2021 (Variance 2022); casact/tryangle (MPL-2.0) | open |
| 3.10 | Candidate grid inside the replay: chain-ladder averaging windows and peril groupings, reporting-delay model with right truncation, Cape Cod / BF with the pricing model as prior, BI trend 3.0-4.0% | Replay score, then the trend test, then downstream D² as a tie-breaker only | chainladder-python, tryangle | open |
| 3.11 | Blend rather than pick: linear pool of the best two or three methods, weights from the replay | Replay score | casact/reserving-ensemble (GPL-3.0, R; idea only) | open |
| 3.12 | Validate the replay on a simulated book with known ultimates before trusting it | Recovery of known ultimates | casact/claim_sim, imaginator (MPL-2.0) | open |

## Tier 5: new libraries (non-society ones need owner approval before use)

| # | Library | Licence | Use | Society-linked | Approval |
|---|---|---|---|---|---|
| 5.1 | CatBoost (native Tweedie, monotone constraints, CPU/GPU) | Apache-2.0 | GBM bake-off and blend member | No | approved 2026-10-18 |
| 5.2 | XGBoost (reg:tweedie, CPU/GPU) | Apache-2.0 | GBM bake-off and blend member | No | approved 2026-10-18 |
| 5.3 | pytabkit (RealMLP and TabM) | Apache-2.0 | Neural blend member with a Tweedie loss | No | approved 2026-10-18 |
| 5.4 | InterpretML EBM (native tweedie_deviance power 1.5) | MIT | Interpretable second model for validation; possible blend member | No | approved 2026-10-18 |
| 5.5 | lorentzenchr/model-diagnostics | MIT | Calibration and Murphy-decomposition checks | No | approved 2026-10-18 |
| 5.6 | lorentzenchr/credibility | Apache-2.0 | Bühlmann-Straub encodings | No | approved 2026-10-18 |
| 5.7 | RonRichman/frmtpl-scaling-laws | GPL-2.0 | TabM pricing pipeline reference | No (author's personal repo) | approved 2026-10-18 |
| 5.8 | casact/tryangle, chainladder-python, double_lift, mg-credibility, claim_sim | MPL-2.0 | Reserving replay, double-lift validation, credibility | Yes (CAS) | not needed |
| 5.9 | PyTorch and tabm (installed for the GPU timing test) | BSD-style / Apache-2.0 | GPU track | No | approved 2026-10-18 |

Reference only (no OSI licence; build our own from the papers): Credibility Transformer repo
(CC-BY-NC-ND), AI Tools for Actuaries notebooks, SAV tutorials, IFoA MLR blog notebooks,
ExcelFormer. Excluded: TabPFN (licence, row limit), KumoRFM (hosted), LimiX (licence unverified).

Not pursued: TabPFN (weights not OSI-licensed; 50k-row design), TabICL, Mitra, CARTE
(licences fine, scale mismatch), Credibility Transformer code (CC-BY-NC-ND), SAV
tutorials and AITools4Actuaries notebooks (no licence), micro-level reserving and
DeepTriangle (need payment histories we don't have).
