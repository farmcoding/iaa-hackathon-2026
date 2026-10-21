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

## Tier 0: re-tests under the corrected keep bar (do these first, in order)

On 2026-10-18 the owner re-based the keep bar on the seeded champion (primary 0.1330, run
20261018T135452) instead of the favourable unseeded draw (0.1339), and added a
confirmation run on fresh seeds. These discarded runs scored close to or above the
corrected bar and are re-tested under it. Runs before 13:54 used the unseeded encoder,
whose draws of the same model averaged about 0.1334, so their scores are not directly
comparable. Re-implement from the CHANGE text in `ai_log/loop_2026-10-18.md` (search the
ledger timestamp) or from the saved patch where one exists.

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 0.1 | Re-test 2.14: XGBoost reg:tweedie blended 50/50 with LightGBM on buildings and contents only | 15:55 run: 0.13504 (A 0.1300 / B 0.1376), both folds above the seeded champion; the only discard that clears the corrected bar | Keep if it confirms | ledger 2026-10-18T15:55:03 | tried 4x, last 18 Oct 19:45, best 0.13367; discarded |
| 0.2 | Re-test the all-lines XGBoost blend if 0.1 confirms, as a follow-up (motor added) | 16:34 run: 0.13491 (A 0.1317 / B 0.1365); stronger fold A, weaker fold B than 0.1 | Keep only if it beats the new champion | `ai_log/diffs/2026-10-18T163416_discard.patch` | not run: conditional on 0.1, which failed |
| 0.3 | Re-test 2.3: isotonic recalibration of each sub-rate | 10:32 run: 0.13474 on the unseeded encoder | Small gain if real | ledger 2026-10-18T10:32:32 | tried 2x, last 18 Oct 19:59, best 0.13451; discarded |
| 0.4 | Re-test 2.17 (property part): drop the customer and marketing fields on buildings and contents | 15:34 run: 0.13442, +0.0014 vs the seeded champion | Inside the floor; confirm or close | ledger 2026-10-18T15:34:58 | tried 1x, last 18 Oct 23:29, best 0.13979; discarded |
| 0.5 | Re-test the wider seed bag (8 seeds, feature and bagging fractions varied per seed), seeds derived from SEED_OFFSET | 11:27 run: 0.13409 on the unseeded encoder | Small, steadier gain | ledger 2026-10-18T11:27:18 | tried 1x, last 18 Oct 20:15, best 0.13371; discarded |

## Tier 0f: motor, from the independent second-opinion review (20 Oct; reports/reviews/2026-10-20_alt_review_motor.md)

Motor is refitted (FROZEN_LINES = buildings, contents); the champion reproduces exactly at 0.14227. Each is one isolated run; combine only after.

| # | Idea | Why | Expected | Status |
|---|---|---|---|---|
| 2.80 | Motor-other EBM with chosen interactions only: (cover_type, driver_age), (cover_type, vehicle_age), (cover_type, urban_density); max_interaction_bins=8 | Cover type dominates the residual tables; vehicle age runs opposite ways for comprehensive and third-party cover, which an additive EBM cannot express | B +0.0004 to +0.0012 motor | tried 5x, last 20 Oct 08:16, best 0.14262; discarded |
| 2.81 | BI frequency model only: drop vehicle/severity fields (cover_type, vehicle_age, vehicle_value_group, fuel_type, overnight_parking, make_model, body_type, seats, engine_size_cc, voluntary_excess, missed_payments_12m); separate feature matrix and monotone vector for BI | These matter for other-peril cost, add ~0 or negative to BI and drive its direction flags | B +0.0003 to +0.0010 motor | tried 1x, last 20 Oct 06:54, best 0.14179; discarded |
| 2.82 | Motor: replace years_licensed by age_at_licensing = driver_age - years_licensed (values outside [17, driver_age] set missing); keep driver_age | The two are collinear; age at licensing is the informative quantity | small | tried 1x, last 20 Oct 07:03, best 0.14226; discarded |
| 2.83 | BI: CatBoost Poisson on raw admissible predictors (depth 4, lr 0.03, l2 20, 600 iterations, 3 seeds), 25% blend with LightGBM, then the BI anchor | BI has a single learner; native categoricals avoid a cost-trained encoder | small | tried 1x, last 20 Oct 07:39, best 0.14239; discarded |
| 2.84 | Motor-other EBM: single interaction (vehicle_value_group, telematics_score) | Strongest motor-other FAST pair with a clear mechanism | small | tried 2x, last 20 Oct 08:07, best 0.14262; discarded |
| 2.85 | Motor-other LightGBM: Tweedie power 1.3 (1.1 was chosen when motor's target still included BI) | Target changed since the sweep | small | tried 2x, last 20 Oct 07:57, best 0.14257; discarded |
| 2.86 | Motor: cap ncb_years at 9 (both sub-models; constraint kept) | BI frequency and other cost flatten after 9 years | small | tried 2x, last 20 Oct 08:16, best 0.14255; discarded |
| 2.87 | Motor-other only: drop dry_summer_index, distance_to_river_km, wildfire_index, mean_temperature_c | Negative or negligible contribution, direction flags, odd weather-weather interactions | small | tried 1x, last 20 Oct 08:16, best 0.14255; discarded |
| 2.88 | Pooled other-perils frequency x severity challenger (Poisson on total other count; Gamma severity on rows with claims), anchored and blended 25% with the current other rate | Separates frequency from severity without the four-way split that lost credibility | B +0.0002 to +0.0009 | tried 1x, last 20 Oct 08:27, best 0.14235; discarded |

## Tier 0e: first, under the 0.0005 floor (owner, 19 Oct evening)

| # | Idea | Earlier result | Diff | Status |
|---|---|---|---|---|
| 0e.1 | Motor other perils: 50/50 LightGBM + spline GLM blend (2.73 follow-up) | 0.14109 vs 0.14048 (+0.0006), min fold 0.13946 | ai_log/diffs/2026-10-19T204916_discard.patch | tried 1x, last 19 Oct 22:07, best 0.14117; discarded |
| 0e.2 | Motor: spline GLM blend, first version (2.73) | 0.14103 (+0.0006), min fold 0.13930 | ai_log/diffs/2026-10-19T203444_discard.patch | tried as 2.73 (3 runs 19-20 Oct, best 0.14176); discarded |
| 0e.3 | (kept 19 Oct 22:40, 0.14171) Property: CatBoost third member plus the best property near misses in one run (set FROZEN_LINES = ()) | CatBoost +0.0007 on 19 Oct; light bake-off +0.0077 / +0.0037 | ai_log/diffs/ (CatBoost runs 18-19 Oct) | kept 19 Oct 22:40 (0.14171) |

## Tier 0c: near misses re-tested under the 0.001 floor (do these first, in order)

Owner decision 2026-10-19 (Karl): the primary margin is now 0.001 (measured seed sd 0.0002). These discards gained between +0.001 and +0.002 against the bar of their day. Re-apply each saved diff on top of the current model (no premium anywhere; carry over its imports), so the harness scores it and runs the fresh-seed confirmation. Ideas built on an older model may not transfer; one attempt each.

| # | Idea | Earlier result | Diff | Status |
|---|---|---|---|---|
| 0c.1 | Property: replace pooled customer claim history with same-line-only counts and cost | 0.14022 vs 0.13867 (+0.0016), min fold 0.13858; the run's hypothesis line was lost | ai_log/diffs/2026-10-19T123541_discard.patch | tried 2x, last 20 Oct 02:15, best 0.14223; discarded |
| 0c.2 | CatBoost Tweedie as a third equal-weight property blend member (2.13) | 0.14027 vs 0.13890 (+0.0014) | ai_log/diffs/2026-10-18T235427_discard.patch | tried 1x, last 19 Oct 18:30, best 0.14115; discarded |
| 0c.3 | Rounds by early stopping on an inner time split instead of a fixed 1,200 (2.34) | 0.14008 (+0.0012), twice | ai_log/diffs/2026-10-19T041121_discard.patch | kept 20 Oct 00:34 (0.14227); 2 other run(s) discarded |
| 0c.4 | Isotonic recalibration of each sub-rate on an inner time holdout (2.3) | 0.13451 on the older model (+0.0015) | ai_log/diffs/2026-10-18T195928_discard.patch | tried 1x, last 19 Oct 23:02, best 0.13818; discarded |
| 0c.5 | XGBoost blended with LightGBM (property, then all lines) (2.14) | 0.13491 on the older model (+0.0019) | ai_log/diffs/2026-10-18T163416_discard.patch | tried 1x, last 20 Oct 03:02, best 0.14169; discarded |

## Tier 0d: data-led, from the 19 Oct evening reviews (after Tier 0c)

Read data/cache/diagnostics.md first; these target misses that persist in both folds.

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 2.73 | Motor: blend a spline GLM (sklearn SplineTransformer on numeric fields + one-hot categoricals + TweedieRegressor or PoissonRegressor for BI, solver newton-cholesky; never glum) with the LightGBM sub-models, 50/50 then re-anchor | In a light bake-off a spline GLM was the best single technique on motor (+0.0075 within-line vs LightGBM) and worst on property; motor is where the gap to the field sits | Motor gain | Interactions review 19 Oct | tried 3x, last 20 Oct 00:13, best 0.14176; discarded |
| 2.71 | Property: EBM member with interactions=15 instead of 0 | Light bake-off +0.0048 buildings / +0.0044 contents (noise sd 0.0012 / 0.0006); flood_zone x elevation_m is the top pair by three methods and the model over-predicts moderate flood zone at high elevation (A/E 0.44-0.52) | Property gain | Interactions review | tried 4x, last 20 Oct 08:07, best 0.14262; discarded |
| 2.70 | Property: CatBoost Tweedie as a third blend member, re-tested alone on the current model (allow_writing_files=False, string categoricals) | Light bake-off +0.0077 buildings / +0.0037 contents; the 19 Oct loop run gained +0.0007 overall but failed the worse-fold check | Property gain | Interactions review | done: same change as 2.13, kept 19 Oct 22:40 (0.14171) |
| 2.62 | Motor: revisit LightGBM regularisation for both sub-models (num_leaves 31-63, min_data_in_leaf 200-500); the 15 leaves / 2,000 min-data were set while the pooled target still carried the 11.9m claim | Motor within-line D2 has been flat at 0.0847-0.0849 since 18 Oct; every keep since went to property | Motor gain | Meta-review 19 Oct | tried 4x, last 20 Oct 06:11, best 0.14221; discarded |
| 2.63 | Motor new business is under-predicted in both folds (A/E 1.10 / 1.14): give the motor models new_business back (pruned) or an interaction with ncb_years | Persistent miss, 9% of motor exposure | ~0.0005 | Diagnostics, meta-review | tried 3x, last 20 Oct 04:14, best 0.14262; discarded |
| 2.64 | Contents and buildings: restore a same-policy claim-history signal (hist_n_policy, pruned on contents; A/E 1.39 / 1.29 for one prior claim) as a count, or the recency/capped version of 2.61 | Persistent miss; contents has no claim history below prior_claims_5y | Contents gain | Diagnostics, variables review | tried 1x, last 19 Oct 20:07, best 0.14055; discarded |
| 2.65 | Short-period rows (exposure under 0.5) under-predicted on motor and contents: let the model see exposure band, or a period-length flag | Persistent miss | ~0.0005 | Meta-review | tried 3x, last 20 Oct 05:18, best 0.14209; discarded |
| 2.66 | Motor: EBM blended with LightGBM on the other-perils model (interactions=10), as on property | EBM gained +0.006 on property; motor never had it with interactions | Motor gain | Meta-review | kept 19 Oct 21:52 (0.14104); 3 other run(s) discarded |
| 2.60 | Buildings: construction_type x property_age, re-tested on 3 seeds | +0.0005 single seed | Small | Variables review | tried 1x, last 19 Oct 23:27, best 0.14197; discarded |

## Tier 0b: from the 2026-10-19 reviews (do these first, in order)

Champion: LightGBM + EBM on property (0.13890). Reviews: reports/reviews/2026-10-19_*.md.

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 2.46 | EBM `outer_bags` 2 to 8 (internal bagging), same blend | EBM runs one core per outer bag, so the machine was half idle overnight; more bags should cost little wall time and steady the EBM member | Small gain, steadier | Overnight CPU check | tried 1x, last 19 Oct 10:01, best 0.13907; discarded |
| 2.45 | Split `hist_n_customer`/`hist_cost_customer` into same-line and cross-line components; test each | For buildings and contents, 64-67% of rows where customer history fires have a claim from another line behind it | Clarity; small gain either way | Features review 1.2 | tried 4x, last 20 Oct 02:15, best 0.14223; discarded |
| 2.47 | Property interactions to replace what premium carried: sum_insured bands, construction_type x property_age, flood_zone x rainfall_mm, soil_type x dry_summer_index, voluntary excess / sum_insured; add one family at a time on top of the champion | Karl 2026-10-19: premium reliance on buildings is problematic, prioritise alternatives. One bundled variant scored worse than dropping premium, so test families singly | Small gain at best | Features review 2.3 | tried 4x, last 20 Oct 09:14, best 0.14265; discarded |
| 2.7 | (above) GLM + GBM blend with sklearn GLMs | CANN-lite on sklearn now runs (0.13882); a GLM blend member is the remaining GLM idea | Small gain | CANN review | tried 5x, last 19 Oct 13:52, best 0.13859; discarded |

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
| 2.4 | CANN-lite: glum Tweedie GLM (main effects, offset log exposure) linear predictor as LightGBM init_score; predict exp(raw + eta) because predict() does not add init_score back | Explainable GLM base plus GBM residual; also the Part 2 transfer mechanism | Neutral to small gain | R3 (Wüthrich & Merz) | tested 2026-10-19 (owner, independent review): crashes were glum's OpenMP runtime clashing with LightGBM's (segfault). Rebuilt on sklearn GLMs: 0.13882 (A 0.1380 / B 0.1392) vs champion 0.13890, inside noise. Code: reports/reviews/2026-10-19_cann.md |
| 2.5 | Wider seed bag (8-10 seeds, varied fractions, fewer rounds per seed) | Nagging predictors; seed sd 0.0026 | Small gain | R6 | tried: 0.1341, inside noise |
| 2.6 | Per-line hyperparameters (leaves, min_data, power) instead of one set | Lines differ in size and tail | Small gain | Loop batch 1 | tried (min_data scaled by line size): below best |
| 2.7 | GLM + GBM blend, weights fitted on an inner time split; use sklearn GLMs (never glum, see program.md) | Diversity | Small gain | K11 | tried 5x, last 19 Oct 13:52, best 0.13859; discarded |
| 2.8 | Null-importance feature selection: drop features that do no better than a shuffled target | Noisy categoricals overfit | Stability | Repos research §3 | tried 2x, last 19 Oct 13:31, best 0.13859; discarded |
| 2.9 | Premium on-levelling: written_premium divided by line-year mean | written_premium drifts train to test | Stability | Data scan | tried: below best |
| 2.10 | Customer-level cross-line features: total sum insured, lines held x credit band | Cross-line risk signal | Small gain | Data model | tried (lines held): below best |
| 2.11 | Cap motor BI with excess reload by exposure (retry of the discarded cap, with reload) | First try was inside noise without reload | Small gain | J3 | tried 1x, last 18 Oct 12:53, best 0.13259; discarded |
| 2.13 | CatBoost Tweedie (variance_power 1.1-1.5, monotone constraints, native categoricals) as a bake-off against LightGBM per line, then as a blend member | Different tree-building and categorical handling; native ordered target statistics | Small gain in a blend | Research Part B (approved 5.1) | kept 19 Oct 22:40 (0.14171); 10 other run(s) discarded |
| 2.14 | XGBoost reg:tweedie as a second challenger and blend member | Community default for this metric | Small gain in a blend | Research Part D (approved 5.2) | tested 2026-10-18 15:55: primary 0.1350 (A 0.1300 / B 0.1376), both folds above the seeded champion; discarded only against the stale 0.1339 bar (Q16). Confirmation run on fresh seeds pending the owner's Q16 decision |
| 2.15 | Explainable Boosting Machine (tweedie_deviance power 1.5) per line: score alone and as a blend member | Interpretable model with near-GBM accuracy; governance evidence | Neutral alone, small gain blended | Research Part B (approved 5.4) | kept 18 Oct 22:24 (0.13890); 3 other run(s) discarded |
| 2.16 | Property lines (buildings, contents) less regularised than motor: per-line num_leaves and min_data_in_leaf, e.g. 31 leaves and min_data 500 for property only | TabM (a neural net) beats LightGBM on buildings and contents on both folds (fold A buildings 0.103 vs 0.084), so the property trees are probably over-regularised; the current settings were tuned on the pooled score | Gain on property lines, motor unchanged | TabM go/no-go 2026-10-18 | tried 3x, last 19 Oct 05:43, best 0.13900; discarded |
| 2.17 | Per-line feature-group exclusion: motor without the vehicle group except the strongest fields (vehicle_age, make_model via the credibility encoder, cover_type); buildings and contents without the customer/marketing group (occupation, marital_status, household_size, email_domain, marketing_consent, document_preference, previous_insurer, campaign_code, branch_id, marketing_segment) | Variable study (reports/features/feature_study.json, fold B group drop tests): motor +0.0058 without the vehicle group; buildings +0.0035 and contents +0.0036 without customer/marketing; motor loses 0.0038 without it, so per line only. The study model used native categoricals, so the vehicle result may be make_model overfitting that the encoder already fixes | Gain on the line concerned; test motor and property as separate iterations | Variable study 2026-10-18 | tested 2026-10-18 15:34 (property part): 0.1344, +0.0014 vs the seeded champion, inside the floor; discarded. Lead came from the study model (native categoricals, one fold, one seed); motor part untested |
| 2.18 | Drop features with negative permutation importance on fold B, per line (motor: engine_size_cc, vehicle_use, campaign_code, soil_type, property_age; buildings: channel, occupation, marital_status, household_size, years_at_address, email_domain, campaign_code, region, property_type, start_month; contents: days_quote_to_inception, email_domain, previous_insurer, branch_id, wind_exposure_index, freezing_days, dry_summer_index, property_age) | Shuffling them improved fold B; fewer noise features mean steadier relativities and less drift exposure | Small gain, mainly on the worse fold | Variable study 2026-10-18 | tested 2026-10-18 ~16:00: discarded. Lead came from the study model; superseded by the study v2 (3.25) |
| 2.19 | Written premium as a relativity, not a level: divide by the line-year mean premium (or rank within line-year) before fitting | Strongest variable on every line, and it drifts +0.42 sd in contents test vs train; tree thresholds on the absolute premium misrank 2025 rows. Premium on-levelling by a fixed index was tried and discarded; this uses the data's own year means | Gain on fold B and on the leaderboard, where the drift bites | Variable study 2026-10-18 | closed: duplicate of 2.9 (tried and discarded); see 2.28 for premium per unit of sum insured |
| 2.20 | Replace sklearn TargetEncoder with an out-of-fold Bühlmann-Straub encoder written in numpy: per level, exposure-weighted mean rate (sum y ÷ sum exposure), credibility Z = E ÷ (E + k) with k = exposure-weighted within-level variance ÷ between-level variance, 5 folds, seed 0, same columns | The current encoder averages y/exposure without weights (`experiment.py:114`); rows under 0.25 exposure hold 1.6% of motor exposure but 63% of the squared rate mass (contents 82%), so short-period claim rows set both the level means and the smoothing | Small gain on motor (make_model, 300 levels), steadier encodings; property neutral | This review; R4; document stage 2 | tried 3x, last 19 Oct 00:49, best 0.13897; discarded |
| 2.21 | Give the trees log(exposure) as a feature (remove `exposure` from the feature exclusion only; target and weights unchanged) | Claim frequency is not proportional to exposure: motor rows under 0.25 exposure run at 0.168 claims per exposure-year vs 0.141 for full years; 39% of 2024 and 2025 motor rows are partial-year | Motor up on both folds; property neutral to slightly up | This review; document "estimate the exposure elasticity" | tried 2x, last 20 Oct 05:18, best 0.14209; discarded |
| 2.22 | Drop exact duplicates: customer_age on motor, policyholder_age on buildings and contents, year_built (keep property_age), renewal_count (keep customer_tenure) | Spearman 1.00 in each pair; duplicates split importance and produce the spurious top motor interaction (years_licensed x customer_age) | Inside the floor; if so, owner keep for cleaner relativities and importances | This review; document stage 3 | tried 2x, last 20 Oct 09:31, best 0.14263; discarded |
| 2.23 | Weather as anomaly: for each weather field, replace the value by (value minus the address's mean over the rows of train and valid, features only) and add the address mean as a separate field | Between 66% and 96% of weather variance is between addresses, so the fields act as an address fingerprint; separating climate from the year's anomaly lets the supplied 2025 weather move weather-peril risk | Buildings and contents up, more on fold B (2024 weather year); motor neutral | This review; H4 in `docs/questions.md` | tried 1x, last 19 Oct 01:48, best 0.13896; discarded |
| 2.24 | Temper property relativities: final rate = anchor(rate^α), α per line chosen from {0.7, 0.8, 0.9, 1.0} on an inner time split (fit on all training years but the last, score the last), then refit on all years and apply α | Champion decile A/E on fold B runs from 1.16 to 0.94 (buildings) and 1.18 to 0.88 (contents); in the blend double-lift the champion's extremes sit at A/E 0.80 and 1.58 and the blend corrects both. Isotonic recalibration (2.3) was inside noise; one exponent has far less variance | Buildings and contents up; motor α likely 1.0 | This review; 2.3 | tried 2x, last 20 Oct 03:45, best 0.14114; discarded |
| 2.25 | Motor BI Poisson with a maturity exposure: fit reported BI counts with exposure x m(year, start quarter), where m is reported BI frequency in that cell ÷ the oldest training year's, estimated inside `fit_predict` from training rows | The BI model learns from reported counts (`experiment.py:139`); recent years are less mature and their mix differs (vehicle_age +0.42 sd, tenure -0.17 sd), so immaturity is credited to drifting features. 1.4 changed sample weights on developed targets; this changes the Poisson exposure for reported counts. m also absorbs true frequency trend, which is harmless because the level is anchored | Motor fold B up slightly; fold A neutral | This review; document long-tail section | tried 3x, last 20 Oct 09:42, best 0.14254; discarded |
| 2.26 | No-premium benchmark, property: remove written_premium from buildings and contents | Quick check: buildings 0.0960 without vs 0.0945 with, contents 0.0773 vs 0.0780; premium is the incumbent's price, inflating about 5% a year | Inside the floor or a small gain; if inside, owner keep under the §2.1 rule | This review; document loss-ratio pitfall | tried 2x, last 19 Oct 04:47, best 0.13889; discarded |
| 2.27 | No-premium benchmark, motor: remove written_premium from both motor sub-models | Quick check: 0.0652 without vs 0.0677 with; motor premium carries prior-year claims, including ones unreported at renewal | Motor down 0.002 to 0.003; expected discard. The ledger row is the measurement for the report | This review | tried 1x, last 19 Oct 04:47, best 0.13889; discarded |
| 2.28 | Property premium as a rate: replace written_premium by written_premium ÷ sum_insured (sum_insured stays) | Removes premium inflation and the 0.57 (buildings) and 0.91 (contents) correlation with sum insured, so trees stop splitting on size twice; differs from 2.9, which divided by the line-year mean | Neutral to small gain on property; better drift behaviour | This review | open |
| 2.29 | Group the encoder's cross-fitting by policy: pass precomputed policy-grouped splits (GroupKFold on policy_id) as the `cv` argument of TargetEncoder, which sklearn 1.9 supports (the 15:35 smoke log shows `random_state` being deprecated in its favour); apply to the 2.20 encoder instead if 2.20 is kept | With random folds a training row's encoding includes the same policy's other years, later ones included, so in-training encodings are more informative than the valid-time encoding of a renewal | Neutral to small gain; more honest training signal | This review; document "grouped separation" | tried 3x, last 19 Oct 05:26, best 0.13883; discarded |
| 2.30 | LightGBM `path_smooth` = 50 on every sub-model | Shrinks each leaf value toward its parent in proportion to the leaf's weight, a credibility weighting of leaf values, aimed at the steep property relativities | Small gain on property; motor neutral | LightGBM documentation; document stage 8 | tried 1x, last 18 Oct 20:25, best 0.13316; discarded |
| 2.31 | Fairness-sensitive drop: remove email_domain, credit_band, occupation and marital_status on all three lines; ages stay | Measures what the contested fields buy. Overlaps 2.17, which dropped the whole customer and marketing group on property only (+0.0014 vs the seeded champion); this one is field-defined and includes motor | Small loss on motor, neutral on property. The run is a measurement; keeping it is an owner judgement | This review; CIA bias and fairness resource | tried 1x, last 19 Oct 05:05, best 0.13867; discarded |
| 2.32 | Property large-loss cap: cap each training row's developed cost at the line's 99.9th percentile of positive row costs, reload the excess through the existing anchoring | Caps were only tried on motor (ledger 02:12; 2.11); buildings fire reaches 631k and flood 352k | Buildings up slightly; contents neutral | `RECOMMENDATION.md` §5; J3 | tried 1x, last 18 Oct 22:47, best 0.13959; discarded |
| 2.33 | Split motor "other" into own damage plus third-party property, theft, and glass and hail: per-peril target = reported peril cost ÷ comp_other x trend_factor, one Tweedie per peril, each anchored to its own developed loss cost | Theft, glass and collision have different drivers (parking, region, mileage, driver); one Tweedie mixes them | Motor up slightly | Document "separate by coverage/claim type" | tried 5x, last 20 Oct 08:27, best 0.14235; discarded |
| 2.34 | Rounds by early stopping on an inner time split: per line and sub-model, fit on the earlier training years, stop on the last, refit all years with best_iteration x 1.1 | n_estimators is 1,200 at learning rate 0.03 for every line and sub-model; leaves (2.16) and min_data (2.6) were varied, rounds never | Neutral to small gain; faster property fits | Document stage 6 (hyperparameter ranges) | kept 20 Oct 00:34 (0.14227); 3 other run(s) discarded |
| 2.12 | Speed: build each line's lgb.Dataset once per fold and reuse across seeds | Binning is a large share of run time | Same score, faster | Advisor review | open (speed only; no score change) |

## Tier 3: OWNER track (human-run; changes the frozen target or the evaluation)

| # | Idea | How it is judged | Source | Status |
|---|---|---|---|---|
| 3.1 | GLM reference run (glum Tweedie, main effects), logged as the actuarial baseline | Ledger row; report baseline | Judges' expectation | done 20 Oct: sklearn Tweedie GLM, main effects, per line: primary 0.12678 (fold B 0.12377) vs champion 0.14260 |
| 3.2 | Report-delay model with right truncation for BI claim counts, replacing the count chain ladder | Trend test; as-if 31 Dec 2023 reporting backtest (counts are fully backtestable) | Crevecoeur, Antonio, Verbelen | tested 20 Oct (truncated-likelihood reporting delay, lognormal/Weibull): overstates emergence; no change. Needs a point mass at delay 0 (Part 2) |
| 3.3 | Cape Cod and Bornhuetter-Ferguson with the pricing model as a priori (chainladder-python, MPL-2.0) | Trend test; backtest; downstream D² | CAS/Friedland; chainladder-python | tested 20 Oct (replay): BF/Cape Cod indistinguishable from the chain ladder outside BI (<0.2% on 2021-24 ultimates) and ~14.5% low on BI against known truth (simulated books); not adopted. Pricing-model prior equals the simple prior once prorated |
| 3.4 | Berquist-Sherman style case-adequacy diagnostic for BI (cross-sectional) | Confirms and sizes the understated-reserve finding | Berquist & Sherman 1977 | done 20 Oct (cross-sectional): recovers the simulated understatement shape; real sample too thin to fit factors; supporting evidence only |
| 3.5 | BI severity trend re-selection on capped severity with a separate excess layer; candidates 3.0%, 3.5%, 4.0%, fitted | Trend test | Reserving research §4 | tested 20 Oct via the replay (not distinguishable) and the capped-target owner test (fold B -0.00008); not adopted, reported as a sensitivity |
| 3.6 | Fold B scoring scale: score on the reported scale (prediction x row completion vs reported cost) as a check on the developed-scale score | Agreement with leaderboard | Advisor review | done 20 Oct: reported-scale scoring gives the same ranking as the developed scale on both folds |
| 3.7 | Drift and adversarial validation across years and (Part 2) across markets | Part 2 readiness | Backlog 13 | done 20 Oct: no drift beyond weather/region and mechanical ageing; found the construction_type T/W recoding (fixed) |
| 3.8 | Per-line trended constant submission: isolates the level part of the score from the ranking part | Public score; tests 2025 levels | Advisor review | deferred by Karl 2026-10-18; revisit before Thursday's final submission |
| 3.13 | Level backtest without a submission: predict each line's 2023 level from 2021-22 and 2024 from 2021-23 with candidate trend estimators (log-linear all years, last two years, exposure-weighted, severity x frequency split) and score the one-year-ahead error | Absolute error of predicted vs developed line loss cost | Q12 follow-up | done 2026-10-18: current method (config trend on all on-levelled years) mean abs error 1.6% vs 2.9-3.3% for re-fitted trends; caveat, config trends were chosen with these years. Largest miss buildings 2024 at -4.7% on every method (weather year): next, a weather-indexed buildings level for 2025 |

## Tier 3c: OWNER track from the pipeline review (2026-10-18)

From the review of the pipeline against the best-practice pricing document (full report: `docs/pipeline_review.md`).

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 3.20 | Keep-rule bar from the reproducible champion (`superseded` verdict or `baseline_run_id`) | `best_kept` compares against the unseeded 0.1339 | Bar falls 0.0009; 2.14 (0.13504) and possibly 2.17 (0.13442) get re-judged, each through a 3.22 confirmation run | Q16; this review | done 19-20 Oct |
| 3.21 | Fingerprint only data-affecting code and config keys; fix P3; re-tag TabM after an identity check | Any harness fix currently orphans TabM artefacts | One rebuild, then stable | This review | done 19-20 Oct |
| 3.22 | Confirmation re-run on fresh seeds before a keep | About one null change in six clears the floor; 35 trials so far | Fewer false keeps; about 5% more runs | This review | done 19-20 Oct |
| 3.23 | A/E band as a ledger flag, rule removed from `program.md` | Unenforced instruction; anchoring sets A/E | Clearer agent instructions | This review | done 20 Oct (ae_flag ledger column; program.md rule removed) |
| 3.24 | `scripts/compare_runs.py`: paired bootstrap CI, double-lift, decile A/E, renewal split | Stage 7 evidence for each keep and blend | Evidence, no score change | This review | done 20 Oct: bootstrap CIs exclude zero on fold B vs baseline, GLM and milestones; double-lift and decile A/E charts |
| 3.25 | Variable study v2 on the champion model, both folds, 3 seeds, repeated and conditional permutation, family drops, ALE, stability matrix | Study model differs from the champion; single run | Revised 2.17 and 2.18 | Document stages 5-6 | partial 21 Oct (Kaggle, stopped at the 12 h limit): fold A permutation importance, 3 seeds, on the 20 Oct midday model; fold B, family drops and stability matrix not reached. Rankings agree with 3.36; no change. See reviews/2026-10-21_kaggle_results.md |
| 3.26 | Target-construction sensitivity of keep decisions (IBNER, trend, BI trend, base year) | "A variable that disappears under modest alternative ultimate constructions is not a stable pricing variable" | Decision table for the report | Document long-tail section | partly done 20 Oct: capped-BI-target sensitivity (fold B -0.00008, BI level ~2% higher); other target variants not run |
| 3.27 | Report-censored as-of fold B at 31 Dec 2023 | Fold B's level uses the 2024 diagonal and 2024 trend fits | Size of the optimism in fold B's level | This review | done 20 Oct: fold B optimism about 0.0006; motor A/E 0.984 as-of vs 1.011 |
| 3.28 | Guardrail tests in §4.3 | Data-contract gaps | Trust evidence; no score change | Document data contract | done 20 Oct: tests/test_data_contract.py (12 tests) |
| 3.29 | Fairness and proxy script; recorded decision per sensitive field | No fairness evidence today | Report section; possible field removals | CIA fairness resource | done 20 Oct: reports/reviews/2026-10-20_fairness.md; keep age, credit band (monitor), occupation on buildings; marital status and household size stay out |
| 3.30 | Premium rule (§2.1) adopted and disclosed; `use_premium` config flag for Part 2 | Incumbent price; motor experience re-rating on unreported claims | Cleaner tariff story; at most a small score cost | This review | done 20 Oct: premium off on every line; harness.use_premium flag (default false) for Part 2 |
| 3.31 | TabM: 3 torch seeds averaged; early stopping on the last training year instead of a random 5% of rows | One seed (`tabm_fit.py:104`); random holdout shares policies with training | Steadier neural member; about 4.5 h GPU | Richman and Wüthrich, nagging predictors | done 20 Oct (fold B, local Metal): TabM alone 0.1350, blend adds nothing; not pursued |
| 3.32 | Pre-declare finalists and the final-selection rule in `docs/SIGNOFF.md` before Thursday | Fold B and the public board have both been used for selection | Final pick follows the declared rule; public rank only breaks ties | Document "repeatedly checking the test set" | done 19-20 Oct |
| 3.33 | Re-prioritise the level-only submission (3.8): re-levelling fold B to actual totals adds 0.0003; use the slot for a no-premium or confirmation submission unless the 3.27 backtest shows a large level error | Low diagnostic value per slot | One submission saved | This review | open |
| 3.34 | History features as of the row start: counts only, or cost capped at the line's 95th percentile | `hist_cost_*` uses incurred valued at 31 Dec 2024 for every training row | Train and test history comparable | This review | tested 20 Oct (cost capped at the 95th percentile): neutral (-0.00004); candidate for Part 2 |
| 3.35 | Charter and feature registry (role, family, availability, competition and real-world admissibility, fairness flag) | Stages 0-1 have no artefact | Report material; Part 2 setup | Document stage 0 | done 20 Oct: docs/charter.md, docs/feature_registry.md (scripts/feature_registry.py) |

## Tier 3d: OWNER track from the 2026-10-19 reviews

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 3.36 | Policy-grouped K-fold (GroupKFold on policy_id, 5 folds, years pooled, on-levelled targets) as a relativity-stability diagnostic for the champion and finalists | Out-of-time folds score few rows and are noisy; grouped folds score every row once without renewal leakage | Diagnostic only, never the keep gate (mixes years; says nothing about level or trend) | Q19 | done 20 Oct: pooled D2 0.139 (sd 0.0025 across folds); importance rank Spearman 0.67-0.74; confirms dropping hist_n_customer (buildings) and distance to river (motor). reports/reviews/2026-10-20_groupkfold.md |
| 3.37 | Motor BI large loss: cap at 1m and load expected excess from a Pareto tail instead of the one realised 11.9m claim (AY2023, open) | In the level the claim implies an 11.9m loss every four years (7% of BI). Tested 2026-10-19 in the model's level step: cap, no load 0.13884; cap + 3.1% Pareto load 0.13890 (A 0.13705 / B 0.13983); champion 0.13890. Fold B motor A/E 0.988 / 1.010 / 1.023 (champion / Pareto / no load) | Score-neutral; a judgement on expected large-loss cost | Claims review Q2 | adopted 2026-10-19 (Karl): cap 1m, Pareto load refitted per fold; see results.tsv |
| 3.38 | Standing as-if-valuation backtest of development (project each year from its first diagonal, compare a year later) | 5 of 7 peril groups within 3%; BI plain chain ladder misses by 36.5%, freq_sev runs ~8% high as-if | Trust evidence for the report | Claims review Q3 | done 20 Oct through the replay (3.9); case-reserve development cannot be backtested from a single snapshot |
| 3.39 | Peril-specific trend for buildings escape of water | Water severity has run 8-12% a year since 2021; the pooled 6.5% understates it every year | Small level gain on buildings | Claims review Q3 | adopted 2026-10-19: 9%, primary 0.13910 (+0.0001), buildings fold B A/E 1.049 to 1.038 |
| 3.48 | Per-peril trends for every buildings peril group (weather 9.0-9.2% in both windows; fire and accidental damage 1.7-4.7%) instead of the pooled 6.5% | Weather, the largest group, trends a steady 9%, which undercuts the 2024-spike reason for the pooled 6.5% | Buildings level; fold B A/E still 1.038 | Water-trend fit 2026-10-19 | adopted 2026-10-20 (Karl): 0.14260, buildings A/E B 1.025 |
| 3.40 | Build delay CDFs and the BI severity base from the matched-claim population target.py uses (drops 410 orphan claims, 0.23% of cost) | Consistency; effect immaterial today (max 0.002 on any completion fraction) | None this week | Claims review Q1 | tested 20 Oct: orphans 0.08% of cost, completion moves <=0.2pp; no change (hygiene fix for Part 2) |
| 3.45 | Decide: customer claim history pooled across lines, or same-line only | Fairness and explainability of a cross-line factor (see 2.45) | Policy decision | Features review 1.2 | adopted 20 Oct (Karl): no customer-level claim history on any line; 0.14266 with the construction_type fix |
| 3.46 | Admissibility set (owner run 2026-10-19): drop premium on property, address hazard and weather on motor, licence_points on property | Karl: email and premium reliance problematic; none of these has a rating rationale on that line | Inside noise | Features review | property part adopted 2026-10-19 (0.13898); motor address/weather kept (hail and flood are motor perils; dropping cost motor fold B 0.0012) |
| 3.47 | Run the Cramer's V proxy test for every customer-level field inside the fairness script (3.29) | Email domain shown independent of all customer fields; other fields not yet checked | Report material | Features review 3.4 | done 20 Oct: 1,415 pairs, 15 above 0.5, all age or marketing segment; email domain independent |

## Tier 4: GPU track (OWNER; ask before each run with run-time and quota estimate)

| # | Idea | Notes | Status |
|---|---|---|---|
| 4.1 | TabM with a Tweedie loss, adapted from RonRichman/frmtpl-scaling-laws (GPL-2.0) | Richman 2026: TabM scales best of the deep tabular models on motor pricing data. Time one epoch on 100k rows locally (M4 Metal, free) before estimating | **go (fold B, 2026-10-18 13:00)**: TabM alone D² 0.1355 vs LightGBM 0.1362 on the same rows; 50/50 blend 0.1397 (+0.0035), flat between weights 0.4 and 0.6. Gain is on property (buildings +0.0086, contents +0.0059); motor -0.0007. 32 minutes on Metal, early stop after 4 epochs. Weight fixed at 0.5 by the owner; fold A is the honest check (`scripts/blend_submission.py`) |
| 4.2 | RealMLP (pytabkit, Apache-2.0) as a second neural blend member | Cheaper to tune than TabM | done 21 Oct (Kaggle GPU): RealMLP alone fold A 0.0625, fold B 0.0986 (no sample-weight support, so exposure enters only as an offset-free rate); no blend weight beats the champion on either fold; not pursued |

## Tier 3b: reserving lab (OWNER; Balona & Richman replay, adapted to one valuation date)

The ultimate method is chosen by replay, never by pricing-model D² (changing the
method changes the target itself, so D² across methods is not comparable).

| # | Idea | How it is judged | Source | Status |
|---|---|---|---|---|
| 3.9 | Replay harness: as-if valuations at each quarter-end 2022-2024, censoring claims reported later; each candidate method predicts later-reported claim counts (and their 2024 incurred) by loss-period cohort | AvE score: exposure-weighted Poisson deviance of predicted vs actual later-reported counts, averaged over as-if dates; CDR score adds the change in ultimate between successive as-if dates (stability). Balona & Richman: prefer AvE where the ultimate is volatile (motor BI), CDR where stable | Balona & Richman 2021 (Variance 2022); casact/tryangle (MPL-2.0) | done 20 Oct (lite): five quarter-end as-if dates Dec 2022-Dec 2023, IBNR only (snapshot data); reports/reviews/2026-10-20_reserving_replay.md |
| 3.10 | Candidate grid inside the replay: chain-ladder averaging windows and peril groupings, reporting-delay model with right truncation, Cape Cod / BF with the pricing model as prior, BI trend 3.0-4.0% | Replay score, then the trend test, then downstream D² as a tie-breaker only | chainladder-python, tryangle | done 20 Oct: averaging windows and peril splits do not beat the volume-weighted pooled chain ladder |
| 3.11 | Blend rather than pick: linear pool of the best two or three methods, weights from the replay | Replay score | casact/reserving-ensemble (GPL-3.0, R; idea only) | done 20 Oct: fixed pools and a leave-one-date-out fitted pool; no pool beats the best single method |
| 3.12 | Validate the replay on a simulated book with known ultimates before trusting it | Recovery of known ultimates | casact/claim_sim, imaginator (MPL-2.0) | done 20 Oct: three simulated books with known ultimates; replay recovers truth; current BI method +1.5%, BF/CC/Benktander -14.5% |

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
