# Experiment backlog: Part 2 (emerging market)

Pre-registered 25 Oct midday (owner), after two GPT-6.1 Sol reviews, a research report
(market transfer best practice) and an advisor check. Bar: P0, the sum of the per-peril
developed bases (owner keep under the deploy metric, primary 0.11554; board 0.15195).
Rules in program_emerging.md: one item and one setting per run, at most the settings listed,
no stacking, every factor gated with `lr_gate` inside the fit. The agent works down tier L
in order; OWNER items are human-only.

Gate evidence (owner check, claim counts against P0, Bonferroni over 30 tests): passes in
both years: motor gender, cover_type, occupation; contents tenure. One year only: motor
driver_age_band (2024), new_business and region (2023). Placebos (vehicle_colour,
email_domain, payment_method, channel) fail in both years.

## Tier L (agent, in order)

- **L1 Motor gender.** `offset_glm` on P0 (offset column `p0`) with cols ["gender"], gated
  (n_tests=1). Settings: alpha 0.001, 0.01, 0.1. Other lines unchanged.
- **L2 Motor gender x age band.** Mains gender and driver_age_band plus the interaction
  (gender, driver_age_band), gated as one block (n_tests=1). Alpha: the best of L1 (or 0.01
  if L1 was discarded at every setting), and 10x it.
- **L3 Motor joint GLM.** (Two crashes at 11:21/11:24 were a helper bug, a single-level
  factor in the smoke subsample; fixed by the owner 11:35. Re-run from the first setting.) Candidate list (n_tests=8): gender, driver_age_band,
  new_business, region, cover_type, occupation, vehicle_value_group, ncb_years. Each gated
  in the fit; passing ones enter one ridge GLM on P0. Alpha 0.001, 0.01, 0.1.
- **L4 Property joint GLMs.** Buildings list (n_tests=6): gender, policyholder_age_band,
  new_business, region, construction_type, energy_rating. Contents list (n_tests=6):
  gender, policyholder_age_band, new_business, region, tenure, works_from_home. Gated; one
  ridge GLM per line on P0. Alpha 0.01, 0.1, 1.
- **L5 New business x age band (motor).** Interaction (new_business, driver_age_band) with
  mains, gated as one block; alpha 0.01 and 0.1. The 2025 book is 76% new business and
  younger: does new business behave differently by age here?
- **L6 Per-peril emerging levels.** `peril_levels(tr, line, k)` on every line with P0;
  k 100 and 300.
- **L7 Hail elasticity on hail perils.** The per-peril motor bases were fitted without
  hail_days (Part 1 pruning), so this adds an effect the base lacks: multiply
  base_glass_hail (motor) and base_storm (buildings) by (hail_days / training-row mean
  hail_days) ** e, e in {0.3, 0.6}.
- **L8 Total vs per-peril base.** Geometric blend sqrt(P0 x base); one setting.
- **L9 Base families** (only once the owner has added `base_compat*` or `base_pooled*`
  columns to the config): fixed blends of P0 with each family, weights 0.25 / 0.5 / 0.75
  on the family; and the family alone.

## Tier M (agent, in order; pre-registered 25 Oct 12:30 after tier L)

Tier L result: only L1 (motor gender, alpha 0.01) kept (board 0.15933, +0.0074 over P0).
Joint GLMs (L3, L4) lost: extra factors diluted gender. Tier M tests, one at a time on top
of L1, the single factors that passed the claim-count gate in both years.

- **M1 Gender by motor peril group.** Replace L1's single motor gender GLM by two: one on
  the bodily-injury base (offset base_bodily_injury) and one on the sum of the other motor
  peril bases, each gated; alpha 0.01. One setting.
- **M2 Motor cover_type** (gate passed both years): offset GLM on the L1 prediction (pass
  the L1 motor prediction as the offset column), gated; alpha 0.01 and 0.1.
- **M3 Motor occupation** (passed both years): as M2; alpha 0.1 and 1.
- **M4 Contents tenure** (passed both years): offset GLM on P0 for contents, gated; alpha
  0.01 and 0.1.
- **M5 Pooled base family** (only if the owner has added `base_pooled*` to the config):
  as L9, with the L1 gender GLM re-fitted on top of whichever base is used.

## Tier N (owner bench then loop; pre-registered 27 Oct 21:00 from the four-model ideation, reports/reviews/2026-10-27_ideation/)

Bar: O4 (primary 0.12217). Same keep rule. At most three settings per item. Count-basis
items fit claim counts n_total with offset log(O4 prediction) (frequency proportional to
the cost prediction), ridge Poisson, and apply the fitted multiplier to cost; gender stays
as fitted in O4 (not re-estimated). Penalty (restated 21:10, before any N run): the offset is scaled to expected
claim counts, so sklearn's alpha equals k / (training claims) for a ridge worth k claims of
credibility per coefficient; the three settings are k = 50 / 200 / 800 (the alpha 1 / 10 / 100
written below would shrink everything to zero on this objective). Shape recalibration (spline of log O4 rate, alpha 10 /
1 / 0.1) was run first, 20:55: -0.00000 / -0.00003 / -0.00061, discarded.

- **N1 Motor count block.** One ridge Poisson on motor counts: previous_insurer,
  missed_payments_12m, licence_points, driver_age_band, years_licensed (quintiles),
  elevation_m (quintiles), region, fuel_type, cover_type, occupation. Alpha 1 / 10 / 100
  (statsmodels-free sklearn scale on mean deviance).
- **N2 Property count block.** Per line (buildings, contents), ridge Poisson: construction_type,
  energy_rating, trees_within_5m, flood_zone, missed_payments_12m, years_at_address,
  previous_insurer, policyholder_age_band, region. Alpha 1 / 10 / 100.
- **N3 New-business regime.** Replace O4's NB factor by a ridge Poisson on counts with
  new_business, new_business x driver_age_band (motor) / x policyholder_age_band (property),
  on every line. Alpha 1 / 10 / 100.
- **N4 Weather row factor for 2025.** Per-peril count elasticities to the row's own weather
  (hail_days, wind_exposure_index, rainfall_mm, freezing_days), estimated on developed data
  across region x year (region fixed effects), applied to each base_<peril> as
  exp(beta (w - mean w on training rows)). The folds cannot see 2025 weather: judged by
  mechanism and one board check, two submissions at most.
- **N5 Temperature.** O4 rate to the power t per line, t fitted on training rows. One setting.
- **N6 Base blend under O4.** Motor and buildings geometric 50:50 with base_compat, as
  contents is with pooled. One setting.
- **N7 Shift weights in the local fits.** O4's local GLMs (gender, NB) fitted with
  training-row weights from a classifier on new_business, age bands, years_licensed,
  previous_insurer, vehicle_value_group, cover_type (no weather, tenure or year), clipped
  at 10. Weights 0.5 / 1 power.
- **N8 Pooled per-peril base, sample-weighted** (Kaggle CPU): developed + emerging with
  `sample_weight` (emerging 5 / 20), min_sum_hessian instead of min_data_in_leaf, compat
  schema; then O4 adaptation on top.
- **N9 Frequency-severity base family** (Kaggle CPU): per peril, developed count model and
  severity model; emerging credibility on frequency and severity separately.

## OWNER

- O1 Schema-compatible developed bases (running): total and per-peril, compared with P0
  under the frozen metric; re-base if they win by the margin.
- O2 Pooled two-market bases (running): emerging weight 1 / 5 / 20, Part 1 pruning off.
- O3 Level method: balance-on-train kept (actuarial balance property, board-checked);
  Tweedie-MLE scalar scored 0.11538 vs 0.11554, inside the margin (25 Oct).
- O4 Combining kept tier-L mechanisms: once, by the owner, then confirmed on the board.
- O6 (owner runs, 25 Oct ~12:35, on top of L1, not logged) New-business count-basis
  credibility factor, k=200: motor+contents primary 0.12034 (+0.0012, boot q10 +0.0002),
  all lines 0.11985; under the 0.002 margin, not adopted. Motor NB count A/E 1.20 (2023) /
  1.07 (2024), contents 1.18 / 1.43, buildings 1.06 / 0.69. Schema-compatible base: 0.11622
  (alone) / 0.11792 (50:50 with P0) vs 0.11910, not adopted. Pooled base w5: 0.09981 alone,
  0.11477 blended, not adopted (duplicated emerging rows let the trees memorise claims).
- O5 Target sensitivity: bodily-injury tail, recoveries on open own-damage claims (~1.5% of
  2024 motor), completion as offset.
  Done 27 Oct as a level sensitivity (`scripts/part2_o5_levels.py`, assumptions log): every
  variant moves the 2025 level 2.6% or less on one family, 3.4% on all open claims; no
  directional evidence on emerging case reserves, so the v1 target stays. Completion as
  offset not run.

## Parked (evidence against, 25 Oct)

Free boosted corrections on the base; one-way credibility factor stacks; emerging-only
refit; region/age credibility factors on cost.
