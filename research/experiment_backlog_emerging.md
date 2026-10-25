# Experiment backlog: Part 2 (emerging market)

Ranked. The agent takes the highest open item; OWNER items are human-only. Bar: the
developed base alone (primary 0.13663, level-neutral, folds B + R). Background:
reports/part2_drift.md and the Part 2 section of reports/assumptions_log.md.

## Tier 0: structured corrections on shifted fields (credibility or ridge)

- **E1 Region.** The base has never seen E01-E06 and treats region as missing. Per line,
  a multiplicative factor per region from the training rows: credibility A/E,
  `z = n / (n + k)` with n the region's claim count and k = 200 claims to start, so
  factor = 1 + z * (sum y / sum base - 1) after rebalancing the line. Follow-ups: k in
  {100, 400}; region within urban_density.
- **E2 Age band.** The base used the mean age of each band (features.AGE_BAND_MEAN).
  Credibility factor per driver_age_band (motor) / policyholder_age_band (property),
  same method as E1.
- **E3 GLM correction.** Ridge log-link Tweedie GLM on y / base (weights base ** 0.5, see
  program_emerging.md) with one-hot region, age band, new_business, channel and
  cover_type (motor); alpha in {1, 10, 100} (one per run).
- **E4 Weather beyond the developed range.** Emerging hail, wildfire and rainfall exceed
  the developed range, so the base's trees extrapolate flat. Credibility factors by
  quintile of hail_days (motor, buildings) and wildfire_index (buildings, contents), or
  add them to the E3 GLM as standardised numerics.
- **E5 Tenure for a new-business book.** Credibility factor on new_business, and on
  renewal_count capped at 3.
- **E6 Property sums insured.** Contents sum insured averages 113k vs 125k developed, and
  2025 buildings sum insured 731k vs 680k in training. Credibility factors by sum_insured
  quintile per property line.

## Tier 1: constrained learners on top of the base

- **E7** LightGBM correction on a whitelist of at most five shifted fields (region, age
  band, hail_days, wildfire_index, new_business), num_leaves 3, at most 50 trees,
  min_data_in_leaf 2000, init_score log(base). Never previous_insurer, occupation or
  make_model as free categoricals: they took the split gain in the failed v1.
- **E8** EBM correction (interpret, `init_score` = log(base)), interactions=0, few
  features (as E7), strong smoothing (`smoothing_rounds`, `max_bins` 16).
- **E9** Combine the kept Tier 0 factors multiplicatively (product of credibility
  factors), in one model.
- **E10** Per-line switch: keep a correction only on the lines where it helped in both
  folds (read the per-line D2 in the ledger).

## Tier 2: blends

- **E11** Geometric blend base^(1-w) x native^w, where native is a heavily regularised
  emerging-only LightGBM (the Part 1 specification lost to the base); w in {0.1, 0.2}.
- **E12** Seed-bag or average the two best correction forms.

## OWNER (human only; the agent skips these)

- **O1 gender**: admissible by the brief. Owner decision pending (plan, decision 1).
  Diagnostics show men under-predicted 12-17% by the base in both folds.
- **O2 Schema-compatible base**: refit the developed model without the fields emerging
  lacks (credit band, mileage, telematics) and with banded ages, so the base stops
  applying developed "missing" relativities to every emerging row (scripts/part2_base.py).
- **O3 Pooled two-market model** (developed + emerging rows with a market indicator).
- **O4 Level**: peril-group levels, fire credibility complement, 2025 weather in the level.
- **O5 Target**: bodily injury count x severity; reported cost with completion as an
  offset instead of dividing by it; recoveries on open own-damage claims (~1.5% of motor 2024).
- **O6 Peril-level frequency x severity** (glass/hail 1.6-1.8x developed frequency,
  water 0.6x): needs per-peril base predictions.
