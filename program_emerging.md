# program_emerging.md: instructions for the Part 2 experiment agent

You are running one experiment in an experiment loop for an actuarial pricing model in a
new market. A human actuary (FCIA) owns this work, sets the direction in this file and
approves every leaderboard submission. You propose and test; you never submit.

## The task

Predict expected ultimate net claim cost per 2025 policy row for motor, buildings and
contents in the **emerging market**. The insurer has written there for two years (2023
and 2024); the 2025 book is 76% new business from a broker. Score: Tweedie D² with power
1.5, pooled across lines.

What is already known (reports/part2_drift.md, reports/assumptions_log.md, Part 2
section):

- The Part 1 model fitted on the developed market ranks emerging policies better than a
  refit on emerging data (level-neutral D² on 2024: 0.151 vs 0.123). Its prediction is
  attached to every row as the column `base` (CU for the row).
- The model to beat is that base alone (owner keep, primary 0.13663).
- Free-form boosted corrections on top of the base made things worse on both folds,
  whatever the regularisation (25-300 trees, leaves of 300-3000 rows). They fit noise:
  contents has about 350 claims a year, buildings about 500. previous_insurer and
  occupation took most of the split gain. **Corrections need structure: few factors,
  strong shrinkage, each one with an actuarial reason.**
- The emerging market differs in level (lower severity, higher motor frequency), climate
  (hail, wildfire and rainfall beyond the developed range), regions (E01-E06, which the
  base has never seen; it treats them as missing) and book (age bands, more new business).

## What you may change

Only `src/iaa/experiment_emerging.py`. Its contract is in its docstring:
`fit_predict(train, valid) -> np.ndarray`. Everything else is frozen: data loading,
target, development, trend, folds, the base predictions, the level step, scoring, the
harness, the tests and the config. The driver reverts the run and logs a violation if any
other file changes.

**The level is not yours.** After `fit_predict` returns, the harness rescales each line
so that its predictions on the training rows sum to the trended training actuals
(`valid` may therefore contain the training rows as well). So a constant factor per line
does nothing. Only relativities within a line count.

## The keep metric

Level-neutral D²: each line's validation predictions are rescaled to the line's actual
total before scoring, so the metric measures ranking and spread only. Two folds:

- B: train 2023, score 2024 (out of time);
- R: train 2024, score 2023 (the reverse: a second read on the same book; its truth is the
  better-developed year).

Primary = mean of B and R. A run is kept when the primary beats the best kept run by more
than 0.001 (measured seed sd 0.0004), the worse fold falls by no more than 0.002, and a
rerun on fresh seeds (`IAA_SEED_OFFSET`) passes again. So every random choice must derive
from `SEED_OFFSET` or `SEEDS`.

## One iteration

1. Read `data/cache/diagnostics.md` first. It shows A/E by segment for the model to beat,
   level-neutral, fold B and fold R side by side. Misses flagged PERSISTENT (both folds,
   same direction, >5%, >=2% of exposure) are your best hypotheses. Ignore misses on
   inadmissible fields (list below); they are not yours to fix.
   Then read `results_emerging.tsv` (all Part 2 runs and verdicts),
   `research/experiment_backlog_emerging.md` (ranked ideas; OWNER items are human-only,
   skip them) and the current `src/iaa/experiment_emerging.py`.
2. Pick ONE hypothesis: the highest-ranked backlog item not yet tried, or a follow-up a
   past result clearly suggests. Do not repeat a discarded idea unless you change
   something material, and say what.
3. Edit `src/iaa/experiment_emerging.py`. Keep the diff small enough to review in two
   minutes. When re-testing a saved diff from `ai_log/diffs/`, copy it exactly, imports
   included.
4. Do not run anything. You have no shell; the driver runs the harness, logs the result
   in `results_emerging.tsv` and commits or reverts. A crash costs an iteration, so check
   column names against `src/iaa/features.py` and the dataset columns listed below.
5. Finish with a final message whose first line is exactly:
   HYPOTHESIS: <what you are testing and why, one line, cite the backlog item or run>
   then:
   CHANGE: <what you edited>
   EXPECTED: <which fold or line you expect to move, and why>

## Rules

- **Inadmissible, never as a feature** (owner decisions): written_premium, email_domain,
  campaign_code, marketing_segment, branch_id, days_quote_to_inception,
  marketing_consent, hist_n_customer, hist_cost_customer; licence_points on buildings and
  contents. **gender** is an owner decision still pending: do not use it, even to test.
- **Leakage**: never renewal_decision, customer_service_contacts, any target column
  (cost_*, n_*, ult_*, comp_*, y, y_ult), trend_factor or the row's year.
- **The base is fixed.** Use `base` as an offset, a feature or a blend member, but don't
  rebuild it: it was fitted on the developed market, outside this module.
- **Shrinkage first.** Any factor estimated on emerging data must be credibility-weighted
  or penalised towards no change (factor 1, coefficient 0). State the shrinkage in the
  HYPOTHESIS line. Prefer a correction on a field the drift report shows has shifted, or
  a persistent diagnostic miss, over a generic model upgrade.
- **Offsets in a GLM**: scikit-learn has no offset argument. For a log-link Tweedie
  correction, fit `TweedieRegressor(power=1.5, link="log", alpha=...)` on `y / base` with
  `sample_weight = base ** 0.5` (the Tweedie deviance scales as mu^(2-p)); predict
  `base * model.predict(X)`. One-hot encode a handful of factors; standardise numerics.
- Time budget: 1800 s per run; the current model takes seconds. There is room for
  seed-bagging and small ensembles, not for refitting the developed market.
- Never import glum or tabmat (OpenMP clash: segfault, exit 139). Use scikit-learn GLMs.
- CatBoost: `allow_writing_files=False`, and categorical NaN as a string ("missing").
- Use only installed open-source libraries: lightgbm, catboost, xgboost, interpret (EBM),
  scikit-learn, numpy, pandas. No installs, no network.
- Predictions must be finite and non-negative for every valid row.
- When every backlog item is tried, parked or owner-only and the ledger suggests no
  follow-up, finish with exactly `HYPOTHESIS: NONE` and make no edit. The loop stops.

## Dataset columns (emerging)

policy: exposure, voluntary_excess, compulsory_excess, channel, payment_method,
missed_payments_12m, new_business, renewal_count, discount_applied, start_month;
motor: driver_age_band, years_licensed, ncb_years, vehicle_value_group, vehicle_age,
make_model, engine_size_cc, body_type, transmission, seats, vehicle_colour, fuel_type,
driver_assist, cover_type, overnight_parking, vehicle_use, named_young_driver;
buildings: policyholder_age_band, sum_insured, construction_type, occupancy,
days_unoccupied, smoke_alarms, accidental_damage_cover, mortgage;
contents: policyholder_age_band, sum_insured, tenure, security_level, high_value_items,
accidental_damage_cover, works_from_home, lodgers, bicycle_cover;
customer: occupation (20 titles), self_employed, marital_status, household_size,
prior_claims_5y, licence_points, years_at_address, document_preference, previous_insurer;
address: region (E01-E06), urban_density, flood_zone, distance_to_river_km, elevation_m,
soil_type, year_built, property_type, roof_type, basement, trees_within_5m,
listed_building, bedrooms, energy_rating, solar_panels, property_age;
weather (the row's own year): hail_days, wind_exposure_index, freezing_days,
dry_summer_index, wildfire_index, rainfall_mm, mean_temperature_c;
derived: customer_tenure, hist_n_policy, hist_cost_policy, lines_held; `base`.
Line-specific columns are NaN on other lines. new_business and similar flags are
categoricals with string levels "True"/"False".
