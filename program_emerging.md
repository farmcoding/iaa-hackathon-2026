# program_emerging.md: instructions for the Part 2 experiment agent

You are running one experiment in an experiment loop for an actuarial pricing model in a
new market. A human actuary (FCIA) owns this work, sets the direction in this file and
approves every leaderboard submission. You propose and test; you never submit.

## The task

Predict expected ultimate net claim cost per 2025 policy row for motor, buildings and
contents in the **emerging market**. The insurer wrote there in 2023 and 2024; the 2025 book
is a broker transfer, 76% new business (14% in history), younger. Score: Tweedie D² with
power 1.5, pooled across lines.

What is known (reports/part2_drift.md; Part 2 section of reports/assumptions_log.md):
- The developed-market model, refitted per peril, is the best emerging model so far: the
  frames carry one developed prediction per peril, `base_<peril>`, and their sum (`P0`) is
  the bar (board 0.15195). `base` is the total-cost version.
- **What failed, so don't repeat it:** free boosted corrections on top of the base; one-way
  credibility factors; and especially *stacking* several small factors that each missed the
  margin: on 25 Oct night eleven stacked factors gained +0.0068 on the folds and lost 0.0029
  on the leaderboard (they were all chosen on the same claims they were scored on).
- Owner decision (25 Oct): **gender is admissible** in this market. The developed base has no
  gender, so any gender effect must come from emerging rows. Diagnostics show the base
  misprices men/women by 12-17% on motor in both years.

## What you may change

Only `src/iaa/experiment_emerging.py`. Contract in its docstring:
`fit_predict(train, valid) -> np.ndarray`. Everything else is frozen (data, target,
development, trend, folds, base predictions, the level step, scoring, harness, tests,
config); any other edited file is reverted and logged as a violation.

**The level is not yours.** After `fit_predict` returns, the harness rescales each line so
its training-row predictions sum to the trended training actuals (`valid` may therefore
contain training rows). A constant factor per line does nothing.

## The keep rule (frozen; declared 25 Oct midday)

Per fold, end-to-end D² at the production level: 2/3 on the natural book + 1/3 on the book
reweighted to look like the 2025 broker book. Primary = mean of fold B (train 2023, score
2024) and fold R (train 2024, score 2023). Keep only if ALL hold:
- primary beats the bar by more than **0.002**;
- neither fold falls by more than 0.003; level-neutral fold B (`d2_ln_B`) falls by no more
  than 0.002;
- a paired bootstrap of the gain over the bar has its 10th percentile above zero;
- a rerun with shifted seeds passes again (deterministic methods pass on exact reproduction).

## Discipline (new; these are why the rule changed)

1. **Pre-registered list only**: take the highest open item in
   `research/experiment_backlog_emerging.md` (tier L). Do not invent new factors.
2. **One mechanism per run, at most three settings per mechanism** (e.g. alpha 0.001 / 0.01
   / 0.1). Do not re-run a mechanism with a fourth setting.
3. **No stacking.** Never combine two discarded runs to clear the margin. Combining kept
   mechanisms is the owner's job.
4. **Gate every factor.** Before a factor enters a model, it must pass
   `lr_gate(train_line, col, n_tests=N)` on the training rows inside `fit_predict` (N = the
   number of factors in the item's pre-registered list); a factor that fails is left out
   *in that fit*. Use the helpers, don't re-implement them:
   - `offset_glm(tr, va, cols, alpha, interactions=())`: ridge log-link Tweedie GLM on top
     of an offset column (default `base`; pass `offset=` for another), exact for p=1.5.
     Returns the multiplier for `va` rows.
   - `lr_gate(tr, col, n_tests)`: claim-count likelihood-ratio test against the offset.
   - `peril_levels`, `peril_prediction`, `perils_of`: per-peril bases.
   - To use P0 as the offset, put it in a column first:
     `tr = tr.assign(p0=peril_prediction(tr, line, ones))` and pass `offset="p0"`.
5. Say in the HYPOTHESIS line which backlog item and which setting you are running.

## One iteration

1. Read `data/cache/diagnostics.md` (segment A/E of the bar, level-neutral, folds B and R),
   `results_emerging.tsv` (all runs; rows before the 25 Oct midday owner keep used other
   metrics and are not comparable), the backlog, and the current experiment file.
2. Pick the next item and setting per the discipline above.
3. Edit `src/iaa/experiment_emerging.py` (small, readable diff).
4. Do not run anything; the driver runs the harness and commits or reverts.
5. Finish with a final message whose first line is exactly:
   HYPOTHESIS: <item, setting, why>
   then:
   CHANGE: <what you edited>
   EXPECTED: <which fold/line should move, and why>

## Rules

- **Inadmissible, never as a feature or factor** (owner): written_premium, email_domain,
  campaign_code, marketing_segment, branch_id, days_quote_to_inception, marketing_consent,
  hist_n_customer, hist_cost_customer; licence_points on buildings and contents.
- **Leakage**: never renewal_decision, customer_service_contacts, any target column
  (cost_*, n_*, ult_*, comp_*, y, y_ult), trend_factor or the row's year. Counts n_* may be
  used on *training rows* (the gate does), never read from valid.
- **The bases are fixed**: use them as offsets or blend members; never refit the developed
  market here.
- Never import glum or tabmat (OpenMP clash, exit 139). scikit-learn GLMs only.
- CatBoost: `allow_writing_files=False`; categorical NaN as a string.
- Installed open-source libraries only; no installs, no network.
- Predictions finite and non-negative for every valid row.
- When every tier L item is done, finish with exactly `HYPOTHESIS: NONE` and make no edit.

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
per-peril bases (`base_<name>`, NaN on other lines): motor base_bodily_injury,
base_own_damage, base_third_party_prop, base_theft, base_glass_hail; buildings base_storm,
base_flood, base_subsidence, base_water, base_fire, base_ad; contents base_theft,
base_escape_of_water, base_accidental_damage, base_flood (shared column names across lines
hold each line's own peril). Training rows also carry cost_<peril>, n_<peril> and
comp_<group> for targets; see the peril helpers in experiment_emerging.py.
Line-specific columns are NaN on other lines. new_business and similar flags are
categoricals with string levels "True"/"False".
