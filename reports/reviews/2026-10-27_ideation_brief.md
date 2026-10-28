# Brief for external idea generation and adversarial review (27 Oct 2026)

You are advising the owner of a solo entry (a qualified actuary, FCIA) in a Kaggle
actuarial pricing competition. The goal for the next 48 hours is the largest **honest**
gain in predictive power on the Part 2 test book: a public-leaderboard top-5 place that
also holds on the private split. You see aggregates and code structure only; no row-level
data leaves the owner's machine.

## 1. The task

- Synthetic insurer data, three lines: private motor, home buildings, home contents.
  Predict, for each 2025 policy row, the expected ultimate claim cost: all claims with a
  loss date in the row's period, net of excess and recoveries, including IBNR, excluding
  declined claims. The prediction is a row total (exposure = years on cover, rows can be
  part-year).
- Metric: Tweedie deviance D² with power 1.5, pooled across the three lines, relative to a
  constant baseline. Higher is better.
- Claims in both markets are valued at 31 Dec 2024. Open claims carry case estimates;
  recoveries are booked only when a claim settles.
- **Part 1 (closed)**: developed market, policy years 2021-24, ~3.0M policy rows, ~160k
  claims. Our final: private 0.13363 (rank 31/51; leader 0.13923).
- **Part 2 (now; board closes Thu 30 Oct 23:59 EDT)**: the insurer wrote in an emerging
  market in 2023-24 (~65k policy rows; per year roughly motor 3,500 claims, buildings 500,
  contents 500). At 1 Jan 2025 it takes on a broker portfolio, so the 2025 test book
  (209k rows) is 76% new business (14% in history) and younger (drivers 17-34: 50% of
  exposure vs 39%). The developed data is available for training.
- Emerging schema differences: no date of birth (age bands 17-24/25-34/35-49/50-64/65+),
  no credit band, no annual mileage, no telematics; gender present and admissible in this
  market; occupation free text (mapped to 20 titles); different construction codes; ~12%
  missing on several fields (2.5% developed). Two training fields leak or are train-only
  and are excluded (customer_service_contacts, renewal_decision).
- Other admissibility exclusions (owner decisions, re-tested 25 Oct, all lowered or did not
  change the score): written premium, email domain, campaign code, marketing segment,
  branch, quote lead time, marketing consent, customer-level claim history; licence points
  on property.
- Judging: 40% pricing performance and adaptation (private leaderboards of both parts plus
  the panel's view), 25% effective use of AI, 20% validation and governance, 15% video,
  report and repo. Ten submissions a day.

Fields available per row (emerging): driver_age_band, years_licensed, ncb_years,
vehicle_value_group, vehicle_age, make_model, engine_size_cc, body_type, transmission,
seats, vehicle_colour, fuel_type, driver_assist, cover_type, overnight_parking,
vehicle_use, named_young_driver, voluntary_excess, compulsory_excess, channel,
payment_method, missed_payments_12m, new_business, renewal_count, discount_applied,
policyholder_age_band, sum_insured, construction_type, occupancy, days_unoccupied,
smoke_alarms, accidental_damage_cover, mortgage, tenure, security_level,
high_value_items, works_from_home, lodgers, bicycle_cover, gender, occupation,
marital_status, household_size, prior_claims_5y, licence_points, years_at_address,
document_preference, previous_insurer, self_employed, region (E01-E06), urban_density,
flood_zone, distance_to_river_km, elevation_m, soil_type, year_built, property_type,
roof_type, basement, trees_within_5m, listed_building, bedrooms, energy_rating,
solar_panels, hail_days, wind_exposure_index, freezing_days, dry_summer_index,
wildfire_index, rainfall_mm, mean_temperature_c (weather fields are the policy year's
realised values, also given for 2025), customer_tenure, property_age, start_month,
hist_n_policy, hist_cost_policy (policy's own prior claims), lines_held. Claims carry
peril (motor: bodily injury, own damage, third-party property, theft, glass/hail;
buildings: storm, flood, subsidence, escape of water, fire, accidental damage; contents:
theft, escape of water, accidental damage, flood), loss/report dates, status, paid,
incurred, recoveries.

## 2. What is known about the emerging market

- Same-maturity ratios emerging/developed: severity 0.6-0.8 on most perils, bodily injury
  0.45-0.62 (largest emerging BI claim 156k vs 11.9m developed); motor frequency 1.2-1.8x
  (glass/hail 1.55-1.84x, third-party property 1.4x, BI 1.37-1.54x); buildings storm
  1.3-1.7x; escape of water 0.6-0.7x on both property lines; fire 2.3x in 2023 and 0.84x in
  2024 (49 vs 21 claims).
- Climate: hail days 14-17 vs ~6 developed, wildfire index 0.5 vs 0.15, rainfall 1,380 vs
  1,040 mm; 2025 emerging weather milder (hail 11.4).
- Adversarial validation: emerging 2023-24 vs 2025 separable (AUC 0.99), mostly weather
  year and tenure; renewal vs new business within 2025: AUC 0.74 motor (make_model,
  previous_insurer, years_licensed), 0.63 property.
- New business claims experience on the current model: motor claim-count A/E 1.20 (2023) /
  1.07 (2024); contents 1.18 / 1.43; buildings 1.06 / 0.69.
- Where the current model misses (A/E actual/predicted, both validation folds the same way,
  each >= 2% of exposure): motor deciles of predicted rate are not monotone in A/E (fold B
  decile 1 0.44, deciles 4-5 1.34-1.37, decile 8 0.73; fold R deciles 7 1.21, 9 0.78).
  Motor segments: previous_insurer levels 0.47-1.69, missed_payments_12m=1 1.18-1.23,
  elevation 0-3 m 1.11-1.19 vs 15-60 m 0.80-0.83, diesel 1.11-1.13, third party fire and
  theft cover 0.86, region E01 1.10-1.19, licence points 3: 1.12-1.25, occupation Doctor
  1.15-1.38 / Retail assistant 0.72-0.80. Buildings: non-standard construction 1.39-1.46,
  energy rating E 1.51-1.84 vs A 0.58-0.73, trees within 5 m 1.32-1.64, flood zone 3
  1.29-1.61, missed payments 1.36-2.05, years at address 0-3 1.25-1.28 (buildings has
  ~500 claims a year, so single-segment A/E is noisy).

## 3. Current Part 2 pipeline

1. **Target**: developed (chain-ladder) ultimate per row: reported net cost per peril group
   divided by the group's completion at the row's loss months, using report-delay patterns
   borrowed from the developed market (both markets are at identical maturity at the
   valuation date). Trends carried over from the developed market: motor 3%, contents 5%,
   buildings by peril 9/9/4/4%. Sensitivity checked 27 Oct: case-reserve adequacy +/-10%
   moves motor's level 3.4%; BI tail +/-5% moves motor 1.9%.
2. **Bases**, fixed per row and attached by the harness, all fitted on other data so a
   fold never trains on the rows it scores:
   - `base_<peril>` (P0 = their sum): the Part 1 final specification (LightGBM + EBM +
     CatBoost blends, credibility target encoding, monotone constraints, Part 1 feature
     pruning) refitted per peril on developed 2021-23 (folds) / 2021-24 (2025). Emerging
     rows are pushed through it with schema mapping (band -> in-band mean age; credit,
     mileage, telematics missing).
   - `base_pooled`: the same specification fitted on developed + emerging rows with a
     market flag (emerging rows weight 1; weighting at 5 was done by duplicating rows,
     which let the trees memorise claims).
   - `base_compat`: developed refit without credit, mileage, telematics, banded ages.
3. **Emerging adaptation** (`fit_predict(train, valid)` in one experiment file, ~380 lines):
   per-peril emerging levels with credibility k=100 towards the line; contents blended
   50:50 with the pooled base; motor gender as a ridge log-link Tweedie offset GLM
   (alpha 0.01) gated by a claim-count likelihood-ratio test; a new-business claim-count
   credibility factor (k=200) on motor and contents.
4. **Level**: the harness rescales each line so predictions on the training rows sum to
   the trended training actuals ("balance on train"); the 2025 mix then moves the level
   through the model's relativities. A Tweedie-MLE scalar instead was inside noise.
5. **Validation**: fold B trains on emerging 2023 and scores 2024; fold R trains on 2024
   and scores 2023 (not out of time, a second read). Primary = mean over B and R of
   end-to-end D² at the production level, 2/3 on the natural book + 1/3 on the book
   reweighted to look like 2025 (classifier odds; effective sample size 15-23%). Keep rule:
   primary gain > 0.002 over the current best, neither fold down > 0.003, level-neutral
   fold B not down > 0.002, paired bootstrap 10th percentile of the gain > 0, deterministic
   reproduction. Public-board noise between different models ~0.002 (sd).
6. **Automated loop**: an LLM agent edits only the experiment file, one pre-registered
   mechanism per run, at most three settings, no stacking of discarded runs, every factor
   gated. The driver runs the frozen harness (a run takes seconds to a few minutes; the
   emerging frame is small). Base families (developed or pooled refits) are owner scripts:
   each loads the 3M-row developed frame (7-11 GB) and takes ~10-40 min on the Mac.

## 4. Results so far (Part 2)

| Model | Primary (deploy) | Public LB |
|---|---|---|
| one rate per line | | 0.08396 |
| Part 1 spec refit on emerging only | | 0.12904 |
| developed base (total), historical level | | 0.14243 |
| developed base, level from 2025 mix index | | 0.15079 |
| P0 per-peril bases, balance on train | 0.11554 | 0.15195 |
| P0 + motor peril levels + 11 stacked credibility factors | | 0.14907 |
| L1: P0 + motor gender GLM | 0.11910 | 0.15933 |
| **O4** (current best, section 3) | **0.12217** | **0.16326** |

Leaderboard 27 Oct (26 teams): 1st 0.17303, 2nd 0.17172, 3rd 0.16914, 4th 0.16836,
5th 0.16662, ..., ours 9th at 0.16326. Fifth needs about +0.0034, first +0.0098.

Tried and lost (25 Oct, on these folds): free boosted LightGBM corrections on top of the
base (offset log base, 25-300 trees, min leaf 300-3000 rows); one-way credibility factors
on cost; stacking several sub-margin factors (+0.0068 folds, -0.0029 board); joint GLMs on
the base with 6-8 gated factors each (extra factors diluted gender); single gated factors
cover_type, occupation, contents tenure, gender by motor peril group (all < margin); a
hail-days elasticity on hail perils; schema-compatible base alone (0.11622) or 50:50 with
P0 (0.11792); pooled base alone (0.0998-0.1089); geometric blend of total and per-peril
bases; emerging-only refits. Part 1 (developed) also found no gain from TabM / RealMLP
neural blends, spline GLM blends (+0.0006) or isotonic recalibration.

## 5. Resources

- Mac mini M4, 16 GB RAM, ~24 GB free disk. **Memory is the binding constraint**: only one
  developed-frame process at a time; three at once crashed the machine on 25 Oct. Emerging-
  only experiments are light.
- Kaggle notebooks, unused so far in Part 2: CPU sessions (4 cores, ~30 GB RAM, 12 h),
  GPU (T4 x2 or P100, ~30 GPU-hours a week), the competition data attached, internet on.
- About 40 hours of wall clock before the board closes, including two nights for
  automated runs; the human reviews each submission.

## 6. What we want from you

1. **Constraints**: rank the constraints in sections 3-5 (modelling, validation,
   governance, compute) by how much predictive power each is likely costing, with your
   reasoning and a rough size. Say which you would relax and how to relax it safely.
2. **White space**: which modelling disciplines and methods have not been used here at all
   (transparent: GLM/GAM, credibility, Bayesian hierarchical, frequency-severity; black-box:
   boosting variants, neural tabular models, transfer learning / fine-tuning, domain
   adaptation and covariate-shift weighting, multi-task models, stacking, distillation,
   AutoML or bandit-style search; anything else) and which of them fit this problem.
3. **Ideas**: 10-15 concrete, testable ideas, ranked by expected gain on the 2025 book.
   For each give: a short name; the mechanism; why it should transfer to the 2025 broker
   book (not just to the two folds); the expected primary gain; how it fits the harness
   (a change inside `fit_predict` on the fixed bases and emerging rows, OR a new base family
   fitted on developed/pooled data, which needs a heavy refit); compute and where to run it
   (Mac, Kaggle CPU, Kaggle GPU) with a time estimate; the overfitting risk and how to test
   it within the keep rule. Prefer ideas with a clear mechanism over generic "tune more".
4. **The weakest spot** in the analysis so far: where you think the most lift is being left
   on the table and why.

Answer in Markdown. End with a JSON array `ideas` (one object per idea with keys name,
mechanism, expected_gain, harness_fit, compute, where, hours, overfit_risk, test) inside a
```json fenced block.
