
# Model evidence pack: IAA hackathon motor/buildings/contents pricing model

Scope: champion run `20261020T071548` (commit `bd49f85`, kept as `keep_owner`, primary
0.1426, fold A D2 0.14256, fold B D2 0.14262). Read-only against
`~/Projects/iaa-hackathon-2026`; all scripts and outputs live under
`~/.cache/iaa-owner/1020/evidence/`. A GLM baseline (`t31_glm.parquet`) landed partway
through this work and is included in items 1 and 2 below.

## What this means for the report

- The champion's lead over the GLM and every prior kept run holds up under a policy-level
  bootstrap on fold B (the fold the harness weights double); on fold A, the lead over the
  GLM only clears the noise floor once the single 11.9m bodily-injury claim is capped,
  confirming that claim (not a general ranking weakness) causes the wide fold A interval.
- Scoring on the reported-to-date scale (no development assumption needed) gives the same
  ranking as the leaderboard's developed-ultimate scale in both folds, so the champion's
  lead is not an artefact of the chain-ladder or frequency-severity assumptions.
- Once region and weather fields are set aside, 2025 looks almost exactly like the
  training years on motor and contents (adversarial AUC drops to 0.53-0.63, near
  chance). Buildings keeps a real gap (AUC 0.89-0.96), but it traces to property_age/
  year_built, which age by construction every year, not to a change in risk mix.
- One structural feature break: `construction_type` (buildings) shows the "T"/timber
  level present in roughly 8% of training rows but absent entirely from the 2025 test
  file, with "W"/wood's share roughly tripling (4% to 11%) to compensate — a probable
  relabelling between extracts, not organic drift, worth a direct check against the raw
  file before relying on this feature in the final fit.
- The proxy-association sweep below extends the fairness review to every customer-table
  field. The one new finding: `marketing_segment` (already dropped from the model) is
  extremely close to age and credit_band (association 0.66-1.00) — stronger redundancy
  than the review's comment on it implied. `email_domain` stays independent of the full
  current feature set on every line.

## 0. Run map

`results.tsv` has 164 rows; 136 carry a `run_id` with a matching saved predictions
parquet under `data/cache/preds/`. Every mapped row's `d2_A`/`d2_B` was recomputed
directly from its parquet with `iaa.score.d2` and matches the ledger to 4 decimal places
(0 mismatches): the run_id -> parquet mapping is sound as a basis for everything below.
Full mapping: `out/run_map.csv`.

The earliest 28 ledger rows predate the `run_id` column; the baseline run used here is
the earliest saved predictions file by filename (`20261018T013945`, ledger timestamp
2026-10-18T01:39:46, "per-line LightGBM tweedie 1.5 (3-seed bag)..."), which is the first
kept row in the ledger.

Milestones carried through items 1-2: regularization (`3e8a968`), the motor BI
frequency/severity split (`3d91e15`), out-of-fold credibility encoding (`a816f58`), and
the owner's reproducibility rebaseline (`2eb383c`). `out/run_summary.json` has the
exact run IDs.

Note on `FROZEN_LINES`: the champion's buildings and contents feature engineering has
been frozen to the run being beaten since the owner's 2026-10-19 meta-review (motor is
where the gap is), so recent motor-only experiments reuse earlier buildings/contents
predictions verbatim. This does not affect the comparisons below, which always compare
full saved prediction sets against each other, but it means a per-line diff that is
exactly zero on buildings or contents between two adjacent runs reflects the freeze, not
a null result.

## 1. Reported-to-date scale check (backlog 3.6)

The leaderboard scores against `y_ult`, the chain-ladder/frequency-severity developed
ultimate from `iaa.ultimate`. This check asks whether the same ranking holds against
`cost_total`, the reported net cost to the valuation date, a figure that needs no
development assumption at all.

Construction: every row's `comp_<group>` columns give the completion fraction (reported
/ ultimate) for its line's peril groups, based only on the row's own period dates and the
line's development pattern, independent of whether the row had a claim. A row's overall
completion weight is `c_row = sum_g s_g * comp_g(row)`, where `s_g` is group g's share of
the line's total developed ultimate, fit separately per fold from that fold's own
training years only (fold A: 2021-22, fold B: 2021-23), so no valuation-year information
leaks into the weight. The identity `sum_g ult_g * comp_g == cost_total` holds exactly
over the full training set (verified: ratio 1.000000, `out/step1_reconciliation.json`),
confirming the columns are being read correctly.

The check: D2 of `cost_total` against `pred x c_row`, constant baseline, same Tweedie
power as the leaderboard.

| fold | run | D2 developed scale | D2 reported scale | sum(pred x completion) / sum(cost_total) |
|---|---|---|---|---|
| A | champion | 0.14256 | 0.14142 | 1.0035 |
| A | GLM | 0.13280 | 0.13131 | 1.0019 |
| A | credibility encoding | 0.12919 | 0.12804 | 0.9964 |
| A | owner rebaseline | 0.12796 | 0.12683 | 0.9964 |
| A | BI freq/severity split | 0.11600 | 0.11513 | 0.9964 |
| A | regularization | 0.11383 | 0.11281 | 0.9963 |
| A | baseline | 0.08998 | 0.08951 | 0.9963 |
| B | champion | 0.14262 | 0.13505 | 0.9765 |
| B | GLM | 0.12377 | 0.11576 | 0.9868 |
| B | credibility encoding | 0.13619 | 0.12834 | 0.9796 |
| B | owner rebaseline | 0.13550 | 0.12754 | 0.9796 |
| B | BI freq/severity split | 0.13199 | 0.12330 | 0.9810 |
| B | regularization | 0.12644 | 0.11838 | 0.9784 |
| B | baseline | 0.12120 | 0.11315 | 0.9785 |

Full table with per-line breakdowns: `step1_reported_scale.csv`.

The ranking is identical on both scales in both folds, and the champion's lead over
every comparator survives. Reported-scale D2 runs below developed-scale D2 throughout,
which is expected since the reported cost is noisier (fewer peril groups have fully
emerged), but the gap is similar in size across runs, so it is not concentrating on one
model over another. The predicted-reported / actual-reported ratio sits in a 0.976-1.004
band for every run, with no consistent pattern favouring the champion over the
comparators; this is a validation of the scale-conversion method (confirming predictions
land close to reported-to-date actuals once completion-weighted) rather than a finding
about any one model's reserving posture.

## 2. Champion vs comparators (backlog 3.24)

The GLM comparator is `t31_glm` (`iaa.experiment` module `exp_glm`): a per-line GLM fit,
primary 0.1268 against the champion's 0.1426, fold B D2 by line 0.065/0.097/0.074 motor/
buildings/contents against the champion's 0.086/0.116/0.095.

Cluster bootstrap resamples whole policies rather than individual rows, so that any
within-policy correlation a row-level resample would miss (same vehicle/address rated
more than once, or any residual intra-policy dependence) is preserved. Each fold here is
a single validation year, so there is no cross-year dependence to capture; with roughly
590,000 distinct policies against 626,000 rows per fold, the policy and row bootstraps
are close, and the result below should be read as a near-row-level bootstrap rather than
a strong correction for clustering. 500 reps, 2.5/97.5 percentile CI:

| fold | comparator | D2 champion | D2 comparator | mean diff | 95% CI | champion wins |
|---|---|---|---|---|---|---|
| A | baseline | 0.1420 | 0.0911 | 0.0509 | [0.0281, 0.0915] | 100% |
| A | regularization | 0.1419 | 0.1135 | 0.0284 | [0.0196, 0.0425] | 100% |
| A | BI freq/severity split | 0.1418 | 0.1153 | 0.0265 | [0.0157, 0.0430] | 100% |
| A | credibility encoding | 0.1420 | 0.1287 | 0.0133 | [0.0100, 0.0170] | 100% |
| A | owner rebaseline | 0.1422 | 0.1275 | 0.0147 | [0.0113, 0.0185] | 100% |
| A | GLM | 0.1418 | 0.1318 | 0.0101 | [-0.0017, 0.0174] | 95% |
| B | baseline | 0.1425 | 0.1210 | 0.0215 | [0.0177, 0.0254] | 100% |
| B | regularization | 0.1424 | 0.1261 | 0.0163 | [0.0127, 0.0204] | 100% |
| B | BI freq/severity split | 0.1429 | 0.1323 | 0.0106 | [0.0078, 0.0136] | 100% |
| B | credibility encoding | 0.1425 | 0.1362 | 0.0064 | [0.0039, 0.0090] | 100% |
| B | owner rebaseline | 0.1425 | 0.1353 | 0.0072 | [0.0050, 0.0098] | 100% |
| B | GLM | 0.1425 | 0.1236 | 0.0189 | [0.0142, 0.0246] | 100% |

Full table: `step2_bootstrap_ci.csv`. Fold A vs GLM is the one comparison where the
interval touches zero. Fold A's intervals are also 3-5x wider than fold B's throughout
(e.g. vs baseline: [0.028, 0.092] on A against [0.018, 0.025] on B) — both effects trace
to the same cause, a single 11.9m motor bodily-injury claim in accident year 2023 that
fold A's validation year (2023) carries and fold B's (2024) does not. Re-running the
fold A bootstrap with that claim's `y` capped at `BI_CAP` (1,000,000, the same cap
`iaa.experiment` applies to its own training target) narrows champion-vs-GLM to
[0.0120, 0.0177] — a clean win, confirming the one claim, not a general ranking
weakness, drove the uncapped interval across zero (`step2_foldA_capped_bootstrap.csv`).
Fold B, which the harness weights double and which tracks the public leaderboard, shows
a clear champion win against every comparator including the GLM without any adjustment.

Double-lift charts (`double_lift_champion_vs_baseline.png`,
`double_lift_champion_vs_glm.png`): rows sorted into exposure-weighted deciles by
the ratio of the two models' predicted rates, fold B. In both charts the champion's rate
line tracks the actual loss-cost line within about 14% at its worst decile, while the
comparator (baseline or GLM) over-predicts its cheapest-ranked decile and
under-predicts its priciest by a wide margin — against the GLM, 638 predicted per
exposure-unit where actual is 319 (roughly double) in the decile it ranks cheapest, and
337 predicted where actual is 666 (roughly half) in the decile it ranks priciest.

Decile A/E tables by line, both folds: `step2_decile_ae.csv`; chart (fold B, all
three lines): `decile_ae_fold_B.png`. One standout: motor fold A decile 7 runs at
A/E 2.00 (actual 23.0m vs predicted 11.5m), the same large bodily-injury claim
discussed above concentrated in that decile; every other motor/buildings/contents
decile in both folds sits in a 0.76-1.16 band.

Renewal vs new-business split (champion, fold B; new business defined as the first year
a `policy_id` appears across the full 2021-2025 history, not the raw `new_business`
flag, which only distinguishes within a single underwriting year):

| line | segment | exposure | D2 | A/E |
|---|---|---|---|---|
| motor | renewal | 181,723 | 0.0784 | 1.003 |
| motor | new business | 18,063 | 0.1209 | 1.077 |
| buildings | renewal | 136,449 | 0.1068 | 1.013 |
| buildings | new business | 13,469 | 0.1542 | 1.154 |
| contents | renewal | 136,500 | 0.0831 | 1.002 |
| contents | new business | 13,467 | 0.1282 | 0.945 |

New business is a tenth of exposure on every line, scores a higher D2 (the segment has
more spread to explain), and carries an A/E above 1 on motor and buildings (the champion
under-predicts new business there by 5-15%), consistent with the backlog's long-standing
note that new-business motor is a persistent miss the `ncb_years_new` split was meant to
address, and suggesting the same gap exists on buildings. Cross-check: the raw
`new_business` flag marks 15.4% of 2024 motor rows as new, well above this derived
definition's 9.0%; the two definitions disagree because the flag can mark a policy that
previously held cover elsewhere or lapsed and returned, not only a policy's first-ever
appearance in this book. The derived, policy-history definition used above is the
stricter and more defensible one for this test.

## 3. Drift and adversarial validation (backlog 3.7)

PSI (train years 2021-23 vs 2025 test), per feature per line, full table
`step3_psi.csv`. Above the conventional 0.25 "material" threshold:

| line | feature | PSI |
|---|---|---|
| buildings | construction_type | 0.948 |
| buildings | dry_summer_index | 0.402 |
| motor | dry_summer_index | 0.399 |
| motor | mean_temperature_c | 0.292 |
| buildings | mean_temperature_c | 0.291 |
| motor | freezing_days | 0.276 |
| contents | freezing_days | 0.268 |

`construction_type`'s PSI is far above every weather field and is not explained by
calendar drift (its levels are fixed at policy inception). Checking the level shares
directly: the "T"/"t" (timber) levels together hold about 7.8% of 2021-23 training rows
and 0.0% of the 2025 test rows — they do not appear at all — while "W"/"w" (wood) rises
from about 4.3% to 11.5% over the same comparison, nearly the same share that vanished
from the T levels. Every other level (Brick, BRICK, brick, Stone, stone, non_standard)
holds its share within half a point. This reads as a relabelling of the timber category
into the wood category between the training and 2025 extracts, not organic drift, and
means a buildings model trained on "T" rows will route 2025's former-"T" risks through
whatever split point its trees gave "W" instead — worth confirming with whoever produced
the 2025 extract before the final fit relies on this feature. The weather-field PSIs
(dry_summer_index, mean_temperature_c, freezing_days, rainfall_mm, wildfire_index)
reflect 2025 genuinely being a different weather year, which is expected and already a
first-class model input on every line.

Adversarial AUC (LightGBM, 2 threads, 3-fold, line's current feature set):

| line | 2024 vs 2021-23 | 2025 test vs 2024 | notime, 2024 vs 2021-23 | notime, 2025 vs 2024 |
|---|---|---|---|---|
| motor | 0.967 | 0.990 | 0.940 | 0.990 |
| buildings | 0.969 | 0.985 | 0.951 | 0.985 |
| contents | 0.887 | 0.983 | 0.887 | 0.983 |

Full table incl. top features: `step3_adversarial.csv`; chart:
`adversarial_auc.png`. All four AUCs above are high mainly because `region` and the
weather fields are almost perfectly separable by calendar year, since weather is joined
per address-year — that answers "can a classifier tell the years apart" but not the
question the task actually asks ("does 2025 look like the training years"). A follow-up
pass drops `region` and every weather field (`freezing_days`, `dry_summer_index`,
`wildfire_index`, `rainfall_mm`, `mean_temperature_c`, `wind_exposure_index`,
`flood_zone`, `soil_type`, `distance_to_river_km`, `elevation_m`) on top of the
time-trending fields already excluded (full table: `step3_adversarial_noweather.csv`):

| line | 2024 vs 2021-23, no weather/region | 2025 test vs 2024, no weather/region |
|---|---|---|
| motor | 0.628 | 0.546 |
| buildings | 0.960 | 0.886 |
| contents | 0.554 | 0.526 |

With weather and region out, motor and contents collapse to near chance (0.53-0.63):
on those two lines, 2025's mix of risk looks essentially like the training years once
the genuinely weather-driven features are set aside, and the adversarial AUC in the
first table is a weather/region calendar effect, not a sign of population drift. Motor's
top remaining feature is `make_model`, consistent with ordinary year-to-year model-mix
turnover rather than anything structural. Buildings keeps a real gap (0.89-0.96), driven
almost entirely by `property_age` and `year_built`: since property age increases by
exactly one every calendar year while year_built is fixed, a classifier can separate
years from that pair alone without any change in the underlying property population —
a mechanical effect of the fields' definitions, not drift to act on.

## 4. Proxy-strength extension of the fairness review (backlog 3.47)

Swept every customer-table field (16 fields, including `email_domain` and
`marketing_segment`, both dropped from the model) against every feature currently in
`iaa.experiment._feature_cols` for that line: 1,415 pairs checked across the three
lines, 300k-row sample per line, bias-corrected Cramer's V (category-category),
correlation ratio (category-number), or absolute Spearman (number-number). Full matrix:
`step4_proxy_matrix.csv`; pairs above 0.5: `step4_proxy_matrix_flagged.csv`.

15 pairs clear 0.5, all of them age- or marketing-segment-driven and already implied by
the fairness review's finding that age is reconstructable from its correlated columns:

| line | customer field | model feature | association | method |
|---|---|---|---|---|
| motor | customer_age | driver_age | 0.9999 | Spearman |
| contents | customer_age | policyholder_age | 0.9999 | Spearman |
| motor | customer_age | years_licensed | 0.9920 | Spearman |
| motor | marketing_segment | driver_age | 0.9676 | corr. ratio |
| contents | marketing_segment | policyholder_age | 0.9676 | corr. ratio |
| contents | marketing_segment | customer_age | 0.9675 | corr. ratio |
| motor | marketing_segment | years_licensed | 0.9626 | corr. ratio |
| motor | marketing_segment | credit_band | 0.7071 | Cramer's V |
| contents | marketing_segment | credit_band | 0.7071 | Cramer's V |
| buildings | marketing_segment | credit_band | 0.7071 | Cramer's V |
| buildings | customer_age | occupation | 0.6609 | corr. ratio |
| motor | occupation | driver_age | 0.6605 | corr. ratio |
| contents | occupation | policyholder_age | 0.6603 | corr. ratio |
| contents | occupation | customer_age | 0.6602 | corr. ratio |
| motor | occupation | years_licensed | 0.6557 | corr. ratio |

`marketing_segment` is the new finding this sweep adds beyond the fairness review's
scope. It is dropped from every line's model (`experiment.py`'s own comment calls it "a
relabel of age and credit band"), and this is the first place that claim is measured
rather than asserted: association with credit_band is exactly 0.7071 on all three
lines, and with age proxies above 0.96 on motor and contents. The identical 0.7071 value
on all three lines to four decimal places is a signature of a deterministic, low-level
lookup between the two fields (both reduce to a small number of levels) rather than a
noisy sampling coincidence. The drop looks correct, and this evidence shows the
redundancy is stronger than the code comment's wording implies.

`occupation` repeats the fairness review's line-dependent pattern seen there: moderate
association with age on motor and contents (0.655-0.661, consistent with the review's
age-relabel reading) but it does not reach 0.5 against anything on buildings, where
occupation is in the model and the review already found it carries independent content.

No other customer field (`household_size`, `prior_claims_5y`, `licence_points`,
`previous_insurer`, `document_preference`, `marketing_consent`, `campaign_code`,
`branch_id`) clears 0.5 against anything in any line's feature set.

`email_domain` (previously shown independent in the fairness review, checked there
against a handful of fields) stays independent against the full current feature set on
every line: maximum association 0.0266 (full detail: `step4_email_domain.csv`),
against features such as occupation, voluntary_excess, distance_to_river_km and
renewal_count. The 2026-10-19 finding holds under a much wider comparison set.

The next tier down, just under the flag line, is informative in a different way:
`credit_band` associates with `missed_payments_12m` at 0.42-0.43 on every line; the
fairness review's own reading, "a real payment-behaviour proxy, not a demographic one,"
is unchanged by this wider sweep. `customer_since_year` was requested as a candidate
field but does not reach this dataset: it is dropped upstream in `features.FORBIDDEN`
because it drifts between train and test by construction, so it was excluded from the
sweep rather than silently scored as independent.

## Method notes and caveats

- All D2 figures in items 1-2 use `iaa.score.d2` with the "constant" baseline, matching
  `iaa.harness`/the leaderboard reading currently in force.
- The GLM baseline (`t31_glm.parquet`) arrived mid-task from the owner's sequential
  queue; it is included in items 1 and 2 (run at 2026-10-20 11:58) but item 3 and item 4
  do not depend on any specific challenger model and are unaffected.
- Dataset reads used the cached parquet matching the current frozen-file fingerprint
  (`902ed0cd8066`) rather than rebuilding, per the read-only/light-compute constraint on
  this task; all LightGBM fits capped at 2 threads.
- Scripts: `scripts/common.py`, `scripts/step0_run_map.py`,
  `scripts/step1_reported_scale.py`, `scripts/step2_compare_runs.py`,
  `scripts/step2b_foldA_capped.py`, `scripts/step2c_decile_chart.py`,
  `scripts/step3_drift.py`, `scripts/step4_proxy_matrix.py`.
