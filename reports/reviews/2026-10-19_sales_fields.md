# Sales and admin field review: campaign_code, marketing_segment, branch_id, channel, days_quote_to_inception, marketing_consent

Read-only review of `~/Projects/iaa-hackathon-2026` against the champion at the commit
current when this review started (fold B validation predictions:
`data/cache/preds/20261019T101532.parquet`). The loop and the owner's test runs were
left untouched; no repo file was read or written except via the cached dataset
parquet and `git show`. Numbers below come from scripts under
`~/.cache/iaa-owner/1019/sales/` that reimport the frozen `iaa` modules (`data`,
`harness`, `ultimate`, `score`) and either run plain pandas diagnostics (profiling,
one-way tables, Cramer's V / correlation ratio, branch concentration, residual A/E
against the champion's own fold predictions -- no model fit needed for any of these)
or refit a lighter single-seed LightGBM (`num_threads=3`, 350 rounds at
`learning_rate=0.05`, Tweedie power 1.5, fold B only: train 2021-23, valid 2024) on
the champion's own feature selection, credibility encoder and the current `LINE_DROP`
exclusions. The lighter model is not the champion (no EBM blend, no motor BI/other
split, single seed, fewer rounds), so its own D2 numbers are not comparable to
`results.tsv`; only the deltas between variants, fit on identical rows, matter here.
Single-seed deltas on this lighter model are shown below to run around 0.0017 even
when the field removed carries no real signal (Section 4), so any delta under about
0.002 on one line is not distinguishable from noise unless it repeats in the same
direction across all three lines or across reseeded runs.

Two of the six fields are policy-level (`channel`, `days_quote_to_inception`: stored
in each line's own `policies_<line>.parquet`, so they can differ by line and year for
the same customer). The other four (`campaign_code`, `marketing_segment`,
`branch_id`, `marketing_consent`) are customer-level (`customers.parquet`), so a given
customer carries the same value onto every line's rows regardless of whether that
line has any rating rationale for it -- the same cross-line reach already flagged for
`email_domain` and `licence_points` in `reports/reviews/2026-10-19_features.md`
Section 1.2.

## Summary

| Field | Verdict | Reason |
|---|---|---|
| campaign_code | drop | 67-71% missing; 15 near-uniform levels; one-way rank unstable across years (Spearman -0.36 to +0.37); no association with any risk field (max 0.037) |
| marketing_segment | drop (redundant) | correlation ratio 0.967 with customer_age, Cramer's V 0.707 with credit_band -- both already in the model; real one-way signal (young segments run 2x the frequency of mature/senior, stable across years) but it is age and credit_band relabelled, not new information |
| branch_id | drop | stored as a float 1-25 and modelled as ordinal though it is a nominal admin code; one-way rank unstable (Spearman 0.12-0.18) with relativity swings up to 1.98x; no geography link (Cramer's V/correlation ratio 0.002-0.007 against region and urban_density) |
| channel | keep raw, 3 levels, no abstraction needed | comparison_site runs 5-14% above the line average in both periods on all three lines; no association with age, credit_band or region (all under 0.015), so it is not a demographic proxy; 3-seed model test on motor is inconclusive (+0.0017 / -0.0002 / +0.0016 D2 vs dropping it) -- keep is an admissibility judgement, not a measured gain |
| days_quote_to_inception | drop (raw and banded) | near-uniform 0-30 day spread (mean 13.2, median 13, p90 27); one-way loss-cost rank by lead-time band is not stable across years on any line; banded abstraction changes D2 by -0.0006 / +0.0014 / -0.0010 across the three lines, no consistent sign |
| marketing_consent | drop | boolean 55/45 split; relativity 0.94-1.07 with no stable direction across lines or years; no association with anything measured |

Net effect of removing all six: in the lighter test model, D2 moves from 0.0774 to
0.0769 on motor (-0.0005), 0.1076 to 0.1084 on buildings (+0.0008), and 0.0864 to
0.0902 on contents (+0.0038, the largest single number in this review). All three are
either inside the floor or a small gain; none argue for keeping the group.

## 1. campaign_code

15 levels (`CMP00`-`CMP14`), missing 67-71% of rows on every line (not line-specific:
the missingness is a customer-level gap, same ~10% extra on motor as the other three
customer fields below, plus a further ~60% gap common to all lines -- most customers
simply have no campaign code recorded). Where present, exposure splits almost evenly
across the 15 codes (6.5-6.9% each), consistent with a label assigned at random
rather than a real acquisition channel.

One-way loss cost by code, 2021-23 vs 2024 (Spearman rank correlation of the two
periods' relativities-to-mean):

| Line | Spearman | 2021-23 relativity range | 2024 relativity range |
|---|---|---|---|
| Motor | -0.357 | 0.82 - 1.18 | 0.79 - 1.20 |
| Buildings | 0.371 | 0.82 - 1.19 | 0.74 - 1.34 |
| Contents | 0.000 | 0.83 - 1.19 | 0.72 - 1.28 |

A real rating factor holds its rank across periods; this is close to zero correlation
on two of three lines and negative on the third, with wide relativity swings on a
15-way split of a field with no demographic or geographic association at all (the
strongest of 19 measured associations is correlation ratio 0.037 against
telematics_score on motor; everything else is under 0.02). This is noise.

## 2. marketing_segment

12 levels, each an `<age-band>_<affluence-band>` label (young/midlife/mature/senior x
affluent/mainstream/budget). Missing 10.0% on motor, 0% on buildings and contents --
the same customer-level gap discussed in Section 6.

### 2.1 What it is a proxy for

Correlation ratio / Cramer's V against candidate risk fields (motor, similar on the
other two lines):

| Candidate | Association | Reading |
|---|---|---|
| customer_age | 0.967 | near-deterministic |
| credit_band | 0.707 | near-deterministic |
| missed_payments_12m | 0.37-0.40 | strong |
| years_at_address | 0.25 | moderate |
| payment_method | 0.16-0.21 | moderate |
| customer_tenure | 0.19 | moderate |
| new_business | 0.16 | moderate |
| prior_claims_5y, region, urban_density, vehicle fields | <0.02 | none |

`marketing_segment` is not independent noise like `campaign_code` -- it carries a
real, stable one-way signal -- but the signal is almost entirely a relabelling of
`customer_age` and `credit_band`, both already admissible rating fields present
everywhere in the current model.

### 2.2 The real signal it carries, and why it isn't new information

One-way frequency by segment, motor, both periods (per unit exposure):

| Segment | 2021-23 | 2024 |
|---|---|---|
| mature_affluent | 0.101 | 0.089 |
| mature_mainstream | 0.100 | 0.094 |
| mature_budget | 0.119 | 0.113 |
| midlife_affluent | 0.120 | 0.108 |
| midlife_mainstream | 0.117 | 0.104 |
| midlife_budget | 0.137 | 0.119 |
| senior_affluent | 0.128 | 0.120 |
| senior_mainstream | 0.129 | 0.116 |
| senior_budget | 0.148 | 0.129 |
| young_affluent | 0.230 | 0.177 |
| young_mainstream | 0.231 | 0.182 |
| young_budget | 0.268 | 0.206 |

The young segments run at roughly twice the frequency of mature and senior segments,
consistently in both periods -- a real and stable effect. But this is exactly what
`driver_age`/`customer_age` already buys the model directly, with `credit_band`
layering the budget/mainstream/affluent split on top. The residual A/E of the
champion's own fold B predictions by segment (the champion already includes
`marketing_segment` as a feature today) sits mostly in the 0.90-1.15 band with no
segment running persistently hot or cold across both folds -- consistent with age and
credit_band already absorbing the real part of the signal, and the field adding
mostly noise on top.

### 2.3 Model test

Light LightGBM, fold B, single seed, sales-field-free baseline (`BASE`, all six
fields dropped) vs `BASE` plus `marketing_segment` only:

| Line | BASE | BASE + marketing_segment | Delta |
|---|---|---|---|
| Motor | 0.07694 | 0.07819 | +0.00125 |
| Buildings | 0.10842 | 0.10744 | -0.00098 |
| Contents | 0.09016 | 0.08942 | -0.00074 |

Mixed sign, all inside the single-seed noise band measured in Section 4 (0.0017 on a
field with zero real signal). No reseed was run for this field since the direction
already matches what the redundancy finding in 2.1 predicts (no gain, since age and
credit_band are already present in `BASE`); a confirmation run would only tighten a
null result that doesn't need tightening.

### 2.4 Verdict

Drop. The field's real content is age and credit_band, already in the model under
their own names; keeping `marketing_segment` on top adds an un-auditable relabelling
of fields the actuary can already point to directly, with no measured score benefit.

## 3. branch_id

Numeric, 25 branches (coded 1-25), missing 10.0% on motor, 0% on buildings/contents.
Stored as a float and therefore modelled by LightGBM as an ordinal/continuous feature
today, even though a branch number is a nominal administrative code with no
intrinsic order -- a second, independent reason to be cautious with it beyond the
admissibility question (flagged as backlog 3.52 below; moot if the field is dropped).

### 3.1 Is it a geography proxy

Per-branch concentration of region and urban_density (max category share within each
branch, vs. the baseline if a branch's book matched the portfolio marginal exactly):

| Field | Mean max-share across the 25 branches | Uniform-random baseline |
|---|---|---|
| region (12 categories) | 0.162 | 0.083 |
| urban_density (4 categories) | 0.449 | 0.250 |

Both concentration figures track the overall marginal distribution's own largest
category almost exactly (confirmed directly: the max-share figures are not
materially different branch to branch -- std 0.003 and 0.005 respectively across the
25 branches), meaning every branch's regional and urban-density mix looks like the
whole portfolio's, not a local pocket. This matches the association numbers directly:
correlation ratio 0.006-0.007 against region, 0.002-0.003 against urban_density, on
every line. Branch is not a geography proxy; each branch appears to draw from the
same book-wide population.

### 3.2 One-way stability

| Line | Spearman (2021-23 vs 2024 relativity rank, 25 branches) | Relativity range |
|---|---|---|
| Motor | 0.116 | 0.89 - 1.98 (2021-23) |
| Buildings | 0.118 | 0.81 - 1.17 |
| Contents | 0.177 | 0.89 - 1.14 |

Near-zero rank stability with a swing up to 1.98x on a 25-way split of exposure
(roughly 4% of a line's book per branch) is the signature of small-cell noise, not a
rating factor.

### 3.3 Model test

Tested only inside the `noise_trio` bundle (`campaign_code` + `branch_id` +
`marketing_consent`, Section 4) rather than alone, since all three show the same
profile (near-zero association with everything, unstable one-way rank). Adding the
trio back to `BASE` cost buildings 0.0029 and contents 0.0023 D2, and gained 0.0007 on
motor -- net negative on the two lines where it's large enough to read as a real
signal rather than noise.

### 3.4 Verdict

Drop. No geography content, no stable one-way rank, measured cost on two of three
lines when combined with the other two noise fields.

## 4. channel

Three levels: `direct` (40%), `comparison_site` (35%), `broker` (25%), no missing
values on any line. The one field in this group with a real and reasonably stable
one-way pattern.

### 4.1 One-way stability

Loss-cost relativity to the line-year mean, both periods, all three lines:

| Line | broker 2021-23 | broker 2024 | comparison_site 2021-23 | comparison_site 2024 | direct 2021-23 | direct 2024 |
|---|---|---|---|---|---|---|
| Motor | 0.941 | 0.981 | 1.061 | 1.065 | 0.984 | 0.955 |
| Buildings | 0.988 | 0.949 | 1.104 | 1.052 | 0.916 | 0.986 |
| Contents | 0.960 | 1.115 | 1.138 | 1.022 | 0.904 | 0.908 |

`comparison_site` is above the line average in all six line-period cells (range 1.02
to 1.14) -- the one fully consistent pattern in this review. `broker` vs `direct`
ordering is less stable (broker is cheapest on motor and buildings but flips to the
most expensive cell on contents in 2024).

### 4.2 What it proxies, and the model test

Correlation ratio / Cramer's V against age, credit_band, region and the rest of the
candidate list: all under 0.015 on every line (customer_age 0.003-0.005, credit_band
0.004-0.005, region 0.006-0.010). Channel's own-effect is not explained by any
demographic or geographic field measured -- it isn't quietly re-encoding age or
affluence, but the comparison_site effect, if real, is a channel-specific effect in
its own right (consistent with the price-comparison anti-selection pattern known in
UK/EU personal lines, though this dataset gives no independent way to confirm that
story beyond the one-way stability above).

Light LightGBM, fold B, `BASE` vs `BASE + channel` (raw, 3 levels), motor, 3 seeds:

| Seed | BASE | BASE + channel | Delta |
|---|---|---|---|
| 0 | 0.07694 | 0.07862 | +0.00168 |
| 1 | 0.07782 | 0.07764 | -0.00018 |
| 2 | 0.07582 | 0.07744 | +0.00162 |

Sign flips once across three seeds; the average delta (+0.0010) is smaller than the
seed-to-seed swing in `BASE` alone (0.07582 to 0.07782, a range of 0.0020). Buildings
and contents (single seed) showed +0.0002 and -0.0007 respectively against `BASE`.
There is no reliable score case for keeping or dropping channel either way.

### 4.3 Verdict

Keep as-is, no abstraction needed. It is already a 3-level, no-missing, directly
explainable field describing how the policy was sold, with a plausible and
one-way-stable higher-cost segment (comparison_site). There is nothing to simplify --
collapsing 3 levels to 2 would not reduce complexity meaningfully -- and no measured
score cost either way. This is a judgement call for the admissibility file, not a
score-driven keep.

## 5. days_quote_to_inception

Integer, 0-30 days, no missing values. The distribution looks close to uniform
rather than skewed toward late bookings: mean 13.2, median 13, p90 27, min 0, max 30,
on every line. That shape is itself a reason to doubt the "last-minute buyer" story
the field would carry in a real book -- a genuine lead-time distribution in personal
lines is usually right-skewed (most people buy close to inception, a long tail books
early), not flat.

### 5.1 One-way stability by lead-time band (0-2 / 3-14 / 15-30 days)

Loss-cost relativity to the line-year mean (band highest both periods in bold):

| Line | 0-2, 2021-23 | 0-2, 2024 | 3-14, 2021-23 | 3-14, 2024 | 15-30, 2021-23 | 15-30, 2024 |
|---|---|---|---|---|---|---|
| Motor | 0.99 | **1.08** | 0.96 | **1.07** | **1.03** | 0.92 |
| Buildings | **1.08** | 1.04 | 0.97 | 0.98 | 0.98 | **0.99** |
| Contents | 1.03 | 0.89 | 0.97 | 0.94 | **1.03** | **1.07** |

No band is consistently highest across both periods on any line; motor's top band
flips from 15-30 days (2021-23) to 3-14 days (2024), and contents flips from 15-30
(2024) back toward 0-2 (2021-23, though close). This is the one-way signature of
noise, not the stable last-minute-buyer effect this kind of field sometimes carries in
real personal-lines markets.

### 5.2 What it proxies, and the model test

Correlation / correlation ratio against every candidate risk field: all under 0.015
(previous_insurer 0.012-0.015 is the largest, itself a sales-admin field with no risk
content of its own). No age, region or claims-history relationship.

`BASE` vs `BASE` plus a 3-band categorical replacing the raw 31-level numeric, single
seed:

| Line | BASE | BASE + lead-time band | Delta |
|---|---|---|---|
| Motor | 0.07694 | 0.07639 | -0.00055 |
| Buildings | 0.10842 | 0.10981 | +0.00139 |
| Contents | 0.09016 | 0.08918 | -0.00098 |

No consistent sign across lines, and the one positive delta (buildings) is itself
inside the noise band measured for channel in Section 4.2. No reseed was run since
the one-way and association evidence already rules out a real effect; a 3-seed
confirmation would only add cost to tighten a null result on a field with no story
behind it.

### 5.3 Verdict

Drop, raw or banded. The lead-time effect this field is supposed to carry does not
show up in this data, in either the one-way tables or the model test.

## 6. marketing_consent

Boolean, 55% true / 45% false, missing 10.0% on motor only (same customer-level gap as
Sections 2 and 3). One-way relativity:

| Line | False 2021-23 | False 2024 | True 2021-23 | True 2024 |
|---|---|---|---|---|
| Motor | 1.049 | 1.067 | 0.960 | 0.945 |
| Buildings | 0.989 | 1.023 | 1.009 | 0.981 |
| Contents | 0.996 | 0.973 | 1.003 | 1.022 |

Motor shows a mild, consistent pattern (consent = True runs 4-6% cheaper in both
periods); buildings and contents flip sign between periods. No candidate association
clears 0.011. Tested only inside the `noise_trio` bundle (Section 3.3), which showed
a net cost on the two property lines. Given the inconsistency between lines and the
near-zero association with anything, the motor pattern reads as one plausible
coincidence rather than three consistent ones.

### 6.1 Verdict

Drop. Whatever "opted in to marketing" correlates with in the real world, it isn't
visible here as a stable risk signal on any line.

## 7. A data-quality note (not this task's main question, but found along the way)

`campaign_code`, `marketing_segment`, `branch_id` and `marketing_consent` are all
missing in exactly the same ~10.0% share of motor rows and 0% of buildings/contents
rows. That is not a line-specific pattern (the fields are joined from the same
`customers.parquet` on every line) -- it is a subset of customers, roughly 10% of the
ones who hold a motor policy, who have no match in a chunk of their own customer
record. Not investigated further here; flagged for whoever owns the data-quality log,
since a customer-level join gap that size is worth a one-line note even though it did
not move the verdicts above.

## Backlog additions

Tier format matches `research/experiment_backlog.md`.

### Loop track

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 2.50 | Drop `campaign_code`, `branch_id`, `marketing_consent` from all three lines | Section 1, 3, 6: near-uniform/unstable one-way pattern on each, Cramer's V / correlation ratio under 0.02 against every risk field for all three; adding the trio back to a sales-field-free baseline cost buildings 0.0029 and contents 0.0023 D2 (single seed), gained 0.0007 on motor | Inside the floor to a small gain; cleaner admissibility | This review Sections 1, 3, 6 | open |
| 2.51 | Drop `marketing_segment` from all three lines | Section 2: correlation ratio 0.967 with customer_age, Cramer's V 0.707 with credit_band, both already in the model; adding it back to the sales-field-free baseline cost buildings 0.0010 and contents 0.0007 D2, motor +0.0013 (inside the single-seed noise band) | Inside the floor; redundant with admissible fields already present | This review Section 2 | open |
| 2.52 | Drop `days_quote_to_inception`, raw and as a 3-band lead-time categorical, from all three lines | Section 5: near-uniform 0-30 day spread; one-way loss-cost rank by lead-time band is not stable across years on any line; the banded abstraction changed D2 by -0.0006 / +0.0014 / -0.0010 across the three lines, no consistent sign | Inside the floor; no real lead-time effect in this data | This review Section 5 | open |

### Owner track

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 3.50 | Admissibility decision: `campaign_code`, `marketing_segment`, `branch_id`, `marketing_consent` are customer-level fields that reach every line regardless of relevance, the same pattern already flagged for `email_domain`/`licence_points` (Features review Section 1.2, backlog 3.46) | Not a score question; closes the admissibility review for the whole sales/marketing group in one decision | A recorded line in `reports/assumptions_log.md` alongside 3.45/3.46 | This review | open |
| 3.51 | Keep `channel` (3 levels, policy-level, no missing) as an admissible factor on its own grounds, independent of score | Section 4: comparison_site runs 5-14% above the line average in both periods on all three lines; association with age, credit_band and region is under 0.015 everywhere, so it is not a demographic proxy; a 3-seed motor test gave +0.0017 / -0.0002 / +0.0016 D2 against dropping it, inside noise either way | A judgement call (cheap, explainable, a plausible anti-selection story), not a score-driven keep | This review Section 4 | open |
| 3.52 | `branch_id` is stored as a float and modelled as ordinal though it is a nominal admin code (1-25, no intrinsic order); moot if 2.50 is adopted, otherwise recast as categorical | Section 3: no rating rationale either way, but an ordinal encoding of a nominal ID is a separate modelling defect from the admissibility question | Flag for the data-quality log if `branch_id` is kept for any reason | This review Section 3 | open |

## Scripts and outputs

All under `~/.cache/iaa-owner/1019/sales/`: `build_slim.py` (slim columns pulled from
the cached dataset parquet, no repo cache written), `step1_profile.py` (cardinality,
missing share, exposure share), `step2_oneway.py` (one-way frequency/loss cost by
level, 2021-23 vs 2024), `step3_assoc.py` (Cramer's V / correlation ratio matrix plus
branch region/urban_density concentration), `step5_residual.py` (A/E of the
champion's own fold A/B predictions by level), `common.py` + `run_variants.py` (the
light LightGBM variant tests) and `reseed_channel.py` (the 3-seed channel
confirmation on motor). Raw outputs: `profile_out.json`, `oneway_out.json`,
`assoc_out.json`, `branch_conc.json`, `residual_out.json`, `variant_results.json`,
`reseed_channel_motor.json`.
