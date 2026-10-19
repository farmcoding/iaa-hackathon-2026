# Telematics, pruned variables, and excluded variables — review for Karl

Read-only review of `~/Projects/iaa-hackathon-2026` against the champion at commit
`f4aef71` (`results.tsv` run `20261019T172626`, the latest `keep_owner` row: "drop
five sales/admin fields on every line (channel kept) and prune predictors adding
<=0.0001 D2 on 2024"). The loop was left untouched; nothing in the repo was written.
Scripts live in `~/.cache/iaa-owner/1019/vars/` and reimport the frozen `iaa` modules
(`data`, `features`, `harness`, `ultimate`, `score`) and the experiment file's own
`_feature_cols` / `_credibility_encode` / `MONOTONE` (read-only import, no edits).
Test model throughout: single-seed LightGBM, `num_threads=3`, Tweedie power 1.5,
400 rounds at `learning_rate=0.05`, fold B (train 2021-23, valid 2024), on the
champion's own feature selection for the line in question. This is lighter than the
champion (no EBM blend on property, no 3-seed bag, no motor BI/Poisson split for
the pruned-variable tests), so absolute D2 numbers here are not comparable to
`results.tsv`; only deltas between variants fit on identical rows matter. Single-seed
noise on this light model runs about 0.001-0.002 per line, matching the prior
reviews (`reports/reviews/2026-10-19_features.md`, `..._sales_fields.md`); nothing
below is promoted to a 3-seed re-test because nothing cleared +0.001.

## 1. Telematics

`telematics_opt_in` is boolean, all motor rows; `telematics_score` (1-100) exists
only for the 12.6% of motor rows with opt-in = True. In the current champion,
`telematics_opt_in` is pruned (model_card.json pooled d2_drop -9e-6: adds nothing
once `telematics_score` and everything else is in); `telematics_score` is kept and
is the 11th-strongest motor predictor (pooled d2_drop 0.00114, ahead of
`licence_points` and `credit_band`). Note the model_card splits this lift by
component: `d2_drop_bi` 0.0001, `d2_drop_other` 0.00027 — the score's measured value
is almost entirely on the "other" perils, even though section 1a below finds the
opt-in *selection* gap is largest on BI frequency (1.11-1.35x). That tension is
real: the raw opt-in/not gap on BI is driven mostly by other rating factors already
in the model (age, mileage), which is exactly why opt-in itself prunes to zero once
those are present — see 1c.

### 1a. One-way: opt-in vs not, by year (reported-to-date, not developed)

| Year | Opt-in rate | Freq BI ratio (opt-in/not) | Freq other-peril ratio | LC other-peril ratio |
|---|---|---|---|---|
| 2021 | 11.7% | 1.247 | 1.131 | 1.165 |
| 2022 | 12.2% | 1.349 | 1.121 | 1.076 |
| 2023 | 12.7% | 1.273 | 1.025 | 1.031 |
| 2024 | 13.0% | 1.112 | 1.090 | 1.075 |

Opt-in drivers run worse than non-opt-in on every metric, every year: BI frequency
11-35% higher, other-peril frequency and loss cost 3-17% higher. Direction is stable
(always adverse, never favourable) but the BI margin drifts down (1.35 to 1.11) while
the other-peril margin is flat and small. This is a classic adverse-selection
pattern for an opt-in telematics program: riskier drivers are more likely to accept
monitoring (for the discount, or because an insurer/broker steered them to it), not a
rating signal that opt-in itself should carry a positive relativity for safe driving.
Opt-in's exposure share is rising slowly (11.7% to 13.0%), so its influence on the
book is small but growing.

**Verdict: opt-in alone is not a usable rating factor; it is a selection flag, and
the model is right to prune it. Do not add it back raw.**

### 1b. Within opt-in: does the score rank risk, and is the ranking stable?

Full decile table, opt-in subset only (loss cost per unit exposure, other perils;
frequency per unit exposure, bodily injury):

| Decile | mean score | LC-other 2021-23 | LC-other 2024 | Freq-BI 2021-23 | Freq-BI 2024 |
|---|---|---|---|---|---|
| 0 (worst) | 34.2 | 449.2 | 468.4 | 0.0174 | 0.0074 |
| 1 | 44.7 | 409.1 | 423.9 | 0.0157 | 0.0075 |
| 2 | 50.1 | 379.1 | 423.6 | 0.0139 | 0.0076 |
| 3 | 54.5 | 366.5 | 381.0 | 0.0168 | 0.0095 |
| 4 | 58.5 | 399.9 | 342.3 | 0.0118 | 0.0073 |
| 5 | 62.4 | 373.0 | 366.1 | 0.0145 | 0.0051 |
| 6 | 66.5 | 339.3 | 311.5 | 0.0116 | 0.0036 |
| 7 | 70.9 | 296.3 | 370.0 | 0.0123 | 0.0064 |
| 8 | 76.3 | 291.7 | 288.2 | 0.0107 | 0.0055 |
| 9 (best) | 86.4 | 296.0 | 319.0 | 0.0096 | 0.0044 |

Monotone with noise in both periods, roughly a 35-40% top-to-bottom spread on
LC-other, and BI frequency falls fairly cleanly with score in both years too.
Spearman rank correlation of the ten deciles' LC-other between 2021-23 and 2024 is
**0.782** (p=0.0075); for BI frequency it is **0.709** (p=0.022). Both are a real,
stable ranking, not noise — the score is doing its job as a risk ranking tool
within the opted-in population, on both the component the model credits it for
(other perils) and the one it currently doesn't (BI).

**Verdict: the score itself is a genuine, stable risk ranking on both peril groups.
Keep it, and the clean monotone relationship argues against banding it (section
1c).**

### 1c. Does the current model use the score well, and can cheap changes improve it?

A/E by score quintile, motor, using the champion's own predictions
(`data/cache/preds/20261019T172626.parquet`, the newest predictions file matching
the latest `keep_owner` run), fold B (valid 2024) first, then fold A (valid 2023)
as a replication check:

| Score quintile | mean score | Fold B (2024) A/E | Fold A (2023) A/E |
|---|---|---|---|
| 0 (worst) | 39.1 | 0.916 | 0.859 |
| 1 | 52.2 | 1.126 | 0.912 |
| 2 | 60.5 | 0.849 | 0.940 |
| 3 | 68.7 | 1.093 | 0.797 |
| 4 (best) | 81.5 | **1.293** | 0.842 |

Fold B alone looked like a worrying pattern: A/E rising from 0.92 to 1.29 across
quintiles while the model's predicted rate falls (802 to 521), suggesting the model
over-discounts the best-scored drivers. **Fold A does not replicate this at all** —
every quintile runs under 1.0, with no rank pattern by score (best quintile is 0.84,
not the highest). Re-stated more carefully: on fold B, the top three quintiles run
1.09-1.29 and the bottom two scatter near 0.85-0.92; on fold A, everything scatters
in a tight 0.80-0.94 band with no relation to score. Capping `y` at 250k or 1m
(matching `BI_CAP`/`BI_TAIL`) barely moves fold B's numbers (band-by-band deltas
under 0.03 A/E, and 1m-capped is identical to uncapped — no single claim in this
motor-other-peril slice exceeds 1m), so the large-loss story doesn't explain fold
B's pattern either. The most likely reading: fold B's rising-A/E pattern is a
single-year sampling fluke on cells of 4,800-5,500 exposure units each, not a real
mis-calibration of the score's relativity — the lack of any fold-A analogue is the
strongest evidence for that.

**Verdict: no evidence of a real score mis-calibration once a second fold is
checked. Downgrade this from "worth a second look" to "resolved — likely noise, no
action," and drop it from the owner backlog below.**

Cheap tests, single seed, motor "other" peril (BI is a separate Poisson/count path
in the champion; the model_card shows BI's own telematics lift is tiny, 0.0001, so
it was not re-tested here), fold B, champion's own feature set as base (base d2_other
= 0.09120):

| Variant | d2_other | delta vs base |
|---|---|---|
| A. Base (raw score, opt_in dropped) — current | 0.09120 | — |
| B. Drop telematics_score | 0.09051 | -0.00069 (confirms the model_card lift) |
| C. Add telematics_opt_in back alongside score | 0.09079 | -0.00041 |
| D. Band score into quintiles instead of raw numeric | 0.09108 | -0.00012 |
| E. Add score x driver_age interaction | 0.09082 | -0.00038 |
| F. Add score x annual_mileage interaction | 0.09074 | -0.00046 |
| G. Both interactions together | 0.09046 | -0.00074 |
| H. Separate model trained on opt-in rows only (88.6k rows), scored on opt-in valid | 0.05615 | vs 0.08957 for the pooled model on the same opt-in valid rows: **-0.034** |

Every variant is at or below the current treatment; none clears even the lower end
of the single-seed noise band, so none was repeated on extra seeds. Banding destroys
information the raw numeric split already captures cleanly (consistent with 1b's
clean monotone ranking). The explicit interactions don't help because, at
`num_leaves=15`, the trees already approximate score x age and score x mileage
splits from the raw fields — the same pattern the 2026-10-19 features review found
for premium-replacement interactions on buildings/contents (section 2.3 there).
Variant H (opt-in-only model) is also disadvantaged by construction, not just
sample size: `min_data_in_leaf=2000` is a much tighter constraint relative to 88.6k
training rows than to the pooled model's 760k, so it has far less room to form
leaves before the credibility from the other 87% of the book (which anchors the
base rate) is available to borrow from. Both effects point the same way, and the
gap (-0.034) is large enough that neither alone explains it away: a dedicated
opt-in model is simply worse here.

**Verdict: the current treatment (raw numeric score, `telematics_opt_in` dropped,
no explicit interactions, pooled not separate) is already close to the cheap-test
ceiling. No change recommended.**

## 2. Pruned variables: is there a transformation that unlocks them?

Checked whether the ~half of predictors pruned on 2026-10-19 (adding <=0.0001 pooled
D2 when shuffled) carry information in a different shape. First, the per-line
permutation lift in `reports/features/model_card.json` (the pre-pruning model) for
every pruned field, excluding the fields already ruled out on admissibility
(campaign_code, marketing_segment, branch_id, days_quote_to_inception,
marketing_consent, email_domain, licence_points-on-property): every one of them is
between -0.0008 and +0.0002 **per line**, not just pooled. That rules out the most
common failure mode (a field that's valuable on one line but washed out pooled) —
nothing was hiding there.

Four transformations with an actuarial rationale were tested directly (single seed,
same methodology as section 1c):

| Transformation | Line(s) | Rationale | d2 (base -> variant) | Delta |
|---|---|---|---|---|
| `construction_type x property_age` (5-band), singly | buildings | Construction and property age interact mechanically (an old timber-frame building ages differently from a new brick one); backlog 2.47 only tested this bundled with 4 other families and bundled scored below baseline — never tested alone | 0.11179 -> 0.11228 | +0.00048 |
| Same | contents | Same rationale, contents has no `construction_type` of its own but inherits the address record | 0.08957 -> 0.08957 | +0.00000 |
| `sum_insured / bedrooms` | buildings | A given sum insured spread over fewer bedrooms suggests higher per-room value/quality, a severity signal distinct from raw sum insured | 0.11179 -> 0.10862 | -0.00317 |
| Same | contents | Same rationale | 0.08957 -> 0.08823 | -0.00134 |
| `engine_size_cc / seats` ("power density") | motor | A small car with a big engine, or a 2-seater with a big engine, is a different risk than either field alone suggests (performance/theft-attractiveness proxy) | 0.09120 -> 0.09069 | -0.00051 |
| `driver_age - years_licensed` (age first licensed) | motor | Backlog 2.22 proposed dropping `years_licensed x customer_age`'s duplicate-driven artefact; this version uses `driver_age` (the field actually kept) instead of the pruned `customer_age`, to see if the age-gap concept itself is real once the duplicate-field artefact is avoided | 0.09120 -> 0.09113 | -0.00008 |

Only `construction_type x property_age` on buildings showed a positive delta, and at
+0.00048 it is inside the single-seed noise band (0.001-0.002), well below the
+0.001 bar for a second-seed check. The ratio features (sum insured per bedroom,
engine size per seat) actively hurt — collapsing two informative raw fields into one
ratio removes information the trees were already using separately, the same failure
mode the features review documented for the bundled premium-replacement interactions
(section 2.3: "the trees already approximate these interactions from the base
fields; explicit terms mostly compete with them for split budget rather than adding
information").

**Verdict: no hidden value found. The pruning decision holds up field-by-field, not
just pooled, and the four transformations tried (two interaction types, two ratio
types) don't change that. `construction_type x property_age` on buildings is the one
item worth a cheap single confirmatory run if the loop has a free slot, but expect
it to land inside the floor.**

Not tested, and not recommended, given the evidence above: missing-value flags for
the pruned booleans (`smoke_alarms`, `mortgage`, `solar_panels`) — their raw values
already show near-zero or negative per-line importance, so a flag derived from the
same field is unlikely to do better; and further peril-mechanics interactions
(flood x elevation x distance-to-river, soil x trees x dry-summer) — already tried
bundled (backlog 2.2, "tried: below best") and the single-family tests that have
been run since (`flood_zone x rainfall_mm`, `excess / sum_insured`) both discarded
(`results.tsv` `01cd8c8`, `d1095f5`).

## 3. Excluded variables, refreshed inventory

Every field not currently used as a model feature, excluding what the owner has
already ruled out on admissibility (`written_premium`, `email_domain`,
`licence_points` on property, `campaign_code`, `marketing_segment`, `branch_id`,
`days_quote_to_inception`, `marketing_consent`, and any sex/ethnicity/religion
proxy — none of the remaining raw fields is one; this was checked in the features
review, section 3.4, for email domain specifically and nothing else in the schema
carries demographic proxy risk at that level).

### 3.1 Structural exclusions (correctly excluded, not candidates)

| Field(s) | Reason |
|---|---|
| `row_id`, `policy_id`, `customer_id`, `address_id` | Identifiers |
| `period_start`, `period_end` | Enter the model through `start_month`, ages and `exposure`; not usable raw |
| `exposure` | Weight and level base; tested as a feature once (run 40), cost 0.003 |
| `year` | Calendar year; trend is handled by on-levelling, not a model-learned year effect that wouldn't extrapolate to 2025 |
| `line` | Structural — each line gets its own model |
| `renewal_decision`, `customer_service_contacts`, `customer_since_year` | Leakage (features.py `FORBIDDEN`): train-only/drift fields, or training-only contamination from claim-related contacts |
| `exposure_raw`, `dq_exposure_fixed` | Data-quality working columns, not risk characteristics |
| `date_of_birth`, `surname` | Replaced by derived age; surname never a candidate |
| claim-level fields (`claim_id`, `peril`, `loss_date`, `report_date`, `status`, `paid_to_date`, `incurred_to_date`, `recoveries`, `repairer_id`, `third_party_id`, `report_lag_days`) | Not known for a future policy period at underwriting time; correctly used only to build the backward-looking `hist_n_*`/`hist_cost_*` features |
| `lines_held` | Tested, no gain (0.1334 vs 0.1339) |

### 3.2 Pruned fields that are conceptually different from what's kept, but empirically weak

Real, distinct actuarial concept — not redundant with a kept field — but the
permutation evidence (section 2) says they add essentially nothing in any line:

| Field | Line(s) pruned | What it would be, in theory | Distinct from kept fields? | Verdict |
|---|---|---|---|---|
| `previous_insurer` | all three | Switching behaviour / shopping proxy | Yes | d2_drop -0.00009 to -0.00085 everywhere; flat |
| `discount_applied` | motor, buildings, contents | Whether a discount was applied at sale | Yes, distinct from channel/payment_method | -0.00007 to -0.00011; flat |
| `document_preference` | all three | Paper vs digital correspondence | Yes, engagement/channel-affinity | 3e-05 to 7e-05; flat |
| `renewal_count` | motor, contents (kept on buildings) | Loyalty/tenure with this insurer specifically | Distinct from customer_tenure (relationship) and years_at_address — though note motor also prunes customer_tenure, so motor retains *no* tenure/loyalty signal at all | -0.00016 to -0.0003; flat on the lines it's pruned |
| `occupation` | motor, contents | Socio-economic/occupational risk | Distinct from credit_band | -0.00007 to +5e-05; flat |
| `days_unoccupied` | buildings | Vacancy risk (fire/water/theft all rise when empty) | Genuinely distinct peril-mechanics field | **-0.00078**, the most negative pruned field on buildings; real concept, data doesn't support it as recorded |
| `hist_cost_customer`, `hist_cost_policy` | buildings and contents (both); contents additionally prunes `hist_n_customer`/`hist_n_policy` | Claim-cost history, same-customer and same-policy | Distinct from the raw counts; the features review found pooled customer history is 64-67% driven by a claim on the *other* line | Buildings keeps both claim-**count** features (`hist_n_customer` d2_drop 0.00018, `hist_n_policy` 0.00026) and drops both claim-**cost** features — buildings has a count-based history signal, just not a severity-weighted one. **Contents drops all four** and so has no claim-history feature of any kind below `prior_claims_5y` — this is the real, contents-specific gap (corrected from an earlier draft of this review that wrongly extended it to buildings too) |
| `property_type` | buildings | House/flat/bungalow etc. | Distinct, structural | -0.0005; flat |
| `freezing_days` | buildings | Pipe-freeze / escape-of-water mechanics | Distinct peril-mechanics field | -0.0006; flat, likely swamped by region/soil_type geography |
| `marital_status`, `household_size` | all three | Household composition | Distinct from age/credit fields | -0.0002 to +3e-05; flat |
| `region` (buildings) | kept elsewhere | Geography | Odd to see pruned given flood_zone/elevation_m are buildings' strongest predictors — likely fully subsumed by the finer address fields | -5e-05; flat |

### 3.3 Pruned fields that are a weaker version of a concept covered elsewhere

| Field(s) | What's going on | Evidence / caveat |
|---|---|---|
| `customer_age` (motor) | Duplicates `driver_age` (kept), Spearman 1.00 per backlog 2.22; also the source of a spurious `years_licensed x customer_age` top-importance artefact flagged there | Correctly dropped; motor keeps the right age field |
| `customer_age` **and** `policyholder_age` (buildings) | **Both pruned — buildings currently has no age variable of any kind**, not "one kept, one dropped." Both tested flat/negative in model_card (`policyholder_age` d2_drop -0.00013). Plausibly fine: property risk may genuinely not need an age variable once credit_band, sum insured and location are present, but this is worth a conscious owner call rather than an artefact of the automated 0.0001 threshold, since age is a defensible rating factor on property in many markets | Flag for Karl, not a re-test recommendation — see 3.60 |
| `property_age` **and** `year_built` (motor) | Both pruned on motor (not a kept/dropped pair — motor doesn't need either; they're buildings-style fields reaching motor rows via the shared address record, per the features review's cross-line leakage section 1) | Correctly dropped; vehicle risk doesn't depend on the dwelling's age |
| `telematics_opt_in` | Subsumed by `telematics_score`'s native-missing split | Section 1 |
| `soil_type`, `trees_within_5m`, `wind_exposure_index`, `hail_days` (various) | Subsumed by flood_zone/elevation_m/region/rainfall_mm | Flat-to-negative everywhere; no line shows a surprise positive |

### 3.4 Net read

Nothing sitting unused today is quietly valuable. The one real gap worth flagging
to the owner is **3.2's claim-history finding, corrected**: the gap is
**contents-only** — contents has no claim-history feature of any kind below
`prior_claims_5y`. Buildings keeps claim-count history (same-customer and
same-policy) but has dropped claim-cost history, a narrower gap. Motor is unaffected
(its own `hist_n_customer`/`hist_cost_customer` are pruned too, per the PRUNED list,
but motor's claim pattern is already well explained by `prior_claims_5y`, `ncb_years`
and the telematics score, so this matters less there). Backlog item 3.34 already
flags a related valuation-date inconsistency in how `hist_cost_*` is built, which
may be part of why the cost-based versions test weakly everywhere.

## Proposed experiments

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 2.60 | Re-test `construction_type x property_age` (5-band) singly on buildings, 3 seeds | This review: +0.00048 single-seed, the only positive delta among four tested transformations and backlog 2.47's last untested single family | Likely inside the floor; low cost to confirm | This review, section 2 | open |
| 2.61 | Recency-weighted or capped claim-cost history for contents (both same-customer and same-policy; it currently has none), and claim-cost history for buildings specifically (it already keeps the count versions) in place of the raw pooled sum | Section 3.2/3.4: contents has no claim-history feature of any kind below prior_claims_5y; the raw cost sum is pruned everywhere it was tried, a recency/cap transform hasn't been | Small gain if the raw version's weakness is a shape problem, not an information problem | This review | open |
| 3.60 | Owner: two small decisions arising from section 3 — (a) accept contents having no claim-cost history feature, or fund 2.61; (b) confirm buildings having no age variable at all (both customer_age and policyholder_age pruned) is an intentional read of the data, not just an artefact of the automated 0.0001 threshold | Both are real gaps relative to what a traditional property rating plan would carry, even though neither tested positively in this data | Decision input, not a score change | This review, section 3 | open |
