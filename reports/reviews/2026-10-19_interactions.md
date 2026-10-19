# Interaction and two-way review — IAA hackathon 2026 Part 1

Scope: current champion at `src/iaa/experiment.py` HEAD (commit `7931648`), latest
owner keep run `20261019T172626` (commit `f4aef71`, primary 0.14048, fold B D² by
line: motor 0.08492, buildings 0.11366, contents 0.09452). All work is read-only
against the repo; four JSON artefacts back this report
(`twoway.json`, `interactions.json`, `noise.json`, `techniques.json`, all in this
folder). Top-8 predictors per sub-model came from `reports/features/model_card.json`,
filtered to the features the current `LINE_DROP`/`PRUNED` rules still admit.

## 1. Two-way tables (`twoway.json`)

28 pairs per sub-model (C(8,2) on the top-8 list), actual on 2021-23 train (motor BI:
claim frequency; motor other-perils / buildings / contents: loss cost per exposure),
predicted on fold-B valid (2024) from a single-seed replica of each sub-model
architecture (BI Poisson, other-perils Tweedie, property LightGBM+EBM 50/50 blend —
same objective, monotone constraints and credibility encoding as `experiment.py`, one
seed instead of three for speed). Quintile bands for continuous features, top-4 +
"other" for categoricals.

Worked example — buildings, `flood_zone` x `elevation_m` (the strongest interaction by
every method below): at flood_zone 4-5 and the lowest elevation quintile, actual loss
cost is 4-9x the flood_zone-1 baseline (928-2925 vs 340-400) and the model tracks it
closely (A/E 0.95-1.17). But at flood_zone 3-4 combined with a higher elevation
quintile, actual falls faster than predicted (cell [flood 3, elev 5]: actual 171,
predicted 388, A/E 0.44; cell [flood 4, elev 5]: actual 367, predicted 701, A/E 0.52).
The model keeps flood_zone's steep main effect even where elevation should be cutting
it down — exactly the interaction direction flagged independently by the H-statistic
and residual-A/E methods below. Contents shows the same shape, milder (A/E 0.23-0.36
in the same corner).

## 2. Interaction detection (`interactions.json`)

Three independent methods per sub-model, ranked separately, then cross-checked:

- (a) EBM FAST (`measure_interactions`), 120k-row sample, all of each sub-model's
  current features (30-33 features, not just the top 8).
- (b) Friedman H-statistic, computed manually (2D vs 1D partial dependence on a
  4,000-row background, 6-point grid) from a light single LightGBM fit (300 trees,
  80k-row sample) — a custom, non-normalized implementation, so treat values only as
  a within-line ranking, not calibrated against the textbook [0,1] scale (one motor
  pair scored 1.999, above the formal bound, confirming this).
- (c) Residual two-way A/E against the champion's actual saved fold-B predictions
  (`data/cache/preds/20261019T172626.parquet`), scored by the SD of log(A/E) across
  band cells with at least 0.2% of exposure.

Cross-method agreement, by line:

| Line | Top interaction (methods agreeing) |
|---|---|
| Buildings | `flood_zone` x `elevation_m` — #1 by H-stat (0.62), #1 by residual A/E (logSD 0.26), #2 by EBM (0.0098). `flood_zone` x `distance_to_river_km` close behind on both H-stat (0.53) and EBM. |
| Contents | `flood_zone`/`elevation_m`/`sum_insured` triangle — `sum_insured`x`flood_zone` top of EBM (0.015), `urban_density`x`region` top of H-stat (0.32), `flood_zone`x`voluntary_excess` top of residual A/E (0.29). |
| Motor (other perils) | `annual_mileage` x `prior_claims_5y` dominates H-stat (1.999 — flag per above, but the gap over #2 at 0.33 is large enough to trust the ranking even if not the magnitude); EBM's own top pair is `vehicle_value_group`x`telematics_score` (0.0048), a different signal — the two methods disagree here, worth checking both. |
| Motor (BI frequency) | Weak and noisy everywhere — EBM's largest strength is 5.6e-5, two orders of magnitude below the weakest buildings/contents EBM pair. `cover_type`x`driver_age` is the residual-A/E leader (0.29) but `cover_type` isn't in the BI top-8, so this is a signal for the other-perils side of motor, not BI. Consistent with motor BI's own weak D² (0.0846 fold B) — there just isn't much structure in frequency beyond the main effects already fitted. |

Caveat on contents EBM pairs: `measure_interactions`'s nominal/continuous type
detector (built on top of `experiment._ebm_safe`) only recognises pandas
`CategoricalDtype` and plain `object` dtype as nominal; several contents fields
(`accidental_damage_cover`, `tenure`, `security_level`, `lodgers`, `bicycle_cover`,
`works_from_home`, `credit_band`, plus `region`, `urban_density`, `property_type`,
`trees_within_5m`, `energy_rating`, `solar_panels`) came through as pandas
`StringDtype`, which the check misses, so `measure_interactions` treated them as
ordinal/continuous and binned them by sorted string value. Non-fatal — the run still
produced a ranking — but any EBM-ranked pair involving one of those fields should be
read as directional only. The H-statistic and residual-A/E rankings for contents are
unaffected (both already treat categoricals correctly) and agree with each other on
`flood_zone`-centred pairs, which is the more trustworthy reading.

## 3. How interactions and non-linearity have been handled so far

Reading `git log -p -- src/iaa/experiment.py` (16 commits) and
`research/experiment_backlog.md`: the pipeline has never fitted interactions
explicitly as model features. Three mechanisms have carried all of the interaction
and non-linearity capture to date:

1. Trees capture interactions implicitly. Every production sub-model is LightGBM
   (plus an EBM blend member on buildings/contents). A `num_leaves=15` tree can split
   on up to ~4 features per path, so two- and three-way interactions between the
   top-8 features are already inside the model whenever LightGBM is in the mix — this
   is why `flood_zone`x`elevation_m` shows up correctly signed in the champion's own
   fold-B predictions (section 1) without anyone engineering the pair.
2. EBM main effects only, by owner choice. `EBM_PARAMS["interactions"] = 0`
   (comment: "EBM blended with LightGBM on buildings and contents... now fits once...
   outer_bags back to 2"). EBM's own interaction detection has never been turned on in
   production; it exists in `interpret` but wasn't used before this review (section 2a
   is the first time `measure_interactions` has been run against this feature set).
3. Engineered interactions were tried twice and both failed or were parked. Backlog
   2.2 ("Peril mechanics features: clay soil x trees_within_5m x dry_summer_index...
   flood_zone x elevation x distance_to_river") — tried, "below best." Backlog 2.47
   (sum_insured bands, construction_type x property_age, flood_zone x rainfall_mm,
   soil_type x dry_summer_index, voluntary excess / sum insured — written to replace
   signal lost when `written_premium` was dropped) — the bundled version scored below
   the no-premium baseline; testing families singly has so far covered only
   `excess_sum_insured_ratio` (discarded, results.tsv `d1095f5`) and
   `flood_zone x rainfall_mm` (discarded, `01cd8c8`); the rest of the family list is
   still open.

Section 4's technique (iii) below is a third, fresh data point on the same question,
using the pairs this review's own detection methods ranked highest rather than the
UK-peril-mechanics priors backlog 2.2/2.47 used — and it fails the same way.

A general note worth recording for future owner decisions: a monotone transform of a
single input (square root, log, square) changes nothing for a tree model (trees split
on rank order, not scale) but does change an EBM's or a GLM's main-effect shape —
relevant if EBM's main-effect smoothness ever becomes a lever. A genuine two-feature
interaction, by contrast, is invisible to both EBM-without-interactions and a plain
GLM unless added as an explicit cross term, and is exactly where the diagonal
`flood_zone`x`elevation_m` pattern in section 1 would hurt a GLM the most — consistent
with section 4's GLM result.

## 4. Techniques bake-off (`techniques.json`)

Fold B (train 2021-23 -> valid 2024), within-line D² (`iaa.score.d2`, power 1.5), light
settings throughout: 150k-row sample, 350 trees where applicable, single seed,
`num_threads=3`. Motor is scored as a flat Tweedie-on-total-cost problem here
(ignoring the champion's BI-Poisson/other-Tweedie split, which is an already-settled
owner decision and out of scope for this bake-off) — so motor's numbers below are not
directly comparable to `d2_motor_B=0.08492` in results.tsv; buildings/contents are
closer in spirit to the champion's own LightGBM+EBM member but still lighter
(sampled, fewer trees), so all absolute D² values here sit below the full-pipeline
numbers. Read everything as relative, within line, within this bake-off.

Noise level (the coordinator's correction applies): `experiment.py`'s own comment
("single-seed noise sd 0.0026") describes the full 3-seed-bagged harness model on the
full dataset, not these light single-seed sampled fits, and is reported to now be
stale (today's measured full-harness primary sd is 0.0002). I measured the noise of
these specific light fits directly: 3 reseeds (different seed + different 150k sample
draw each time) of the pruned-feature LightGBM baseline, buildings and contents only
(motor's noise was not separately measured — treat its deltas with extra caution):
buildings sd = 0.0012, contents sd = 0.0006 (`noise.json`). I use twice the measured sd
as the "clearly real" threshold below (buildings +/-0.0024, contents +/-0.0013); motor
deltas are flagged against the rough range these two give (0.0006-0.0012) since no
in-line measurement exists.

| Technique | Motor (flat) | Buildings | Contents |
|---|---|---|---|
| (i) pruned feature set (67->33 / 54->31 / 55->30 features) | 0.07402 | 0.09693 | 0.07966 |
| (i) full valid feature set | 0.06729 (-0.0067) | 0.08870 (-0.0082) | 0.07922 (-0.0004) |
| (ii) EBM, interactions=15 | 0.04173 (-0.0323) | 0.10177 (+0.0048) | 0.08404 (+0.0044) |
| (iii) LightGBM + top-5 explicit interaction pairs | 0.07160 (-0.0024) | 0.09052 (-0.0064) | 0.07751 (-0.0022) |
| (iv) GLM with splines (SplineTransformer + TweedieRegressor) | 0.08155 (+0.0075) | 0.08353 (-0.0134) | 0.06264 (-0.0170) |
| (v) CatBoost, native categoricals | 0.07426 (+0.0002) | 0.10470 (+0.0077) | 0.08334 (+0.0037) |

Deltas are vs. each line's own (i)-pruned row.

Verdicts:

- Pruning is real and should stay. Full-valid-set deltas (-0.0067 motor, -0.0082
  buildings, both several times any plausible noise; -0.0004 contents, inside
  contents' own noise) confirm the owner's 2026-10-19 pruning decision rather than
  contradicting it, on an independently-built sample.
- CatBoost adds real, line-level value on buildings and contents (+0.0077, +0.0037 —
  both outside their measured noise bands) and is roughly flat on motor (+0.0002).
  This matches the repo's own history (backlog 2.13: `b46f2be` +0.00137, `89324f8`
  +0.00183 as a blend member) and lines up with today's `9d37636` run, which added
  CatBoost as a third property blend member and scored `d2_buildings_B=0.11508` /
  `d2_contents_B=0.09515` — both above today's champion — yet was discarded on the
  combined primary/min_fold bar (fold A pulled it down), not on the per-line D². The
  isolated, line-level signal and the full-pipeline per-line numbers point the same
  way; the discard was about fold stability, not about whether CatBoost helps
  buildings/contents. Worth re-testing as a solo addition with its own blend weight
  tuned, rather than folded in alongside other changes as before.
- EBM-with-interactions helps property, hurts motor. +0.0048 buildings, +0.0044
  contents (both outside noise, and EBM is now the best or second-best technique on
  both property lines) vs. -0.0323 on motor — a large, one-sided drop. The property
  gain is consistent with turning on exactly the `EBM_PARAMS["interactions"]=0`
  switch that section 3 noted has never been tried in production; motor's BI tail
  (the 11.9m claim) likely breaks EBM's binned interaction terms on a sampled,
  single-seed fit the same way it breaks everything else about motor without special
  handling, so this result should not be read as "EBM interactions don't work for
  motor" so much as "don't try this on motor without the existing BI/Poisson
  treatment."
- Explicit interaction-as-feature engineering does not help any line (-0.0024,
  -0.0064, -0.0022 — buildings' and contents' deltas both outside noise, in the
  "worse" direction). This is the same conclusion backlog 2.2 and 2.47 already
  reached with differently-chosen pairs; this review reaches it again with
  automatically detected ones (motor other's `annual_mileage`x`prior_claims_5y`,
  buildings' `flood_zone`x`elevation_m`/`distance_to_river_km`, contents'
  `sum_insured`x`flood_zone`/`elevation_m` and three more from each line's own H-stat
  top-5). Trees already have these; adding them as explicit product/combo columns
  just adds redundant, noisier splits for LightGBM to choose between.
- GLM with splines is a genuine surprise on motor (+0.0075, best technique in the
  motor table, beating even pruned LightGBM) but the worst technique by far on both
  property lines (-0.0134 buildings, -0.0170 contents — both far outside noise). The
  likely explanation: motor's signal here (ncb_years, annual_mileage, driver_age,
  years_licensed) is dominated by smooth, roughly monotone main effects that a spline
  GLM fits natively and without a single 11.9m outlier (the flat target here is
  smoothed further than a 350-tree sampled fit can manage), while buildings/contents
  are dominated by the sharp, diagonal `flood_zone`x`elevation_m`/`sum_insured`
  interaction from section 1-2, which a main-effects-only GLM structurally cannot
  see. This is a plausible, testable hypothesis, not a proven one — flag it as such.

## Proposed experiments

### Loop track (`research/experiment_backlog.md`, append under Tier 2)

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 2.70 | Re-test CatBoost as a solo third property blend member (lgbm/ebm/catboost, tune the weight split rather than fixed equal thirds) on top of today's champion, isolated from every other change | This review's light isolated fit: CatBoost beats pruned LightGBM by +0.0077 buildings / +0.0037 contents, both outside measured noise (sd 0.0012 / 0.0006); today's full-pipeline `9d37636` run (discarded) independently showed `d2_buildings_B`/`d2_contents_B` above the current champion despite the combined bar failing it | Buildings/contents D² up if the discard really was a fold-A/min_fold issue and not CatBoost itself | 1019 interactions review, `interactions.json`/`techniques.json`; results.tsv `9d37636`, `b46f2be`, `89324f8` | open |
| 2.71 | Turn on EBM's own interaction detection (`EBM_PARAMS["interactions"]`, e.g. 10-15) on buildings and contents only, keeping the 50/50 LightGBM blend | Light sampled test: EBM interactions=15 beats pruned baseline by +0.0048 buildings / +0.0044 contents (outside noise), and is the best or near-best technique on both; `EBM_PARAMS["interactions"]=0` has never been tried otherwise in production | Buildings/contents D² up a few tenths of a point; do NOT apply to motor | 1019 interactions review, `techniques.json` | open |
| 2.72 | Close out backlog 2.47's remaining untested property interaction families (construction_type x property_age, soil_type x dry_summer_index) as a confirmation that explicit interaction features don't help, before retiring the item | This review's own auto-detected top interactions (flood_zone x elevation_m, sum_insured x flood_zone, etc.), added as explicit LightGBM features, scored worse than the pruned baseline on every line (-0.0064 buildings, -0.0022 contents, both outside noise) — a second, independent confirmation of backlog 2.2/2.47's own findings with a different pair selection | Expect discard; closes the open half of 2.47 | 1019 interactions review, `techniques.json`; backlog 2.2, 2.47 | open |
| 2.73 | A GLM-with-splines blend member on motor only (not property), e.g. 70/30 LightGBM/GLM-spline on the other-perils rate, BI frequency untouched | Light sampled test: GLM-spline alone beat pruned LightGBM by +0.0075 on motor's flat target (best of all five techniques tried), the opposite of its property result (-0.013/-0.017) — motor's signal looks smoother/more monotone than property's | Small motor D² gain if the smoothness story holds; could also be sample-size noise given motor's own noise sd wasn't measured — confirm with reseeded light fits before committing harness time | 1019 interactions review, `techniques.json` | open |

### Owner track (append under Tier 3d)

| # | Idea | Why | Expected | Source | Status |
|---|---|---|---|---|---|
| 3.70 | Decide whether to re-run this review's noise_check for motor specifically (3 reseeds, flat-target light LightGBM) before acting on 2.70/2.73, since buildings/contents noise (sd 0.0006-0.0012) was measured but motor's wasn't, and motor's deltas in `techniques.json` are judged against that proxy only | Items 2.70 and 2.73 both lean on motor-line evidence judged against a borrowed noise estimate; a direct measurement would remove the one real gap in this review's evidence | Either confirms the existing read (cheap, ~2 min) or narrows/widens which deltas count as real | 1019 interactions review, `noise.json` | open |
| 3.71 | Fix `_ebm_safe`'s nominal/continuous type detection to also catch pandas `StringDtype` columns (not just `CategoricalDtype` and `object`), so `interpret.utils.measure_interactions` stops silently treating contents fields like `accidental_damage_cover`, `tenure`, `credit_band`, `region` as ordinal/continuous | Confirmed bug in this review (section 2 caveat): several contents categoricals produced "indicated as continuous" warnings and were binned by sorted string value, degrading EBM's own interaction ranking for those specific fields (the production EBM blend member itself is unaffected, since it never calls `measure_interactions`) | Cleaner contents interaction ranking if this detection is ever used again (e.g. before item 2.71's EBM-interactions test) | 1019 interactions review, `stage2_interactions.py`, `interactions.json` | open |

## Key files

- `~/.cache/iaa-owner/1019/inter/twoway.json` — task 1 deliverable, schema
  `{submodel: [{x, y, x_levels, y_levels, exposure[][], actual[][], predicted[][], ae[][]}]}`.
- `~/.cache/iaa-owner/1019/inter/interactions.json` — task 2 (EBM/H-stat/residual rankings).
- `~/.cache/iaa-owner/1019/inter/noise.json` — measured light-fit noise (buildings sd 0.0012, contents sd 0.0006).
- `~/.cache/iaa-owner/1019/inter/techniques.json` — task 4 bake-off.
- `~/.cache/iaa-owner/1019/inter/top8.json`, `common.py`, `stage1_twoway.py`, `stage2_interactions.py`, `stage3_techniques.py`, `noise_check.py` — the scripts behind all four, kept for reproducibility.
