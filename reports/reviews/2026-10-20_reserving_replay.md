# Backlog 3.9-lite: as-if reserving replay

Evaluation only. Nothing here changes the modelling target; this is a check of
candidate claim-development / ultimate methods before any of them is adopted.
All code and intermediate output live under `~/.cache/iaa-owner/1020/replay/`;
the competition repository at `~/Projects/iaa-hackathon-2026` was read-only
throughout (imported as a library, never edited).


> **Corrections after owner review (20 Oct, afternoon).** (1) The `BF_priorB_foldA` prior used a full year of expected cost at as-if dates part-way through 2023; scaled to the exposure earned by the as-if date it matches `BF_priorA` at every date (pooled -9.2% vs -9.0% on the same cells). The "+32%" result and its explanation in section 9 are withdrawn. (2) The replay applied the BI `max(1, .)` floor that the owner removed from the model level on 20 Oct; rerun without it (`out_v2`, `sim/nofloor`), the capped-severity variants separate from the current method, the simulated-book result holds (current +1.5%, BF/Cape Cod/Benktander -14.5%), and no method can be ranked on BI cost because one 11.9m claim exceeds every between-method difference. (3) The buildings-fire "90%" is on late-reported fire cost of about 0.3m per date, 2% of a year's fire ultimate; the credibility recommendation is withdrawn. (4) The 11.9m claim takes 12.8m of the fixed 2023 frequency x severity total, leaving the other 2023 BI rows about 21% below neighbouring years per vehicle-year. Explainer with charts: `2026-10-20_replay_explained.html`.

## 1. Limitation: this is a snapshot, not a valuation history

The claims table is a single extract valued 31 December 2024. It carries no
valuation history and no claim-close date, so there is no way to ask "what did
the case reserve look like on 30 June 2023." The only field that lets a claim
be placed in time relative to an as-if date is `report_date`. An as-if
valuation at date D can therefore only censor on reporting:

  known at D = claims with `loss_date <= D` and `report_date <= D`, valued at
  the claim's actual 31-Dec-2024 net cost (not at whatever the case reserve
  happened to be worth on D, which is unobservable).

That means this replay tests pure IBNR (how many claims had not yet been
reported, and roughly how large they would turn out to be) but cannot test
IBNER (case-reserve development on claims already reported at D). The
project's own frozen code treats open-claim case reserves as unbiased
(`ibner_open_factor = 1.0`); this replay inherits that same assumption and
cannot confirm or refute it. Any method that leans on case-reserve adequacy
is being graded here on reporting speed and initial cost only.

As-if dates used: 31 Dec 2022, 31 Mar 2023, 30 Jun 2023, 30 Sep 2023, 31 Dec
2023. 31 Dec 2021 was dropped: the claims table starts at AY2021, so at that
date there is no development history at all to fit a pattern from -- not "a
thin pattern," but none.

## 2. Design

Cohort grain is (line, peril group, accident year), not loss month. At as-if
date D, a cohort's "elapsed" loss months are those that had already occurred
by D (a partial current year is naturally restricted to its elapsed months,
since a claim can't exist with a loss date after D if `report_date <= D`
implies `loss_date <= D`). Within-year timing is handled by averaging the
monthly completion fraction (from the fitted development CDF) over the
elapsed loss months of that accident year, an exposure-uniform approximation
of the pipeline's own `row_completion` logic.

For every (line, group) the monthly report-delay CDF is refit at each as-if
date using only claims with `report_date <= D`, by calling the pipeline's own
frozen `iaa.ultimate.delay_cdfs` with `valuation_date` set to D and the input
restricted to the censored claim set -- i.e. the identical chain-ladder
arithmetic the pipeline uses for its "current" method, just fit on less data.
This was checked deliberately rather than assumed: see the reconciliation in
section 7.

For a cohort at age a (months, at D) with ultimate estimate U and a fitted
cumulative pattern F, predicted emergence between D and 31 Dec 2024 (age b)
is `U x (F(b) - F(a))`. No comparison in this report sets a full ultimate
against a partial emergence; all scoring is emergence-to-emergence.

Candidates implemented:

- `CL_current`: the pipeline's current config -- cost chain ladder per peril
  group; motor bodily injury via frequency x severity (ultimate count from a
  count chain ladder, severity = base-year-2021 mean net cost trended at
  3.5%/yr), with the same floor rule the pipeline uses (the freq-sev scaling
  factor cannot go below 1.0, so once a large claim is actually known, the
  freq-sev estimate cannot fall below the raw chain-ladder figure that
  already contains it).
- `b35_trend{0.030,0.035,0.040}_by{2021,2022}` and `b35_fitted_by{2021,2022}`:
  backlog 3.5 motor-BI variants -- severity capped at 1,000,000 per claim,
  trended at a fixed or log-linear-fitted rate, plus a flat excess load per
  ultimate claim (mean of `max(net - cap, 0)` over claims at report delay
  <= 12 months, pooled across accident years mature enough to have reached
  that delay as of D). The log-linear fit and the base severities are
  themselves computed at equal maturity (delay <= 12 months, years that have
  reached that maturity as of D) so the trend is not confounded with the
  known tendency of late-reported claims to be larger.
- `BF_priorA` / `CapeCod_priorA`: Bornhuetter-Ferguson and Cape Cod on cost,
  per peril group. Prior = trended chain-ladder ultimate per unit of earned
  exposure from loss years fully earlier than the cohort (Cape Cod derives
  its ELR the standard way, sum(reported)/sum(exposure x %reported), over
  those same earlier years, self-derived from the censored data each time).
  The trend applied is the config's own line/peril-group trend; a censored
  refit of the trend itself was not attempted (only 1-2 prior years are
  available at the earliest as-if dates, not enough to fit a trend without
  it being noise).
- `BF_count_sev` / `CapeCod_count_sev`: BI only, same BF/Cape Cod machinery
  run on counts, multiplied by the current severity assumption, as the brief
  asked for ("on counts, then x severity as in freq_sev").
- `BF_priorB_foldA`: BF prior taken from the fold-A pricing model's
  row-level ultimate-cost predictions (the kept run's fold-A validation
  parquet), summed by line and split to peril group by that group's share of
  raw reported cost in AY2021-2022. Used only for the AY2023 cohort at as-if
  dates <= 31 Dec 2023, per the brief (fold A trained on AY2021-22, valid on
  AY2023, so it is a legitimate as-if-2022 prior for AY2023 -- noting that
  the fold-A *target* it was trained against was itself developed with the
  full 2024 report pattern, so the prior is as-if in which years went in,
  not in what "ultimate" meant when it was built).

Threads capped at 4 (`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`)
throughout; nothing here fits a model, so this was never a real constraint.

## 3. Observed development horizon by as-if date

The fitted CDF's tail is 1.0 by construction wherever no cohort has reached
that delay yet, so every method implicitly assumes zero further emergence
beyond the horizon actually observed at D:

| as-if date | max observed report delay (months) |
|---|---|
| 2022-12-31 | 22 |
| 2023-03-31 | 26 |
| 2023-06-30 | 27 |
| 2023-09-30 | 31 |
| 2023-12-31 | 31 |

This horizon is well short of the config's 48-month max delay at every as-if
date tested. It did not visibly distort the results here (see section 4),
because the great majority of net cost in every peril group reports within
12-24 months, but a method that depended on genuine 36-48 month tail
development would not get a fair test from this dataset.

## 4. Results by method

Pooled across all 5 as-if dates, 7 peril groups and however many accident
years had emerged business at each date (98 cohort-method cells for CL,
narrower for BF/Cape Cod where a prior year's data did not yet exist):

| method | cost actual | cost pred | rel. error (cost) | count pred vs actual | Poisson deviance (counts, exposure-wtd) |
|---|---|---|---|---|---|
| CL_current | 173.33m | 170.51m | -1.6% | -3.0% | 3.14 |
| BF_priorA | 173.33m | 166.40m | -4.0% | n/a | n/a |
| CapeCod_priorA | 173.33m | 166.44m | -4.0% | n/a | n/a |
| BF_priorB_foldA (AY2023 only) | 102.40m | 135.28m | +32.1% | n/a | n/a |

(BF/Cape Cod here produce a cost ultimate only, except the BI count variants
below; "n/a" is not zero, it means no count prediction exists to score.)

By peril group (CL vs BF_priorA, cost relative error):

| line | group | CL_current | BF_priorA |
|---|---|---|---|
| motor | bodily_injury | -0.4% | -3.7% |
| motor | other | -3.8% | -3.6% |
| buildings | weather | -15.2% | -14.4% |
| buildings | water | -11.2% | -11.9% |
| buildings | fire | +90.2% | +90.0% |
| buildings | ad | +14.9% | +15.2% |
| contents | all | +7.7% | +7.7% |

Two things stand out. First, BF/Cape Cod with the exposure-trend prior track
`CL_current` closely everywhere except motor BI -- group by group the
relative errors move together, which says the prior is adding almost no
information beyond what the chain ladder's own reported-to-date already
carries, for these cohort sizes and maturities. Second, buildings fire misses
by about 90% under every method tried, in the same direction, by almost the
same amount -- that is a volume problem (24-449 claims a quarter across the
five as-if dates, concentrated in a handful of large losses), not a method
problem. No candidate method fixes a thin-book problem; a method-only lens
would wrongly credit or blame whichever method happened to be running when a
large fire claim landed.

Backlog 3.5 motor-BI variants (severity trend re-selected on capped severity,
excess layer separate), pooled over all as-if dates and accident years:

| variant | rel. error (cost) |
|---|---|
| CL_current | -0.4% |
| b35_trend 3.0% / base 2021 | -1.7% |
| b35_trend 3.5% / base 2021 (closest to current's own 3.5%) | -1.4% |
| b35_trend 4.0% / base 2021 | -1.1% |
| b35_trend 3.0% / base 2022 | -1.0% |
| b35_trend 3.5% / base 2022 | -0.9% |
| b35_trend 4.0% / base 2022 | -0.7% |
| b35_fitted (log-linear, base 2021) | -0.4% |
| b35_fitted (log-linear, base 2022) | -0.4% |

All of these sit within about 1.5 percentage points of `CL_current` and of
each other, in the pooled figures, for a reason worth stating plainly: the
pipeline's own floor rule (the freq-sev scaling factor cannot fall below 1.0)
applies identically here, so for any cohort where a claim the size of the one
in section 5 is already known, every variant's cost prediction collapses back
onto the raw chain-ladder figure that already contains it. The cap-plus-excess
design only has room to differ from the current method in the window before
such a claim is reported -- and in that window (as-if dates 31 Dec 2022
through 30 Sep 2023, before the claim discussed in section 5 was reported) none
of the capped-severity variants anticipate a loss of that size either, by
design (a flat excess load spreads expected large-claim cost thinly across
all claims; it does not predict one). So the backlog-3.5 variants are not
distinguishable from the current method on this evidence, in either
direction -- they do not resolve the single-claim risk the current method
also cannot resolve, and the severity-trend reselection moves the pooled
number by only a percentage point or two.

## 5. The AY2023 cell at 31 Dec 2023 (0-11 months maturity)

At 31 Dec 2023, the AY2023 cohort for every group is at most 11 months
mature -- the only maturity window in this replay where the 2024 loss year
(fully in the past by now, "today" in the hackathon's own time frame) is
still in the future of the as-if date. This is also the cell containing claim
`CL30973963` (motor bodily injury, net 11,894,276.93, loss 25 Jun 2023,
reported 16 Dec 2023 -- already known as of 31 Dec 2023, so it inflates the
*known* reported-to-date figure rather than the emergence being predicted).

| method | predicted emergence | actual emergence |
|---|---|---|
| CL_current | 26.45m | 13.99m |
| BF_priorA / CapeCod_priorA | 23.79m / 23.80m | 13.99m |
| BF_priorB_foldA | 23.81m | 13.99m |
| BF_count_sev / CapeCod_count_sev | 18.80m / 18.90m | 13.99m |

Every method overpredicts this cell by 34-89%, and the count-based BI
variants (which multiply an ultimate claim count by a severity assumption
that does not know about this one claim either) come closest, because they
are not mechanically dividing an already-inflated known amount by a small
completion fraction the way the raw cost chain ladder does. This is the
clearest single piece of evidence in the replay that a single very large
claim, once known, will distort a cost chain ladder's near-term ultimate for
every other method that still treats it as representative of what remains to
emerge -- and that a frequency x severity formulation is structurally less
exposed to that distortion, even though (section 6) its long-run cost
accuracy for BI is biased low for an unrelated reason.

Motor other, buildings and contents cohorts at the same as-if date are far
closer (see `ay2023_at_dec23.csv` / the `ay2023_cell_at_2023_12_31` block in
replay.json): errors of roughly -9% to +19%, consistent with ordinary
small-cohort noise rather than a single-claim effect.

## 6. Motor bodily injury, in focus

Including the giant claim (pooled across all 5 as-if dates and the AY2021-24
cohorts that had emerged business): `CL_current` actual 128.34m vs predicted
127.82m (-0.4%). Excluding it from both the actual and the predicted side
(the only fair way to exclude a claim -- removing it from truth without
removing it from the triangle used to produce the prediction would overstate
the error):

| | cost actual (ex giant) | cost pred (ex giant) | rel. error |
|---|---|---|---|
| CL_current | 104.56m | 120.56m | +15.3% |

With the giant claim removed, `CL_current` overpredicts BI by 15%. Put the
two results side by side and the method's behaviour is exactly what the
freq-sev design is meant to do: it holds a cushion for exactly the kind of
large claim BI is known to produce, which looks like a clean hit when a large
claim lands in the window being predicted and looks like a 15% overstatement
when one does not. Over a longer run than five quarters this is the expected
trade-off of a frequency x severity formulation with an uncapped severity
input, not a flaw particular to this replay.

The brief's caveat applies throughout this section: "emerged" BI cost is the
claim's case estimate as of 31 Dec 2024, which the pipeline's own
documentation and the project owner's prior review (see
`reports/reviews/2026-10-19_claims.md` in the project, referenced from
`configs/market_dev.yaml`) already treat as understated for open BI claims.
So every cost figure in this section -- actual and predicted alike -- is
biased low by the same unknown amount; the count figures (rel. error -2.3%
to -3.9% depending on method, in the group table in section 4) are not
subject to that bias and are the more trustworthy comparison between methods
for BI.

## 7. Chain-ladder cross-check (chainladder-python)

chainladder-python 1.x was installed into its own environment
(`~/.cache/iaa-owner/1020/replay/.venv-cl/`, MPL-2.0, owner-approved) and run
against two triangles exported from the replay's claim data: the full-history
motor bodily-injury triangle, and the buildings/accidental-damage triangle
censored at 31 Mar 2023. `Development(average="volume")` was fit to each and
its cumulative-development-to-ultimate factors compared against this
replay's own fitted CDF (converted to the same "share of ultimate reported"
units). Agreement is directional rather than exact -- for buildings AD at 31
Mar 2023, both put over 98% of ultimate reported by development month 2;
for motor BI full-history, both show the same monotonic build-up from ~17-20%
at month 1 to full maturity by month ~44. The two did not match to high
precision (differences of 1-5 percentage points at individual delays); the
most likely cause is a discretization artefact in the chainladder-python
triangle construction (development periods were approximated as fixed 30-day
steps from each origin month, which drifts from true calendar months by a
few days per step and can shift which claims land in which development
bucket at the margin), not a difference in the underlying chain-ladder
algorithm. Both are volume-weighted link-ratio chain ladders with a tail
factor of 1.0 past the data's horizon; the cross-check confirms this
replay's own development fit is a standard chain ladder and not, for
example, an off-by-one error in the delay index or a double-count in the
cumulative triangle. Full factor tables: `out/chainladder_crosscheck.json`.

## 8. Effect on the 2021-2024 modelling target

At the real valuation date (31 Dec 2024, full data, no censoring), this
replay's own cohort-level chain ladder reproduces the pipeline's row-level
`developed_targets()` ultimate within 0.2% in every line and year (see
`out/reconciliation.csv`):

| line | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| motor | 1.0009 | 1.0012 | 1.0007 | 0.9985 |
| buildings | 1.0004 | 1.0008 | 1.0019 | 0.9995 |
| contents | 1.0013 | 1.0012 | 1.0010 | 1.0007 |

(ratio = this replay's native CL / the pipeline's own developed ultimate, by
line and accident year, summed across that line's peril groups). This
confirms the replay's method is measuring the same thing the pipeline
measures, not a different construction that happens to look similar.

Candidate methods' effect on the 2021-2024 ultimates, as a ratio to
`CL_current` (full `ultimates_pivot.csv` has every line/group/year):

- Every non-BI peril group: `BF_priorA` sits within 0.2% of `CL_current` in
  every year, including 2024 (the least mature year). By 31 Dec 2024 these
  books are mature enough that a prior adds essentially nothing to a chain
  ladder that already has most of the triangle filled in.
- Motor bodily injury is the one group where method choice visibly moves the
  target: `BF_priorA` is 0.6-0.9% above current for 2022-2023 but 9.4% below
  current for 2024 (the prior pulls the least mature year down, since BI's
  own chain ladder is particularly volatile at low maturity and the prior
  is comparatively stable). The backlog-3.5 severity-trend variants move the
  opposite way and by more: roughly +4 to +5% above current for 2022, +2 to
  +4% for 2023, and +3 to +7% for 2024, scaling up with the trend rate
  (3.0% through 4.0%) and depending on which base year (2021 or 2022) is
  used to anchor the severity. Base year 2022 with a given trend rate always
  produces a slightly higher ultimate than base year 2021 at the same rate,
  because 2022's own capped severity already sits above a 3-3.5%-trended
  projection from 2021.

## 9. Recommendation

- **Motor bodily injury is worth a real decision, not a default.** The
  current freq-sev design (base year 2021, 3.5% trend, no cap) and the
  backlog-3.5 capped-severity variants are not distinguishable from each
  other on this replay's cost evidence (section 4) because the pipeline's own
  floor rule erases the difference the moment a large claim is known -- but
  they imply materially different 2024 ultimates once there is no large known
  claim to floor against (section 8: current vs backlog-3.5 differ by 3-7
  points on 2024). That is a real choice about how much cushion to hold for
  BI's long tail of large claims, and it should be made as an explicit
  severity-cap-and-trend decision, not left as a side effect of whichever
  variant happens to get run next. The count-based BI variants
  (`BF_count_sev`/`CapeCod_count_sev`) are worth adopting as the BI count
  check going forward: they come closest on the one cell that stress-tests
  this (section 5) and are immune to the cost-side bias in section 6.
  The simulated book (section 11) sharpens this: against a known true ultimate, BF/Cape Cod/Benktander understate motor BI by a consistent 14-15% (std well under 1 point across three seeds) while `CL_current` is within 2%. That is a structural bias, not noise, and on its own is reason enough not to move BI onto BF/Cape Cod/Benktander.
- **BF/Cape Cod with the exposure-trend prior are not distinguishable from
  the current chain ladder for motor other, buildings or contents**, at any
  as-if date tested (section 4, section 8). Adopting either as the target
  for these groups would not change the 2021-2024 numbers materially and
  would add a dependency (the prior trend selection) that the current method
  doesn't need. Not recommended as a target change for these groups.
- **Buildings fire needs a credibility or capping treatment, not a different
  development method.** Every method missed this group by about the same 90%
  in the same direction (section 4); the fix is a volume/credibility
  adjustment (blend with a wider buildings book, or an excess layer, the way
  BI already has one) rather than choosing among CL/BF/Cape Cod.
- **The fold-A pricing-model prior (`BF_priorB_foldA`) should not be adopted
  as a BF prior.** It overpredicted the one cohort it was tested on by 32%,
  worse than the simple exposure-trend prior's -4.0% on the same comparison
  basis. Plausible reason: the fold-A model's own training target was itself
  built with the *full* 2024-12-31 development pattern, so it inherits
  whatever bias the pattern itself carries, on top of being trained on only
  two years of data.
- The IBNER/case-reserve-development question raised in section 1 cannot be
  answered from this data at all; before leaning further on BI case
  estimates, the modelling target should state explicitly (as `market_dev.yaml`
  already gestures at) that `ibner_open_factor = 1.0` is an assumption, not a
  measured fact.

## Files

- `out/claims.parquet`, `out/policy_periods.parquet`, `out/preds_foldA.parquet`,
  `out/group_share.parquet`, `out/pipeline_ultimates_by_year.parquet`: exported
  base tables (project venv, `scripts/00_export.py`).
- `scripts/replay_lib.py`: core cohort/CDF/BF/Cape Cod arithmetic.
- `scripts/01_replay.py`: the as-if replay loop -> `out/predictions.parquet`,
  `out/score_*.csv`, `out/diagnostics.csv`, `out/ay2023_at_dec23.csv`.
- `scripts/02_valuation_table.py`: 2021-2024 ultimates at the real valuation
  date + pipeline reconciliation -> `out/ultimates_pivot.csv`,
  `out/reconciliation.csv`.
- `scripts/03a_export_triangle.py` / `03b_cl_crosscheck.py`: chainladder-python
  cross-check (needs the `.venv-cl` environment for the second script) ->
  `out/chainladder_crosscheck.json`.
- `scripts/04_build_replay_json.py`: assembles `replay.json`.
- `scripts/replay_core.py`: the reusable `run_replay(...)` function (backlog 3.9-lite
  infrastructure item) -- same method set and scoring, callable against any
  `data.claims()`-schema frame plus a (line, loss_month, exposure) table.
- `scripts/05_blending.py`: backlog 3.11-lite fixed and fitted method pools ->
  `out/blending.json`.
- `scripts/06_sim_check.py`: backlog 3.12 simulated-book check (runs once
  `~/.cache/iaa-owner/1020/sim/DONE` exists) -> `sim/sim_score_emerged.csv`,
  `sim/sim_score_vs_truth.csv`, `sim/sim_score_bi_vs_truth.csv`.

## 10. Method blending (backlog 3.11-lite)

The replay was restructured so the whole method set and scoring run from one function,
`replay_core.run_replay(claims, exposure, valuation_date, as_if_dates, cfg, ...)`, taking a
`data.claims()`-schema frame, an exposure frame (line, loss month, exposure -- no peril-group
split), and a valuation date, and returning scored predictions for every candidate method.
`scripts/01_replay.py` now just loads the real snapshot and calls it; the same function is
what `scripts/06_sim_check.py` (section 11) calls on the simulated book.

Pools tested, all evaluated on the same 98-cohort pooled basis as section 4 (cost, giant
claim included):

| pool | rel. error (cost) |
|---|---|
| CL_current (best single method) | -1.6% |
| BF_priorA | -3.9% |
| CapeCod_priorA | -3.9% |
| 50/50 CL + BF | -2.8% |
| 50/50 CL + Cape Cod | -2.7% |
| Benktander (one BF-on-BF iteration) | -4.7% |

None of the fixed pools beat `CL_current` in this in-sample comparison; a 50/50 blend
lands between its two parents by construction, and since `CL_current` is already the
closer of the two to actual, diluting it with BF or Cape Cod only pulls the pooled error
away from zero. Benktander (iterating BF once more, using the first BF ultimate as the
new prior) comes out worse than plain BF here, consistent with BI being the group where
the single-claim effect in section 5 dominates and a second BF iteration leans further on
a prior that is, for BI specifically, less informative than the raw chain ladder (section
4's per-group table).

A fitted linear pool of the two closest contenders, `w x CL_current + (1-w) x BF_priorA`,
was fit by leave-one-as-if-date-out: `w` chosen by least squares on the other 4 dates,
scored on the held-out date, so the weight is never judged on the data that picked it.
The fit landed on a corner solution every time (`w` = 0 or 1 in all 5 folds -- see
`out/blending.json` for the per-date weights), meaning no interior blend ever beat picking
one of the two methods outright in-sample on the training folds. Out of sample:

| | mean abs. relative error across the 5 held-out folds |
|---|---|
| CL_current alone | 19.2% |
| BF_priorA alone | 17.0% |
| fitted blend (OOS) | 18.9% |

The fitted blend is worse than the better single method (BF_priorA) and about level with
the worse one (CL_current). Date-to-date spread of the better single method's own error
(max fold error minus min fold error) is 51 percentage points -- the blend's apparent
"improvement" over the worse single method (0.3 points) is nowhere near that spread, and
it doesn't even beat the better single method. **No pool is supported out of sample.**
The honest reading is that these methods are not diversified enough, at this cohort count
and maturity range, for blending to add anything; adopt the single best method per group
(section 9) rather than a pool. Full numbers: `out/blending.json` /
`replay.json -> blending_backlog_3_11_lite`.

## 11. Simulated book with known truth (backlog 3.12)

A separate agent built a simulated claims book with known true ultimates
(`~/.cache/iaa-owner/1020/sim/outputs/`, three independent seeds, each calibrated to the
real book's frequency, severity, report-delay and settlement patterns by line and peril
group -- its own calibration report found all four within a few percent of the real data,
motor BI's open-claim age/severity understatement pattern matching to the third decimal
place). `scripts/06_sim_check.py` runs the identical `replay_core.run_replay` function
against each seed's simulated claims/exposure and scores every method both ways: against
emerged-in-snapshot (same construction as section 4) and against the seed's true ultimate
cost (from its own `truth_{seed}.parquet`, not a sum over `sim_claims`' `true_ultimate_net`
-- see the truncation note below).

**Important correction made during this check, not a result on its own:** `sim_claims` is
right-truncated at 31 Dec 2024 exactly like the real book (a claim with loss date in 2024
that would not report until after the snapshot is simply not a row yet). Summing
`true_ultimate_net` only over the claims present in the snapshot therefore understates the
true ultimate of the least mature cohorts -- confirmed directly: AY2024 motor BI carries
1,221 claims in `sim_claims` against `truth.parquet`'s true count of 2,158, a 43% gap, with
the gap at 0% for fully mature (2021-2022) cohorts. This is the same right-truncation this
whole replay exists to study, reproduced faithfully by the simulator -- not a flaw in it or
in this check -- but it means the sim's own independently-computed `truth_{seed}.parquet`
has to be the truth reference, not a rollup from the truncated claim rows. All results
below use `truth_{seed}.parquet`.

**Does the replay code recover a known answer?** Pooled across all 5 as-if dates, 7 peril
groups and 3 seeds, emerged-in-snapshot cost error averages +5.7% (std 5.3pp across seeds)
for `CL_current`, +1.2% for `BF_priorA`/`CapeCod_priorA`, and +0.06% for `Benktander` -- all
small and all the same sign, with no method badly miscalibrated. This is reassurance that
the replay's arithmetic is sound (a code bug would be far more likely to show up as a large
or sign-inconsistent error than a consistent low-single-digit one), not evidence that any
one method is "best": these are the same noise-level differences seen on the real book in
section 4, now confirmed not to be an artifact of the real book's specific history.

**How does each method handle the motor-BI case-reserve understatement?** This is the
question the real snapshot cannot answer (section 1), because the real data has no
independent truth. Comparing each method's BI ultimate estimate against the sim's known
true ultimate, pooled across as-if dates and all 3 seeds:

| method | mean rel. error vs true BI ultimate | std across seeds |
|---|---|---|
| CL_current (freq-sev) | +1.8% | 0.5pp |
| BF_priorA | -14.5% | 0.7pp |
| CapeCod_priorA | -14.5% | 0.7pp |
| Benktander | -14.5% | 0.3pp |

This is the clearest finding backlog 3.12 was built to surface. The current freq-sev
method recovers motor BI's true ultimate to within 2%, consistently across seeds (the sim
does not even include a single dominating claim the way the real book does, so this is
not section 6's cushion effect -- it is the frequency x severity design doing what it is
meant to do on an otherwise well-behaved book). BF, Cape Cod and Benktander all understate
the true BI ultimate by about 14-15%, tightly and consistently (standard deviation well
under 1 percentage point across three independent seeds, so this is a structural bias in
how the exposure-trended prior handles BI's slow-developing, right-skewed reporting
pattern, not sampling noise). **This directly supports the section 9 recommendation
against adopting BF/Cape Cod/Benktander for motor bodily injury**, on grounds the real
data alone could not have established: the real book only showed these methods tracking
`CL_current` to within a few percent (section 4), which on its own looked like "no
material difference"; the simulated book with known truth shows that similarity was
coincidental to the specific claims that happened to be large in the real window, and
that BF/Cape Cod/Benktander carry a real, structural downward bias for this group.

Across all 7 peril groups pooled, every method understates the true ultimate somewhat
(`CL_current` -7.1%, `BF_priorA`/`CapeCod_priorA`/`Benktander` around -10.7%), consistent
with immature cohorts being hard to estimate from the development pattern alone --
`CL_current` is the least biased of the four on this measure too. Full per-seed detail:
`sim/sim_score_bi_vs_truth_summary.csv`, `sim/sim_score_vs_truth_summary.csv`,
`sim/sim_truth_reconciliation.csv`, `sim/sim_score_emerged_summary.csv`.
