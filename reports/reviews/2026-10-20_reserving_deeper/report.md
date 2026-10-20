# Reserving replay extension: backlog 3.2, 3.4, 3.10, 3.40

Evaluation only. Nothing here changes the modelling target; this extends the as-if
reserving replay (`~/.cache/iaa-owner/1020/replay/report.md`) with four further checks.
Code and output live under `~/.cache/iaa-owner/1020/reserving2/`; the competition
repository at `~/Projects/iaa-hackathon-2026` was read-only throughout (imported as a
library, never edited). Threads capped at 2 throughout.

Every item is scored two ways: on the real snapshot, censored at the same five as-if
dates as the base replay (31 Dec 2022, 31 Mar/Jun/Sep/Dec 2023), emergence-to-emergence
against what actually reported by 31 Dec 2024; and on the three simulated books
(`~/.cache/iaa-owner/1020/sim/outputs/`), which carry an independently-computed true
ultimate, so the real book's "does this look reasonable" can be checked against "does
this recover a known answer." The base replay's own floor removal (`REPLAY_FLOOR=0`) and
exposure prorating (`REPLAY_PRORATE=1`) are carried over here.

## 1. Report-delay model for BI counts: right-truncated MLE vs the count chain ladder

The pipeline's frequency x severity design for motor bodily injury takes its claim count
from a count chain ladder (the same nonparametric, empirical development-factor method
used for cost). Backlog 3.2 asks whether a parametric report-delay model, fit by maximum
likelihood with explicit right truncation at the as-if date, does better.

A claim with loss month `lm`, observed at as-if date D, is only in the data if its report
delay `d` is no more than `T = (months from lm to D)`. The fitted model is a continuous
delay distribution (lognormal and Weibull both tried; AIC picks lognormal in all but one
cell) maximizing `sum[log(F(d+1) - F(d)) - log(F(T+1))]` over the censored claims of one
(line, group) at one as-if date. The fitted CDF is normalized so `F(48) = 1`, the same
convention the frozen pipeline's own empirical CDF uses (zero further emergence assumed
past the config's 48-month cap) -- so the only thing that can differ between the two
methods is the curve's shape between the as-if date's observed horizon (22-31 months,
per the base replay's section 3) and 48 months, not an extrapolation past the config's
own limit.

Pooled over all 5 as-if dates and all accident years, real book, count emergence:

| line | group | actual | CL count (current) rel. error | MLE rel. error |
|---|---|---|---|---|
| motor | bodily_injury | 5,148 | -2.4% | +53.6% |
| motor | other | 5,567 | -0.4% | +243.3% |
| buildings | weather | 1,253 | -20.3% | +105.2% |
| buildings | water | 684 | -6.2% | +213.3% |
| buildings | fire | 24 | +30.2% | +599.5% |
| buildings | ad | 161 | +17.7% | +323.2% |
| contents | all | 1,122 | -1.2% | +283.8% |

The MLE method overstates emergence everywhere, and badly so for the fast-reporting
groups. The reason is visible in the fitted curve itself (`item1_bi_delay_curve.png`):
a continuous lognormal or Weibull assigns essentially zero density to delay-zero
reporting, because both are smooth distributions with no mass concentrated at the origin.
Several of these books report a large share of claims within the same calendar month as
the loss (motor other: 62% at delay 0 under the empirical CDF; buildings water, ad and
fire are similar) -- a near-instant-reporting hump a smooth continuous curve cannot match.
The MLE fit compensates by pulling probability mass later across the whole curve, which
understates completion at the ages that matter for a current, still-developing accident
year, and the resulting ultimate-count estimate inflates sharply.

Motor bodily injury is the slowest-reporting group in the book (20% complete by delay 1,
87% by delay 12 at this as-if date) and is the one group where the parametric shape comes
closest to matching the empirical curve -- which is exactly why it is still off by 54%
pooled, not 5x like the fast groups, but still materially worse than the chain ladder's
-2.4%.

The simulated books confirm this is not an artifact of the real book's own history, using
the known true ultimate count at 31 Dec 2023 (`item1_sim_ultimate_count_error.png`):

| line | group | CL rel. error (mean, 3 seeds) | MLE rel. error (mean, 3 seeds) |
|---|---|---|---|
| motor | bodily_injury | +0.3% | +9.4% |
| motor | other | -0.0% | +3.1% |
| buildings | weather | -0.1% | +2.7% |
| buildings | water | -0.1% | +3.0% |
| buildings | fire | -0.1% | +3.0% |
| buildings | ad | +0.0% | +3.1% |
| contents | all | +0.0% | +3.1% |

Same pattern, smaller magnitude because the sim's own reporting-delay generator is itself
a smooth lognormal/Pareto mixture (no artificial delay-zero spike), so the MLE's
functional-form mismatch is milder -- but it is still consistently worse than the chain
ladder in every group, including BI, by 3 to 9 points.

Recommendation: no change. The count chain ladder already handles right truncation
correctly and nonparametrically; a smooth parametric delay distribution is the wrong tool
for data with a reporting spike at or near delay zero, which describes every group here to
some degree and most groups severely. If a parametric delay model is wanted for a future
backlog item (e.g. to get a smooth hazard for extrapolation past 48 months), it needs a
discrete point mass or a mixture term at delay 0, not a bare lognormal/Weibull.

## 2. Cross-sectional case-adequacy diagnostic (Berquist-Sherman style), motor BI

With one valuation date there is no way to watch a cohort's own case reserve run off --
the textbook Berquist-Sherman compares successive diagonals of the same triangle. The
substitute tried here is cross-sectional: bucket claims by age since loss (days from
`loss_date` to 31 Dec 2024), compare the mean carried value of claims still open at that
age against the mean settled cost of claims that closed at a similar age, trended to a
common price level at the config's own 3.5%/yr BI severity trend. If young open claims
carry less on average than same-age claims that have already closed, that is consistent
with case understatement.

Two structural problems apply before any number is read. First, age and accident year are
almost perfectly confounded in a single snapshot: the youngest age bucket is populated
almost entirely by AY2024 claims, the oldest almost entirely by AY2021, so "by age" is
very close to "by accident year," and the trend adjustment is carrying most of the
cross-year comparability. Second, claims that close young are a selected population --
typically smaller, simpler claims -- so "settled at a similar age" systematically
under-represents the large, slow claims that make BI understatement material in the first
place.

Real book (giant claim `CL30973963` excluded, as in the base replay's section 5):

| age bucket (days since loss) | n open | n closed | open mean (trended) | closed mean (trended) | diagnostic ratio |
|---|---|---|---|---|---|
| (0, 180] | 356 | 44 | 15,439 | 20,478 | 0.75 |
| (180, 365] | 584 | 194 | 19,515 | 16,538 | 1.18 |
| (365, 545] | 566 | 432 | 17,172 | 18,776 | 0.91 |
| (545, 730] | 448 | 606 | 18,789 | 18,023 | 1.04 |
| (730, 1000] | 484 | 1,169 | 22,392 | 25,184 | 0.89 |
| (1000, 1500] | 444 | 2,236 | 22,404 | 23,585 | 0.95 |

No clean monotonic pattern -- the ratio bounces between 0.75 and 1.18 with no trend by
age, which given 44-606 closed claims per younger bucket is consistent with sampling noise
swamping a real but modest understatement signal, not with an absence of understatement.

The simulated books, with a known true understatement factor by the same age buckets
(0.698 youngest to 0.899 oldest, from the simulator's own calibration), show the
diagnostic does recover the right pattern when the underlying signal is real and the
sample is large (`item2_case_adequacy_diagnostic.png`):

| age bucket | true factor | diagnostic estimate (mean, 3 seeds) | error |
|---|---|---|---|
| (0, 180] | 0.698 | 0.668 | -0.030 |
| (180, 365] | 0.755 | 0.784 | +0.029 |
| (365, 545] | 0.733 | 0.711 | -0.022 |
| (545, 730] | 0.734 | 0.736 | +0.002 |
| (730, 1000] | 0.863 | 0.809 | -0.054 |
| (1000, 1500] | 0.899 | 0.839 | -0.060 |

The diagnostic tracks the true factor's rising shape (understated most when young,
closing the gap with age) to within 2-6 points at every bucket, on a population an order
of magnitude larger per bucket than the real book's. That gap is the selection bias named
above, not sampling noise -- it is consistent in sign at the oldest ages (diagnostic reads
low) and small enough not to reverse the qualitative conclusion.

Recommendation: adopt directionally, not numerically. On a sample this size, the
diagnostic is a credible qualitative signal that young open BI claims carry less than
their eventual cost, and the simulated check shows the method's own bias is a few points,
not enough to flip the sign. But the real book's own numbers (0.75-1.18, no visible trend)
are too noisy at these cohort sizes to extract an age-specific loading factor from this
snapshot alone -- the right use of this result is to support the existing freq-sev design
qualitatively, not to replace `ibner_open_factor = 1.0` with a fitted age curve from this
data.

## 3. Chain-ladder averaging windows and peril-group splits

### Averaging windows

Five windows compared against the frozen pipeline's own choice (`volume_all`, every
eligible loss month, volume-weighted): `simple_avg` (unweighted mean of ratios),
`latest1`/`latest2` (one or two most recent diagonals), `latest12mo`/`latest24mo` (every
diagonal inside the last 12 or 24 calendar months). Real book, cost rel. error pooled over
the 5 as-if dates (`item3_averaging_windows.png`):

| line | group | volume_all (current) | simple_avg | latest1 | latest2 | latest12mo | latest24mo |
|---|---|---|---|---|---|---|---|
| motor | bodily_injury | -8.0% | -5.6% | +16.5% | +2.5% | -5.6% | -7.6% |
| motor | other | -3.8% | -3.5% | -2.8% | -6.1% | -4.3% | -3.7% |
| buildings | weather | -15.2% | -13.9% | +4.6% | -3.4% | -13.1% | -15.0% |
| buildings | water | -11.2% | -10.8% | -16.0% | -10.3% | -10.6% | -11.6% |
| buildings | fire | +90.2% | +143.0% | -47.9% | +41.9% | +113.9% | +94.4% |
| buildings | ad | +14.9% | +19.0% | -7.0% | +5.1% | +10.6% | +14.3% |
| contents | all | +7.7% | +9.2% | +19.5% | -2.9% | +1.4% | +7.3% |

No window is consistently closer than the current one. `latest1`/`latest2` swing wildly
(buildings fire moves from +90% to -48% on one diagonal) because at these monthly
triangle granularities a single diagonal is a handful of claims; they are not a safe
default anywhere in this book. `simple_avg` and `latest12mo`/`latest24mo` track
`volume_all` closely everywhere except the already-noted thin books (buildings fire is
~449 claims a quarter at most, ad fewer still), where every variant disagrees with actual
by a similar large amount in the same direction -- a volume problem, not a window problem,
consistent with the base replay's own finding on buildings fire. The simulated books show
the same lack of a consistent winner (full table: `sim/item3_windows_sim.csv`).

### Peril-group splits

Splitting buildings weather into storm/flood/subsidence, and motor other into its four
constituent perils, each fit with its own chain ladder and summed back to the parent
group:

| line | group | pooled (current) rel. error | split-by-peril rel. error |
|---|---|---|---|
| buildings | weather | -15.2% | -13.0% |
| motor | other | -3.8% | -3.6% |

Simulated books, mean across 3 seeds: buildings weather -6.4% pooled vs -6.3% split; motor
other -2.7% vs -2.7%. The split is indistinguishable from the pool in both books -- the
sub-perils inside each group report at close enough to the same speed that separating
them adds parameter count without adding information, matching the base replay's own
observation that BF/Cape Cod priors add little once a chain ladder already has the
triangle mostly filled in.

Recommendation: no change on both counts. Keep `volume_all`; it is not worse than any
alternative tried and is the simplest to explain and audit. Do not split buildings weather
or motor other by peril; the extra granularity does not pay for itself on this evidence.

## 4. Matched-claim population vs all claims: delay CDFs, BI severity, ultimates

`target.match_claims` drops 410 of 161,439 non-repudiated claims as orphans (no policy
period contains the loss date) -- 0.08% of net cost (\$621,742 of \$744.95m), spread
across AY2021-2023 (48/47/44 claims respectively, none in 2024). The frozen pipeline's own
`developed_targets()` is inconsistent on this point: it fits `delay_cdfs` on **all**
claims, then divides **matched-claim** cost by that completion fraction -- this item
checks whether that inconsistency matters.

Refitting the delay CDF on matched-only claims moves completion fractions by at most 0.2
percentage points at any delay, in any group, at the real 31 Dec 2024 valuation date (full
table: `out/item4_completion_fractions.csv`). The BI severity base (AY2021 mean net,
feeding the freq-sev severity trend) moves by +0.04% (\$21,442 to \$21,451). The resulting
2021-2024 developed ultimates move by -0.2% at most (2024 cohorts, the least mature and
most exposed to a completion-fraction change), 0% in several already-mature cells
(`out/item4_ultimates_effect.csv`).

Recommendation: no change. At 0.08% of cost and spread fairly evenly across three
accident years, the orphan population is too small and too undifferentiated by age to move
either the delay pattern or the ultimates by a material amount. The inconsistency named
above is real but not costly here; it is worth a one-line fix in `developed_targets()` for
hygiene (fit on the same population it develops), not a reason to re-run anything.

## Files

- `scripts/replay_core.py`, `replay_lib.py`: unmodified copies of the base replay's
  reusable arithmetic, imported by every item script here.
- `scripts/10_item1_delay_mle.py` -> `out/item1_*.csv`, `sim/item1_*.csv`.
- `scripts/11_item2_bs_diagnostic.py` -> `out/item2_*.csv`, `sim/item2_*.csv`.
- `scripts/12_item3_windows_and_splits.py` -> `out/item3_*.csv`, `sim/item3_*.csv`.
- `scripts/13_item4_matched_vs_all.py` -> `out/item4_*.csv`.
- `item1_bi_delay_curve.png`, `item1_sim_ultimate_count_error.png`,
  `item2_case_adequacy_diagnostic.png`, `item3_averaging_windows.png`.
