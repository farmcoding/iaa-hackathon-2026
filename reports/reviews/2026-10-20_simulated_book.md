# Simulated claims book: calibration report

This simulator reproduces the real 2021-2024 claims book of ~/Projects/iaa-hackathon-2026 at an aggregate level (monthly exposure, frequency trend, report delay, severity, and settlement by line and peril group), so a reserving method can be replayed against a book with known true ultimates. Three seeds (1, 2, 3) give three independent draws from the same calibration, so a method ranking can be checked for stability rather than trusted on one draw.

## Frequency: ultimate rate per unit exposure, by year

Real rates are reported counts grossed to ultimate using the count chain ladder (`ultimate.delay_cdfs(..., value="count")`), since reported counts for 2024 are still incomplete. Simulated rates use the true (fully reported) count by loss year, divided by the same exposure base, averaged across the three seeds.

| line.group | year | real ultimate rate | sim true rate (avg of 3 seeds) |
|---|---|---|---|
| motor.bodily_injury | 2021 | 0.01144 | 0.01130 |
| motor.bodily_injury | 2022 | 0.01105 | 0.01125 |
| motor.bodily_injury | 2023 | 0.01094 | 0.01099 |
| motor.bodily_injury | 2024 | 0.01047 | 0.01073 |
| motor.other | 2021 | 0.13428 | 0.13643 |
| motor.other | 2022 | 0.13790 | 0.13474 |
| motor.other | 2023 | 0.13026 | 0.13311 |
| motor.other | 2024 | 0.12879 | 0.13041 |
| buildings.weather | 2021 | 0.01855 | 0.01805 |
| buildings.weather | 2022 | 0.01953 | 0.01956 |
| buildings.weather | 2023 | 0.02051 | 0.02041 |
| buildings.weather | 2024 | 0.01951 | 0.02149 |
| buildings.water | 2021 | 0.01762 | 0.01756 |
| buildings.water | 2022 | 0.01714 | 0.01681 |
| buildings.water | 2023 | 0.01679 | 0.01682 |
| buildings.water | 2024 | 0.01792 | 0.01632 |
| buildings.fire | 2021 | 0.00168 | 0.00177 |
| buildings.fire | 2022 | 0.00194 | 0.00192 |
| buildings.fire | 2023 | 0.00186 | 0.00188 |
| buildings.fire | 2024 | 0.00210 | 0.00204 |
| buildings.ad | 2021 | 0.00623 | 0.00644 |
| buildings.ad | 2022 | 0.00674 | 0.00680 |
| buildings.ad | 2023 | 0.00646 | 0.00666 |
| buildings.ad | 2024 | 0.00682 | 0.00688 |
| contents.all | 2021 | 0.03965 | 0.04007 |
| contents.all | 2022 | 0.04022 | 0.04003 |
| contents.all | 2023 | 0.04118 | 0.04156 |
| contents.all | 2024 | 0.04131 | 0.04157 |

The 2024 row is the least reliable check: the trend used to simulate it is a log-linear fit to 2021-2023 only, extrapolated one year forward, so a group with a noisy three-point fit (buildings weather, buildings water) can show a few points of gap in 2024 that is not present in 2021-2023.

## Report-delay CDF at 3 / 6 / 12 / 24 months (mature cohorts only)

Real-data CDFs use only loss months old enough to have observed that delay in full (motor bodily injury: losses through 2021; all other groups: losses through 2022), so AY2024 right-truncation does not bias the comparison. Simulated CDFs are computed on reported claims from the same mature loss months, pooled across the three seeds.

| line.group | months | real CDF | sim CDF |
|---|---|---|---|
| motor.bodily_injury | 3 | 0.384 | 0.383 |
| motor.bodily_injury | 6 | 0.625 | 0.620 |
| motor.bodily_injury | 12 | 0.858 | 0.863 |
| motor.bodily_injury | 24 | 0.986 | 0.986 |
| motor.other | 3 | 0.991 | 0.977 |
| motor.other | 6 | 0.995 | 0.995 |
| motor.other | 12 | 0.995 | 1.000 |
| motor.other | 24 | 1.000 | 1.000 |
| buildings.weather | 3 | 0.938 | 0.939 |
| buildings.weather | 6 | 0.969 | 0.967 |
| buildings.weather | 12 | 0.990 | 0.991 |
| buildings.weather | 24 | 1.000 | 1.000 |
| buildings.water | 3 | 0.992 | 0.974 |
| buildings.water | 6 | 0.994 | 0.993 |
| buildings.water | 12 | 0.994 | 1.000 |
| buildings.water | 24 | 1.000 | 1.000 |
| buildings.fire | 3 | 0.996 | 0.991 |
| buildings.fire | 6 | 0.996 | 0.999 |
| buildings.fire | 12 | 0.996 | 1.000 |
| buildings.fire | 24 | 1.000 | 1.000 |
| buildings.ad | 3 | 0.994 | 0.981 |
| buildings.ad | 6 | 0.994 | 0.997 |
| buildings.ad | 12 | 0.994 | 1.000 |
| buildings.ad | 24 | 1.000 | 1.000 |
| contents.all | 3 | 0.994 | 0.980 |
| contents.all | 6 | 0.994 | 0.995 |
| contents.all | 12 | 0.994 | 1.000 |
| contents.all | 24 | 1.000 | 1.000 |

## Mean severity, closed claims (net cost)

The fit target is the mature-cohort column (AY2021-2022 closed claims), which is what the lognormal body and Pareto tail were calibrated against. The all-years column is shown for context: it mixes in younger claims that have not finished settling, so it understates true severity in the real data by construction (small claims close fastest), while the simulated all-years mean is unbiased because closure age is independent of claim size.

| line.group | real mean (2021-22 cohort) | sim mean (2021-22 cohort, avg of 3 seeds) | real mean (all years) | sim mean (all years, avg of 3 seeds) |
|---|---|---|---|---|
| motor.bodily_injury | 22,126 | 20,904 | 20,901 | 21,130 |
| motor.other | 2,431 | 2,097 | 2,539 | 2,088 |
| buildings.weather | 10,504 | 9,261 | 10,404 | 9,436 |
| buildings.water | 5,236 | 4,595 | 5,637 | 4,580 |
| buildings.fire | 52,600 | 44,040 | 53,259 | 45,668 |
| buildings.ad | 1,207 | 1,053 | 1,208 | 1,063 |
| contents.all | 2,663 | 2,440 | 2,724 | 2,444 |

## Share of claims open at 31 Dec 2024

| line.group | real | sim (avg of 3 seeds) |
|---|---|---|
| motor.bodily_injury | 0.381 | 0.380 |
| motor.other | 0.044 | 0.038 |
| buildings.weather | 0.086 | 0.108 |
| buildings.water | 0.068 | 0.053 |
| buildings.fire | 0.134 | 0.152 |
| buildings.ad | 0.019 | 0.013 |
| contents.all | 0.034 | 0.025 |

Groups with longer real report delays (buildings weather, buildings water, contents) show a few points of drift here, because the simulator clips a drawn closure age so it cannot fall before the claim's report age, and closure dates are rounded to a 30.44-day month. Both push simulated closure slightly later for claims reported late in their loss month, which raises the simulated open share a little versus real data for those groups, and lowers it for the two groups (buildings water, contents) where the opposite rounding direction dominates. Motor bodily injury, where the open share matters most for the replay, matches to three decimal places.

## Motor bodily injury: open-claim average net by age at valuation

Real averages exclude one AY2023 claim with net above 11.9m, which alone would dominate its age bucket. Expected pattern: younger open claims carry lower average incurred than mature ones, because case reserves understate the eventual cost early in the life of a bodily-injury claim.

| age bucket (days) | real mean open net | sim mean open net (avg of 3 seeds) |
|---|---|---|
| (0, 180] | 15,439 | 15,329 |
| (180, 365] | 16,701 | 17,435 |
| (365, 545] | 16,227 | 15,927 |
| (545, 730] | 16,229 | 15,306 |
| (730, 1000] | 19,102 | 17,505 |
| (1000, 1500] | 19,893 | 17,179 |

## Limitations

- The report-delay tail beyond the mature observation window (losses through 2021 for motor bodily injury, giving at least 36 months of development; losses through 2022, at least 24 months, for every other group) is an extrapolated exponential. Its shape past 48 months is a modelling assumption.
- Severity is independent of report delay in this simulator. The real pipeline's own documentation says late-reported claims tend to run larger, which is one reason it prefers a cost chain ladder. A reserving replay run on this simulated book will not show that same correlation, so a method that wins on the real book partly for that reason may rank differently here.
- Claim closure age is drawn independent of claim size for every group, including motor bodily injury. In reality, large claims likely take longer to settle; this simulator does not reproduce that link.
- Severity trends reuse the project's own config assumptions (configs/market_dev.yaml): motor bodily injury 3.5% a year (as instructed), motor other and contents from the pooled line trend, and buildings per peril group (weather/water 9%, fire/ad 4%). These are themselves fitted estimates from the same real book. A replay that recovers the config's own trend confirms internal consistency, but it is a weaker check than validating the trend-selection step against an independent source.
- The peril mix within each peril group (e.g. storm/flood/subsidence inside buildings.weather) is held fixed over time at its 2021-2024 average share; any real drift in that mix is not reproduced.
- Repudiated claims (about 1% of the real book) are omitted entirely from the simulated book rather than simulated with a repudiation process.
- Case-estimate noise and the bodily-injury understatement factor are simple multiplicative lognormal/age-bucket models, fit to a small number of open claims (2,885 real motor bodily injury claims open at valuation, spread across 7 age buckets); the per-bucket factors carry real sampling noise.
- The simulated mean severity sits 10-16% below the real mature-cohort mean in most groups (see the severity table). The lognormal body is truncated at the fitted threshold (the 97th percentile of mature-cohort net) so the Pareto tail above it does not double-count mass, and the resulting mean depends on the fitted tail share and Pareto alpha rather than being forced to match the real mean exactly; a tighter match would need re-tuning the tail share per group.
