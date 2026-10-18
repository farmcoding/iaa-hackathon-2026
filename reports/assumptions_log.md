# Assumptions log (CIA SOP 1600)

Changes to these need a human decision, recorded here with the date and reason. The
values live in `configs/market_dev.yaml`.

| Date | Assumption | Value | Basis | Sensitivity to run |
|---|---|---|---|---|
| 2026-10-18 | Development method | Chain ladder on reported net cost by loss month and report-delay month, per line and peril group | Single valuation date; report dates give the delay pattern | Peril grouping |
| 2026-10-18 | Case reserve adequacy (IBNER) | Open-claim incurred taken as unbiased, except motor bodily injury | Only one valuation, so IBNER can't be estimated directly | +/-10% on open BI |
| 2026-10-18 | Motor bodily injury | Ultimate count (count chain ladder) x 2021 mean severity trended at 3.5% a year | 2021 is the most mature cohort; see data-quality finding 3 | Trend 1.5% to 5.5%; base year 2022 |
| 2026-10-18 | Loss-cost trend to the prediction year | Motor 3.0%, buildings 6.5%, contents 5.0% a year | Log-linear fits to developed loss cost: motor 2.9% (2021-24) and 3.6% (2021-23); buildings 8.0% and 6.3% (2024 weather-elevated); contents 5.1% and 5.4% | +/-1.5% |
| 2026-10-18 | Overall level of predictions | Line loss cost = on-levelled historical loss cost; the model supplies relativities | Model alone predicted about 9% low out of time (feature drift) | Anchor on 2023-24 only |
| 2026-10-18 | Validation scorer | Tweedie D2, power 1.5, constant-mean baseline (sklearn definition) | Calibration submission: public 0.0305 vs local 0.0395 (constant) and -0.0004 (rate) | n/a |
| 2026-10-18 | Keep rule noise floor | 0.002 on mean fold D2 | Single-seed sd 0.0026; experiments bag 3 seeds | n/a |
