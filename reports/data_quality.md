# Data-quality log (CIA SOP 1440)

Each finding: what, how found, size, treatment, where it is enforced.

| # | Finding | How found | Size | Treatment | Enforced in |
|---|---|---|---|---|---|
| 1 | Exposure recorded in days instead of years (values 365, 89, 292...) and a cluster at 1.01 | AI profiling: train-2024 mean exposure above 1.0 for one-year rows; confirmed by comparing exposure with period length | about 0.4% (days) and 0.3% (1.01) of training rows; test files clean | Exposure recomputed as (end - start + 1) / 365 for every row | `data.py`; `tests/test_core.py` |
| 2 | `customer_service_contacts` includes claim-related contacts in training only | Feature importance (72% of gain), then distribution check | Train rows with a claim 3.5 contacts per exposure-year vs 0.5 without; test 0.50 | Excluded as a feature (target leak) | `features.FORBIDDEN`; test |
| 3 | Motor bodily injury case estimates understate final cost | Same report-delay cohorts show falling severity in younger accident years while other perils inflate; open shares 76-92% in 2024 | 2024 BI developed by reported-cost chain ladder came out 24% below earlier years | Frequency x severity projection for BI (count chain ladder x trended 2021 severity) | `ultimate.py`; config `freq_sev` |
| 4 | Claims that match no policy period | Join integrity check | 410 of 161,439 non-repudiated claims: 271 dated 2020 (before any policy row), 47, 47 and 44 in 2021-2023 | Excluded; counted every build | `target.py` stats; test (< 0.5%) |
| 5 | Missing values spread evenly across unrelated columns | Profiling | about 2.5% in several columns (e.g. vehicle_age, smoke_alarms, security_level); same pattern in test | Left as missing; GBM handles missing natively; any GLM imputes the same way in train and test | `experiment.py` |
| 6 | `renewal_decision` exists only in training files and describes the outcome after the period | Schema comparison | all training rows | Excluded as a feature | `features.FORBIDDEN`; test |
| 7 | 2024 rows end on 30 December (365-day year in a leap year) | Period profiling | all 2024 rows | No effect: no claims are dated 2024-12-31 | documented only |
| 8 | `customer_since_year` drifts between train and test by construction | Train vs test shift scan | +0.56 sd | Replaced by tenure (`year - customer_since_year`) | `features.py` |

Open item: why 2021-2023 orphan claims fall outside every policy period (cancelled or
lapsed rows, or date errors) is not yet explained.

## Pipeline issues (code, not data)

| # | Issue | How found | Effect | Treatment | Status |
|---|---|---|---|---|---|
| P1 | `n_lines_held` shared the `n_` prefix of the claim-count columns, so the experiment filter and the harness's `hide_target` both removed it | Reading the feature filter while preparing TabM | The feature was never used by any model before 2026-10-18 | Renamed `lines_held` in `features.py`; tested (no gain) and excluded from LightGBM; TabM uses it | fixed |
| P2 | The credibility encoder's cross-fitting folds were unseeded | The best model did not reproduce after a dataset rebuild | Same model scored 0.1332 to 0.1339 across four runs | `random_state=0`; best model re-baselined (0.1330) | fixed |
| P3 | On the run that rebuilds the dataset cache, `harness.dataset()` returns the in-memory frame, whose categorical columns differ from the parquet round-trip; the target encoder then crashes | First harness run after the P1 rename | Loud crash, never a wrong score; every later run reads the cache and works | Fixed 2026-10-18 (owner-approved): `dataset()` returns `pd.read_parquet(path)` after writing. The fingerprint now covers only data-building code and config, so harness changes no longer orphan the TabM files; rebuilt dataset verified identical | fixed |
