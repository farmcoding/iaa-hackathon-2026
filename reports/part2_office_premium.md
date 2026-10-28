# Office premium build-up, 20 emerging reference policies

Risk premium: `o4_mech_lvl_20261028.csv` (expected ultimate claim cost for the row's period).
Emerging expense assumptions (`expenses.parquet`):

| item | developed | emerging | unit |
|---|---|---|---|
| fixed_expense_per_policy | 45.0 | 85.0 | CU |
| variable_expense | 0.12 | 0.15 | share of premium |
| commission_direct | 0.0 | 0.0 | share of premium |
| commission_comparison_site | 0.1 |  | share of premium |
| commission_broker | 0.15 | 0.2 | share of premium |
| premium_tax | 0.12 | 0.08 | share of premium |
| cat_reinsurance | 0.06 | 0.12 | share of property risk premium |
| target_profit_margin | 0.05 | 0.08 | share of premium |

Office premium = (risk premium + fixed expense + cat reinsurance) / (1 - variable expense -
commission - premium tax - profit margin). Cat reinsurance is 12% of the risk premium on
buildings and contents only. Share-of-premium items are shares of the office premium, so each
row adds back exactly. The fixed expense is charged in full per policy, so short-period
policies (exposure 0.07 and 0.11 here) carry it in full; pro-rating it by exposure is the
alternative if the expense is incurred over the policy year. `incumbent_premium` is the
broker book's existing written premium, for comparison only (not a model input).

| row_id | line | channel | exposure | risk_premium | cat_reinsurance | fixed_expense | variable_expense | commission | premium_tax | profit | office_premium | loss_ratio | incumbent_premium |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M00161473 | motor | broker | 0.9644 | 1,446.02 | 0.00 | 85.00 | 468.68 | 624.90 | 249.96 | 249.96 | 3,124.52 | 46.3% | 1,131.78 |
| M00176777 | motor | broker | 0.9918 | 1,128.34 | 0.00 | 85.00 | 371.43 | 495.24 | 198.10 | 198.10 | 2,476.20 | 45.6% | 443.61 |
| M00185311 | motor | direct | 0.9644 | 969.51 | 0.00 | 85.00 | 229.24 | 0.00 | 122.26 | 122.26 | 1,528.28 | 63.4% | 872.31 |
| M00167496 | motor | broker | 0.9836 | 624.19 | 0.00 | 85.00 | 217.10 | 289.47 | 115.79 | 115.79 | 1,447.33 | 43.1% | 463.95 |
| M00129413 | motor | broker | 0.3315 | 148.36 | 0.00 | 85.00 | 71.44 | 95.25 | 38.10 | 38.10 | 476.24 | 31.2% | 354.15 |
| M00118569 | motor | direct | 0.9397 | 625.98 | 0.00 | 85.00 | 154.56 | 0.00 | 82.43 | 82.43 | 1,030.41 | 60.8% | 1,095.48 |
| M00153851 | motor | broker | 0.9616 | 323.09 | 0.00 | 85.00 | 124.93 | 166.57 | 66.63 | 66.63 | 832.83 | 38.8% | 414.72 |
| B00229821 | buildings | broker | 0.9918 | 315.87 | 37.90 | 85.00 | 134.32 | 179.09 | 71.64 | 71.64 | 895.45 | 35.3% | 285.60 |
| B00218282 | buildings | broker | 0.9726 | 320.99 | 38.52 | 85.00 | 136.07 | 181.43 | 72.57 | 72.57 | 907.16 | 35.4% | 230.80 |
| B00250679 | buildings | broker | 0.1096 | 42.44 | 5.09 | 85.00 | 40.57 | 54.10 | 21.64 | 21.64 | 270.48 | 15.7% | 180.72 |
| B00223369 | buildings | broker | 0.9945 | 544.67 | 65.36 | 85.00 | 212.77 | 283.69 | 113.48 | 113.48 | 1,418.44 | 38.4% | 204.56 |
| B00215809 | buildings | direct | 0.9753 | 449.67 | 53.96 | 85.00 | 127.96 | 0.00 | 68.25 | 68.25 | 853.09 | 52.7% | 237.22 |
| B00215707 | buildings | broker | 0.9863 | 303.60 | 36.43 | 85.00 | 130.11 | 173.48 | 69.39 | 69.39 | 867.41 | 35.0% | 272.44 |
| B00217584 | buildings | broker | 0.9671 | 866.82 | 104.02 | 85.00 | 323.22 | 430.95 | 172.38 | 172.38 | 2,154.77 | 40.2% | 323.36 |
| C00259568 | contents | broker | 1.0 | 42.93 | 5.15 | 85.00 | 40.74 | 54.32 | 21.73 | 21.73 | 271.59 | 15.8% | 83.18 |
| C00283470 | contents | broker | 0.9726 | 63.28 | 7.59 | 85.00 | 47.72 | 63.62 | 25.45 | 25.45 | 318.10 | 19.9% | 117.97 |
| C00270331 | contents | broker | 0.9726 | 35.89 | 4.31 | 85.00 | 38.33 | 51.10 | 20.44 | 20.44 | 255.51 | 14.0% | 140.29 |
| C00314425 | contents | broker | 0.0685 | 5.03 | 0.60 | 85.00 | 27.75 | 36.99 | 14.80 | 14.80 | 184.97 | 2.7% | 102.87 |
| C00302985 | contents | broker | 0.9918 | 44.42 | 5.33 | 85.00 | 41.25 | 55.00 | 22.00 | 22.00 | 275.00 | 16.2% | 115.88 |
| C00286287 | contents | broker | 0.9671 | 110.04 | 13.21 | 85.00 | 63.75 | 85.00 | 34.00 | 34.00 | 425.00 | 25.9% | 269.23 |
