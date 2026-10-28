# Office premium build-up, 20 emerging reference policies

Risk premium: `exp_o4_0c4e403_20261025_131648.csv` (expected ultimate claim cost for the row's period).
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
| M00161473 | motor | broker | 0.9644 | 1,203.63 | 0.00 | 85.00 | 394.48 | 525.97 | 210.39 | 210.39 | 2,629.86 | 45.8% | 1,131.78 |
| M00176777 | motor | broker | 0.9918 | 954.94 | 0.00 | 85.00 | 318.35 | 424.46 | 169.79 | 169.79 | 2,122.32 | 45.0% | 443.61 |
| M00185311 | motor | direct | 0.9644 | 798.69 | 0.00 | 85.00 | 192.11 | 0.00 | 102.46 | 102.46 | 1,280.71 | 62.4% | 872.31 |
| M00167496 | motor | broker | 0.9836 | 522.63 | 0.00 | 85.00 | 186.01 | 248.01 | 99.20 | 99.20 | 1,240.06 | 42.1% | 463.95 |
| M00129413 | motor | broker | 0.3315 | 127.67 | 0.00 | 85.00 | 65.10 | 86.81 | 34.72 | 34.72 | 434.03 | 29.4% | 354.15 |
| M00118569 | motor | direct | 0.9397 | 575.16 | 0.00 | 85.00 | 143.51 | 0.00 | 76.54 | 76.54 | 956.76 | 60.1% | 1,095.48 |
| M00153851 | motor | broker | 0.9616 | 276.11 | 0.00 | 85.00 | 110.54 | 147.39 | 58.96 | 58.96 | 736.96 | 37.5% | 414.72 |
| B00229821 | buildings | broker | 0.9918 | 315.87 | 37.90 | 85.00 | 134.32 | 179.09 | 71.64 | 71.64 | 895.45 | 35.3% | 285.60 |
| B00218282 | buildings | broker | 0.9726 | 320.99 | 38.52 | 85.00 | 136.07 | 181.43 | 72.57 | 72.57 | 907.16 | 35.4% | 230.80 |
| B00250679 | buildings | broker | 0.1096 | 42.44 | 5.09 | 85.00 | 40.57 | 54.10 | 21.64 | 21.64 | 270.48 | 15.7% | 180.72 |
| B00223369 | buildings | broker | 0.9945 | 544.67 | 65.36 | 85.00 | 212.77 | 283.69 | 113.48 | 113.48 | 1,418.44 | 38.4% | 204.56 |
| B00215809 | buildings | direct | 0.9753 | 449.67 | 53.96 | 85.00 | 127.96 | 0.00 | 68.25 | 68.25 | 853.09 | 52.7% | 237.22 |
| B00215707 | buildings | broker | 0.9863 | 303.60 | 36.43 | 85.00 | 130.11 | 173.48 | 69.39 | 69.39 | 867.41 | 35.0% | 272.44 |
| B00217584 | buildings | broker | 0.9671 | 866.82 | 104.02 | 85.00 | 323.22 | 430.95 | 172.38 | 172.38 | 2,154.77 | 40.2% | 323.36 |
| C00259568 | contents | broker | 1.0 | 46.16 | 5.54 | 85.00 | 41.85 | 55.79 | 22.32 | 22.32 | 278.97 | 16.5% | 83.18 |
| C00283470 | contents | broker | 0.9726 | 68.05 | 8.17 | 85.00 | 49.35 | 65.80 | 26.32 | 26.32 | 329.01 | 20.7% | 117.97 |
| C00270331 | contents | broker | 0.9726 | 38.60 | 4.63 | 85.00 | 39.26 | 52.34 | 20.94 | 20.94 | 261.71 | 14.8% | 140.29 |
| C00314425 | contents | broker | 0.0685 | 5.41 | 0.65 | 85.00 | 27.88 | 37.17 | 14.87 | 14.87 | 185.84 | 2.9% | 102.87 |
| C00302985 | contents | broker | 0.9918 | 47.77 | 5.73 | 85.00 | 42.40 | 56.53 | 22.61 | 22.61 | 282.66 | 16.9% | 115.88 |
| C00286287 | contents | broker | 0.9671 | 118.40 | 14.21 | 85.00 | 66.62 | 88.82 | 35.53 | 35.53 | 444.11 | 26.7% | 269.23 |
