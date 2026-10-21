# Policy-grouped K-fold stability check (backlog 3.36)

Five folds grouped by policy_id over all four training years (2021-2024 pooled, on-levelled with the harness trend logic), run on Kaggle CPU with the champion code as of 13:34 on 20 October (before the motor constraints). Every row is scored once, by a model that never saw its policy. Diagnostic only: it says how stable the relativities are, not how well the model forecasts a later year, so it is never a keep gate. Total run time 7.1 hours on Kaggle.

## Accuracy across folds

| Line | Mean D2 | Std across folds | Range |
|---|---|---|---|
| Motor | 0.0846 | 0.0057 | 0.079 to 0.093 |
| Buildings | 0.1014 | 0.0037 | 0.097 to 0.106 |
| Contents | 0.0962 | 0.0076 | 0.087 to 0.104 |
| Pooled | 0.1392 | 0.0025 | 0.135 to 0.142 |

The spread between folds (pooled sd 0.0025) is about ten times the seed noise of the out-of-time folds, so differences below a few thousandths between models cannot be read from any single random split. Actual/expected in these folds is not meaningful: predictions are on-levelled to 2025 while the pooled actuals span 2021-2024.

## Which predictors are stable

Mean pairwise Spearman correlation of permutation-importance ranks across the ten fold pairs: motor 0.67, buildings 0.69, contents 0.74. The top predictors are stable on every line (the top ten have positive importance in every fold, nine of ten on motor):

- Motor: no-claims years, annual mileage, driver age, years licensed, voluntary excess, prior claims, region, cover type.
- Buildings: flood zone, sum insured, elevation, prior claims, distance to river, energy rating, year built, soil type.
- Contents: sum insured, flood zone, elevation, voluntary excess, prior claims, urban density, accidental damage cover, region.

Predictors with negative importance in at least three of five folds (they add noise, not signal):

- Motor: dry summer index, mean temperature, seats, distance to river, freezing days, engine size, make and model.
- Buildings: listed building, renewal count, customer claim history (`hist_n_customer`).
- Contents: bicycle cover, solar panels, lodgers.

Two of these confirm decisions taken the same afternoon on other evidence: customer claim history on buildings (dropped under 3.45) and distance to river on motor (dropped under constraints option 2b). The motor weather fields match the earlier finding that weather adds little to motor; removing them on the out-of-time folds was neutral (backlog 2.87). The rest are candidates for Part 2 pruning, to be confirmed on the out-of-time folds.
