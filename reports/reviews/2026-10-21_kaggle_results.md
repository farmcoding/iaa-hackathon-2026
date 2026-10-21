# Kaggle results collected 21 October

Two GPU/CPU jobs from 20 October finished overnight. Neither changes the champion (run 20261020T161441, primary 0.14252, fold A 0.14234, fold B 0.14261).

## 4.2 RealMLP as a blend member

| | Fold A | Fold B |
|---|---|---|
| RealMLP alone, pooled | 0.0625 | 0.0986 |
| Motor | -0.0533 | 0.0179 |
| Buildings | 0.0819 | 0.0892 |
| Contents | 0.0610 | 0.0675 |

The pytabkit RealMLP regressor does not accept sample weights, so exposure could not weight the loss, and motor in particular fits poorly. A no-refit blend sweep against the champion's saved validation predictions (weights 0 to 0.5, re-anchored to the champion's line totals) gives no gain at any weight: fold B is flat at 0.1 (0.14260 against 0.14261) and falls after that; fold A falls from the first step. Not pursued. Together with TabM (3.31), this closes the neural-member line for Part 1.

## 3.25 Variable study v2 (partial)

The job reached Kaggle's 12-hour limit after fold A. It produced permutation importance for 3 seeds on fold A only. Fold B, the family drops and the stability matrix were not reached. The model was the champion as of midday 20 October, before customer history was dropped from buildings.

- Top features match the grouped K-fold study (3.36): flood zone, sum insured and elevation on property; no-claims years, annual mileage and driver age on motor.
- Buildings `hist_n_customer` has a negative mean importance (-0.00016), which agrees with the decision to drop it (3.45).
- Motor `cover_type`, `payment_method` and `body_type` show negative importance on fold A with seed spread up to 0.0035. That is one fold, so it goes to Part 2 as a prune candidate alongside the 3.36 list rather than into the Part 1 model.

Source files stay in the owner cache (`kaggle_out/realmlp`, `kaggle_out/varstudy`); competition outputs are not committed.
