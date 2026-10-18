"""THE EXPERIMENT FILE. This is the only file the experiment agent may edit.

Contract (enforced by harness.py):
    fit_predict(train, valid) -> np.ndarray of expected ultimate net claim cost per
    valid row, finite and >= 0, in the order of valid.

train carries the developed, on-levelled target `y` (plus per-group developed costs
ult_<group>, reported counts n_<peril>, trend_factor). valid carries features only.
Both frames hold all three lines; `line` says which.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATURES = {
    "row_id",
    "policy_id",
    "customer_id",
    "address_id",
    "period_start",
    "period_end",
    "split",
    "y",
    "y_ult",
    "trend_factor",
    "year",
    "line",
    "exposure",
}
SEEDS = (0, 1, 2)
PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.3,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 600,
    "verbose": -1,
    "seed": 0,
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {c for c in df.columns if c.startswith(("cost_", "n_", "ult_", "comp_"))}
    cols = [c for c in df.columns if c not in NON_FEATURES and c not in drop]
    return [c for c in cols if df[c].notna().any()]


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(valid))
    for line in ("motor", "buildings", "contents"):
        tr = train[train["line"] == line]
        mask = (valid["line"] == line).to_numpy()
        va = valid[mask]
        cols = _feature_cols(tr)
        # seed bagging: single-seed noise (sd 0.0026 on the primary metric) is larger
        # than most real improvements, so every model is an average over SEEDS
        rate = np.zeros(len(va))
        for seed in SEEDS:
            model = lgb.LGBMRegressor(**{**PARAMS, "seed": seed})
            # rate target with exposure weight
            model.fit(tr[cols], tr["y"] / tr["exposure"], sample_weight=tr["exposure"])
            rate += model.predict(va[cols]) / len(SEEDS)
        # level anchoring: the model supplies relativities; the line's overall loss cost
        # is the on-levelled historical loss cost (trend assumption in the config)
        target_lc = tr["y"].sum() / tr["exposure"].sum()
        rate *= target_lc / np.average(rate, weights=va["exposure"])
        pred[mask] = rate * va["exposure"].to_numpy()
    return pred
