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
# Owner decision 2026-10-18 (Karl): monotone constraints adopted for defensible,
# portable relativities even though the score effect is inside seed noise. Adding
# missed_payments_12m and sum_insured cost 0.0008 (worse fold -0.0026), so they are left
# unconstrained.
# +1 = cost may only rise with the feature, -1 = only fall. See assumptions_log.md.
MONOTONE = {
    "voluntary_excess": -1,
    "compulsory_excess": -1,
    "ncb_years": -1,
    "licence_points": 1,
    "flood_zone": 1,
    "prior_claims_5y": 1,
}
SEEDS = (0, 1, 2)
PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 1200,
    "verbose": -1,
    "seed": 0,
    # reproducibility (owner, 2026-10-18): identical results on rerun
    "deterministic": True,
    "force_row_wise": True,
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
            mono = [MONOTONE.get(c, 0) for c in cols]
            model = lgb.LGBMRegressor(
                **{**PARAMS, "seed": seed},
                monotone_constraints=mono,
                monotone_constraints_method="advanced",
            )
            # rate target with exposure weight
            model.fit(tr[cols], tr["y"] / tr["exposure"], sample_weight=tr["exposure"])
            rate += model.predict(va[cols]) / len(SEEDS)
        # level anchoring: the model supplies relativities; the line's overall loss cost
        # is the on-levelled historical loss cost (trend assumption in the config)
        target_lc = tr["y"].sum() / tr["exposure"].sum()
        rate *= target_lc / np.average(rate, weights=va["exposure"])
        pred[mask] = rate * va["exposure"].to_numpy()
    return pred
