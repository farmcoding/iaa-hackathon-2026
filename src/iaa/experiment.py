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


def _anchor(rate: np.ndarray, target_lc: float, va: pd.DataFrame) -> np.ndarray:
    # level anchoring: the model supplies relativities; the overall loss cost is the
    # on-levelled historical loss cost (trend assumption in the config)
    return rate * target_lc / np.average(rate, weights=va["exposure"])


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(valid))
    for line in ("motor", "buildings", "contents"):
        tr = train[train["line"] == line]
        mask = (valid["line"] == line).to_numpy()
        va = valid[mask]
        cols = _feature_cols(tr)
        mono = [MONOTONE.get(c, 0) for c in cols]

        # seed bagging: single-seed noise (sd 0.0026 on the primary metric) is larger
        # than most real improvements, so every model is an average over SEEDS
        def bag(target: pd.Series, weight: pd.Series, params: dict = PARAMS) -> np.ndarray:
            out = np.zeros(len(va))
            for seed in SEEDS:
                model = lgb.LGBMRegressor(
                    **{**params, "seed": seed},
                    monotone_constraints=mono,
                    monotone_constraints_method="advanced",
                )
                model.fit(tr[cols], target, sample_weight=weight)
                out += model.predict(va[cols]) / len(SEEDS)
            return out

        if line == "motor":
            # backlog 1.1: one 11.9m bodily-injury claim sits in every split of a cost
            # Tweedie on the combined target. BI claim frequency is far more
            # predictable than its severity, so BI relativities come from a Poisson
            # count model; severity is a flat level shared by every policy, supplied
            # entirely by anchoring to the developed BI loss cost. Other motor perils
            # keep the original Tweedie-on-cost treatment.
            bi_params = {k: v for k, v in PARAMS.items() if k != "tweedie_variance_power"}
            bi_params["objective"] = "poisson"
            freq = bag(tr["n_bodily_injury"], tr["exposure"], bi_params)
            bi_lc = (tr["ult_bodily_injury"] * tr["trend_factor"]).sum() / tr["exposure"].sum()
            bi_rate = _anchor(freq, bi_lc, va)

            other = tr["ult_other"] * tr["trend_factor"]
            other_rate = bag(other / tr["exposure"], tr["exposure"])
            other_lc = other.sum() / tr["exposure"].sum()
            other_rate = _anchor(other_rate, other_lc, va)

            rate = bi_rate + other_rate
        else:
            rate = bag(tr["y"] / tr["exposure"], tr["exposure"])
            rate = _anchor(rate, tr["y"].sum() / tr["exposure"].sum(), va)
        pred[mask] = rate * va["exposure"].to_numpy()
    return pred
