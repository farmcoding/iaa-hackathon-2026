"""THE PART 2 EXPERIMENT FILE (emerging market). The only file the Part 2 experiment
agent may edit. Selected with IAA_MARKET=emerging and --module iaa.experiment_emerging.

Contract (enforced by harness.py):
    fit_predict(train, valid) -> np.ndarray of expected ultimate net claim cost per
    valid row, finite and >= 0, in the order of valid.

train carries the developed, on-levelled target `y` (plus per-group developed costs
ult_<group>, reported counts n_<peril>, trend_factor). valid carries features only.
Both frames hold all three lines; `line` says which. Both carry `base`: the Part 1 model,
fitted on the developed market without the validation year, predicting this row (CU).
The harness attaches it and sets each line's level afterwards (balance on the training
rows), so only relativities matter here; valid may include the training rows.

v0 (owner, 2026-10-25): the transfer test showed the developed relativities beat a
native emerging refit (level-neutral D2 on 2024: 0.151 vs 0.123), so the emerging data
corrects the developed model instead of replacing it: LightGBM per line, Tweedie, with
log(base) as the starting score (an offset). Few, shallow, heavily regularised trees:
two emerging years cannot support many corrections.
"""

from __future__ import annotations

import os

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
    "base",
}
# owner decisions carried over from Part 1 (admissibility, not score; reports/assumptions_log.md):
# no premium, email domain, sales/admin fields or customer-level claim history on any line;
# no licence points on property. gender: owner decision pending (Part 2 plan, decision 1),
# excluded until then; the agent may not add it.
INADMISSIBLE = {
    "written_premium",
    "email_domain",
    "campaign_code",
    "marketing_segment",
    "branch_id",
    "days_quote_to_inception",
    "marketing_consent",
    "hist_n_customer",
    "hist_cost_customer",
    "gender",
}
LINE_DROP = {"buildings": {"licence_points"}, "contents": {"licence_points"}}
# banded in this market: the band is the information; the mapped mean age (features.py)
# is for the transfer base only
DUPLICATES = {"customer_age", "driver_age", "policyholder_age"}
MONOTONE = {
    "voluntary_excess": -1,
    "compulsory_excess": -1,
    "ncb_years": -1,
    "licence_points": 1,
    "flood_zone": 1,
    "prior_claims_5y": 1,
}
SEED_OFFSET = int(os.environ.get("IAA_SEED_OFFSET", "0"))
SEEDS = tuple(SEED_OFFSET + s for s in (0, 1, 2))
PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "min_data_per_group": 200,
    "n_estimators": 300,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}
# False scores the developed base alone. Owner bar, 2026-10-25: every free-form boosted
# correction tried on one emerging year (25-300 trees, leaves of 300-3000 rows, with and
# without the high-cardinality categoricals) lowered level-neutral D2 on both folds;
# previous_insurer and occupation took most of the split gain (overfitting ~350 claims a
# year on contents). Corrections need structure: few factors, strong shrinkage.
USE_CORRECTION = False


def credibility_factors(
    tr: pd.DataFrame, col: str, k: float = 200.0, basis: str = "cost"
) -> pd.Series:
    """Multiplicative correction per level of `col` for one line's training rows.

    A/E of each level is taken relative to the line's own A/E (the base carries the
    developed level, so raw y / base mixes level and relativity), then shrunk towards 1 by
    z = n / (n + k), n = claims in that level. Missing is its own level.

    basis "cost": actual developed cost against the base. One year of cost is dominated by
    a few large claims (bodily injury, fire), so cost factors carry severity noise.
    basis "count": the level's share of reported claims (n_total) against its share of
    the base prediction; assumes the base's severity relativities hold and corrects
    frequency only. Far less noisy on a small book.
    Use as: apply_factors(valid, col, credibility_factors(train_line, col)).
    """
    key = tr[col].astype("string").fillna("missing")
    num = tr["n_total"] if basis == "count" else tr["y"]
    g = pd.DataFrame({"num": num, "base": tr["base"], "n": tr["n_total"]}).groupby(key)
    ae = (g["num"].sum() / num.sum()) / (g["base"].sum() / tr["base"].sum())
    n = g["n"].sum()
    z = n / (n + k)
    return 1 + z * (ae - 1)


def apply_factors(va: pd.DataFrame, col: str, factors: pd.Series) -> np.ndarray:
    key = va[col].astype("string").fillna("missing")
    return key.map(factors).fillna(1.0).to_numpy(dtype=float)


def perils_of(line: str) -> list[tuple[str, str, str]]:
    """(base column suffix, peril, peril group) for each peril of a line, following the
    development config: single-peril groups are named by the group (buildings water,
    fire, ad), others by the peril (scripts/part2_base_perils.py --fine)."""
    from iaa import ultimate

    groups = ultimate.load_config()["development"]["peril_groups"][line]
    return [(p if len(perils) > 1 else g, p, g) for g, perils in groups.items() for p in perils]


def peril_target(tr: pd.DataFrame, line: str, peril: str, group: str) -> np.ndarray:
    """Trended developed cost of one peril on training rows: its reported cost over its
    group's completion (the peril shares the group's development pattern), on-levelled
    with the group's trend (config peril_trend, else the line trend)."""
    from iaa import ultimate

    cfg = ultimate.load_config()
    line_rate = cfg["trend"][line]
    rate = (cfg.get("peril_trend") or {}).get(line, {}).get(group, line_rate)
    gap = np.round(np.log(tr["trend_factor"].to_numpy()) / np.log(1 + line_rate))
    comp = tr[f"comp_{group}"].to_numpy()
    cost = tr[f"cost_{peril}"].fillna(0.0).to_numpy() if f"cost_{peril}" in tr else 0.0
    return np.where(comp > 0, cost / np.where(comp > 0, comp, 1.0), 0.0) * (1 + rate) ** gap


def peril_levels(tr: pd.DataFrame, line: str, k: float = 100.0) -> dict[str, float]:
    """Emerging level of each peril relative to the developed base, per line.

    ratio_p = sum(peril cost) / sum(base_<peril>) on the training rows, shrunk towards the
    line's overall ratio by z = n_p / (n_p + k), n_p = the peril's claim count. Only the
    differences between perils matter (the harness rebalances the line). Prediction:
    sum over perils of base_<peril> * ratio_p (needs harness.base_perils).
    """
    names = perils_of(line)
    ys = {name: peril_target(tr, line, p, g) for name, p, g in names}
    total_y = sum(v.sum() for v in ys.values())
    total_b = sum(tr[f"base_{name}"].sum() for name, _, _ in names)
    line_ratio = total_y / total_b
    out = {}
    for name, peril, _ in names:
        n = float(tr[f"n_{peril}"].sum()) if f"n_{peril}" in tr else 0.0
        z = n / (n + k)
        r = ys[name].sum() / max(tr[f"base_{name}"].sum(), 1e-9)
        out[name] = z * r + (1 - z) * line_ratio
    return out


def peril_prediction(va: pd.DataFrame, line: str, levels: dict[str, float]) -> np.ndarray:
    return sum(va[f"base_{name}"].to_numpy() * levels[name] for name, _, _ in perils_of(line))


def _feature_cols(df: pd.DataFrame, line: str) -> list[str]:
    drop = NON_FEATURES | INADMISSIBLE | DUPLICATES | LINE_DROP.get(line, set())
    drop |= {c for c in df.columns if c.startswith(("cost_", "n_", "ult_", "comp_"))}
    return [c for c in df.columns if c not in drop and df[c].notna().any()]


def _fit_line(tr: pd.DataFrame, va: pd.DataFrame) -> np.ndarray:
    line = str(tr["line"].iloc[0])
    cols = _feature_cols(tr, line)
    mono = [MONOTONE.get(c, 0) for c in cols]
    off_tr = np.log(np.maximum(tr["base"].to_numpy(), 1e-3))
    off_va = np.log(np.maximum(va["base"].to_numpy(), 1e-3))
    out = np.zeros(len(va))
    for seed in SEEDS:
        m = lgb.LGBMRegressor(**PARAMS, seed=seed, monotone_constraints=mono)
        m.fit(tr[cols], tr["y"], init_score=off_tr)
        out += np.exp(off_va + m.predict(va[cols], raw_score=True))
    return out / len(SEEDS)


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(valid))
    for line in ("motor", "buildings", "contents"):
        mask = (valid["line"] == line).to_numpy()
        if not mask.any():
            continue
        va = valid[mask]
        if not USE_CORRECTION:
            pred[mask] = va["base"].to_numpy()
            continue
        pred[mask] = _fit_line(train[train["line"] == line], va)
    return pred
