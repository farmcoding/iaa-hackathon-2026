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
# no licence points on property. gender: admissible in the emerging market (brief; owner
# decision, Karl, 25 Oct). The developed market has no gender field, so the base cannot
# carry it: any gender effect has to be estimated on emerging rows.
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
# v2 (agent, 2026-10-25, backlog P0): each base_<peril> column is its own refit of the
# Part 1 property/motor specification on that peril's trended developed cost (fine split,
# scripts/part2_base_perils.py), not an algebraic share of `base` -- only the motor BI/
# other split is anchored to sum exactly to `base`. So summing the fine per-peril bases at
# level 1 (no emerging-fitted shrinkage, nothing estimated here) is a different prediction
# from `base` for buildings and contents, and P0 asks whether that peril-level view of the
# same developed information is already a better shape than the single total-cost fit,
# before any peril gets its own emerging level (P1+). Deterministic: no SEEDS involved.
USE_PERIL_SUM = True
# L1 (backlog tier L, agent, 25 Oct): motor gender on top of P0. The developed base has no
# gender field (admissible in this market, owner decision 25 Oct); gated with n_tests=1
# per the backlog's pre-registered single-factor item. alpha setting 2 of 3 (0.001/0.01/0.1).
MOTOR_GENDER_ALPHA = 0.01


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


def _design(tr: pd.DataFrame, va: pd.DataFrame, cols: list[str], interactions=()):
    """One-hot categoricals (missing as its own level) and standardised numerics, built on
    the training rows; interactions are (a, b) pairs of categoricals crossed into one."""
    tr, va = tr.copy(), va.copy()
    for a, b in interactions:
        for f in (tr, va):
            f[f"{a}*{b}"] = (
                f[a].astype("string").fillna("NA") + "|" + f[b].astype("string").fillna("NA")
            )
    cols = list(cols) + [f"{a}*{b}" for a, b in interactions]
    parts_tr, parts_va = [], []
    for c in cols:
        num = pd.api.types.is_numeric_dtype(tr[c]) and not pd.api.types.is_bool_dtype(tr[c])
        if num and tr[c].nunique() > 12:
            mu, sd = tr[c].mean(), tr[c].std() or 1.0
            parts_tr.append(((tr[c].fillna(mu) - mu) / sd).rename(c).to_frame())
            parts_va.append(((va[c].fillna(mu) - mu) / sd).rename(c).to_frame())
        else:
            lv = tr[c].astype("string").fillna("NA")
            levels = sorted(lv.unique())[1:]  # first level is the reference
            vv = va[c].astype("string").fillna("NA")
            parts_tr.append(pd.DataFrame({f"{c}={x}": (lv == x).astype(float) for x in levels}))
            parts_va.append(pd.DataFrame({f"{c}={x}": (vv == x).astype(float) for x in levels}))
    # a categorical with one level contributes an empty frame (no rows): give it the rows
    parts_tr = [p if len(p) else pd.DataFrame(index=range(len(tr))) for p in parts_tr]
    parts_va = [p if len(p) else pd.DataFrame(index=range(len(va))) for p in parts_va]
    xt = pd.concat([p.reset_index(drop=True) for p in parts_tr], axis=1)
    xv = pd.concat([p.reset_index(drop=True) for p in parts_va], axis=1)
    return xt.to_numpy(float), xv.to_numpy(float), list(xt.columns)


def offset_glm(tr, va, cols, alpha=0.01, interactions=(), offset="base"):
    """Ridge log-link Tweedie (p=1.5) GLM correction on top of an offset (default `base`).

    For p=1.5 the deviance of y against b*m equals sqrt(b) times the deviance of y/b against
    m, so fitting y/b with sample weight sqrt(b) is an exact offset GLM. The ridge penalty
    (alpha, scikit-learn scale: mean weighted deviance / 2 + alpha/2 * |coef|^2) shrinks every
    coefficient towards 0, i.e. towards the base. Returns the multiplier for va rows
    (the harness rebalances each line's level afterwards).
    """
    from sklearn.linear_model import TweedieRegressor

    b_tr = np.maximum(tr[offset].to_numpy(float), 1e-3)
    xt, xv, _ = _design(tr, va, cols, interactions)
    if xt.shape[1] == 0:  # every candidate has a single level in these rows
        return np.ones(len(va))
    m = TweedieRegressor(power=1.5, link="log", alpha=alpha, solver="newton-cholesky", max_iter=300)
    m.fit(xt, tr["y"].to_numpy(float) / b_tr, sample_weight=np.sqrt(b_tr))
    return m.predict(xv) / np.exp(m.intercept_)


def lr_gate(
    tr, col, n_tests=1, level=0.05, offset="base", interactions=(), basis="count"
) -> tuple[bool, float]:
    """Likelihood-ratio test of one candidate factor against the offset alone, on training rows.

    Null: y ~ offset (intercept only). Alternative: + the factor (unpenalised). Deviance drop
    over the null's Pearson dispersion is compared with chi-square(df), at level / n_tests
    (Bonferroni over the pre-registered list). Returns (passes, p_value). A factor that does
    not pass must not enter a model: with ~350-3,500 claims a year, an untested factor is
    more likely noise than signal (the 25 Oct factor stack).

    basis "count" (default): claim counts n_total with the base as exposure measure
    (assumes the base's severity relativities hold); quasi-Poisson. Cost ("cost") has almost
    no power here: its dispersion is set by a few large bodily-injury and fire claims (on
    2023 motor, gender gave p = 0.21 on cost).
    """
    from scipy.stats import chi2
    from sklearn.linear_model import PoissonRegressor, TweedieRegressor

    from iaa import score

    b = np.maximum(tr[offset].to_numpy(float), 1e-3)
    xt, _, names = _design(
        tr, tr.iloc[:1], [col] if isinstance(col, str) else list(col), interactions
    )
    if xt.shape[1] == 0:  # a single level in these rows: nothing to test
        return False, 1.0
    if basis == "count":
        y = tr["n_total"].to_numpy(float)
        fit = PoissonRegressor(alpha=0.0, solver="newton-cholesky", max_iter=300)
        fit.fit(xt, y / b, sample_weight=b)
        mu1 = b * fit.predict(xt)
        mu0 = b * y.sum() / b.sum()
        pos = y > 0
        safe = np.where(pos, y, 1.0)

        def dev(mu):
            return 2 * (np.where(pos, y * np.log(safe / mu), 0.0) - (y - mu)).sum()

        d0, d1 = dev(mu0), dev(mu1)
        phi = max(((y - mu0) ** 2 / mu0).sum() / (len(y) - 1), 1.0)
    else:
        y = tr["y"].to_numpy(float)
        fit = TweedieRegressor(
            power=1.5, link="log", alpha=0.0, solver="newton-cholesky", max_iter=300
        )
        fit.fit(xt, y / b, sample_weight=np.sqrt(b))
        mu1 = b * fit.predict(xt)
        mu0 = b * (y.sum() / b.sum()) * np.ones_like(y)
        d0, d1 = score.tweedie_deviance(y, mu0).sum(), score.tweedie_deviance(y, mu1).sum()
        phi = ((y - mu0) ** 2 / mu0**1.5).sum() / (len(y) - 1)
    p = float(chi2.sf(max(d0 - d1, 0.0) / phi, df=max(len(names), 1)))
    return p < level / n_tests, p


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


# Owner combination O4 (25 Oct, pre-registered before scoring; one run, normal keep rule):
# the near misses of the day, each sub-margin alone, combined once. Per-peril emerging
# levels (L6, +0.00097), contents geometric 50:50 with the pooled two-market base (better
# on contents in both folds), and a new-business claim-count credibility factor on motor
# and contents (+0.0012). Motor gender (L1, kept) stays.
USE_PERIL_LEVELS = True
PERIL_LEVEL_K = 100.0
USE_POOLED_CONTENTS = True
USE_NB_FACTOR = True
NB_FACTOR_K = 200.0
NB_LINES = ("motor", "contents")


def _line_base(tr_line: pd.DataFrame, frame: pd.DataFrame, line: str) -> np.ndarray:
    """Pre-factor prediction for one line: per-peril bases at their emerging levels, and
    for contents the geometric blend with the pooled base when configured."""
    if USE_PERIL_LEVELS:
        levels = peril_levels(tr_line, line, k=PERIL_LEVEL_K)
    else:
        levels = {name: 1.0 for name, _, _ in perils_of(line)}
    out = peril_prediction(frame, line, levels)
    if USE_POOLED_CONTENTS and line == "contents" and "base_pooled" in frame.columns:
        pooled = np.maximum(frame["base_pooled"].to_numpy(float), 1e-3)
        out = np.sqrt(np.maximum(out, 1e-3) * pooled)
    return out


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(valid))
    for line in ("motor", "buildings", "contents"):
        mask = (valid["line"] == line).to_numpy()
        if not mask.any():
            continue
        va = valid[mask]
        if USE_CORRECTION:
            pred[mask] = _fit_line(train[train["line"] == line], va)
            continue
        if USE_PERIL_SUM:
            tr_line = train[train["line"] == line]
            p0 = _line_base(tr_line, va, line)
            p0_tr = _line_base(tr_line, tr_line, line)
            if line == "motor":
                tr_g = tr_line.assign(p0=p0_tr)
                passes, _ = lr_gate(tr_g, "gender", n_tests=1, offset="p0")
                if passes:
                    p0 = p0 * offset_glm(
                        tr_g, va, ["gender"], alpha=MOTOR_GENDER_ALPHA, offset="p0"
                    )
                    p0_tr = p0_tr * offset_glm(
                        tr_g, tr_line, ["gender"], alpha=MOTOR_GENDER_ALPHA, offset="p0"
                    )
            if USE_NB_FACTOR and line in NB_LINES:
                f = credibility_factors(
                    tr_line.assign(base=p0_tr), "new_business", NB_FACTOR_K, "count"
                )
                p0 = p0 * apply_factors(va, "new_business", f)
            pred[mask] = p0
            continue
        pred[mask] = va["base"].to_numpy()
    return pred
