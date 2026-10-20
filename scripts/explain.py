"""Test explainability pack for iaa.experiment (Kaggle IAA hackathon, Part 1).

Fits the current experiment once on fold B (train 2021-23, valid 2024) and once (lighter)
on fold A (train 2021-22, valid 2023), capturing every fitted sub-model exactly as
scripts/model_card.py does (monkeypatched predict + _anchor). For each sub-model and
each predictor it uses: a partial-dependence (PD) and accumulated-local-effects (ALE)
relativity curve (normalised to 1.00 at a base level), the one-way observed loss-cost
relativity on 2021-23, an ICE-violation share against an actuarial direction prior, a
fold A vs fold B direction-stability check, and a per-seed relativity band. Also: a
coarse two-way interaction sanity check on the top predictors per line, and a surrogate
Tweedie GLM per line fitted to the model's own predictions on banded features.

Read-only on the repo. Writes only under ~/.cache/iaa-owner/1019/explain/.

Usage: nice -n 15 <repo>/.venv/bin/python -I explain.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Self-locating: if this file is dropped into <repo>/scripts/, REPO is the repo two
# levels up; otherwise fall back to the owner's known checkout path.
_CAND = Path(__file__).resolve().parents[1]
REPO = _CAND if (_CAND / "src" / "iaa").exists() else Path.home() / "Projects" / "iaa-hackathon-2026"
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "3")

OUT_DIR = Path.home() / ".cache" / "iaa-owner" / "1019" / "explain"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "explain.json"

import lightgbm as lgb  # noqa: E402
from interpret.glassbox import ExplainableBoostingRegressor  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import TweedieRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from iaa import harness, score, ultimate  # noqa: E402

SCHEMA = (
    "fold_B.lines[line].sub_models[sub_name].predictors[col] -> {grid, pd_relativity, "
    "seed_relativity (per-seed PD, same shape as grid for n_seeds>1), ale "
    "{grid, value (1.00-base relativity), raw_log_effect (centered log effect)}, "
    "spearman_vs_level (PD Spearman vs. grid order), spearman_fold_A (same metric "
    "from fold_A_light, filled in by direction_stability), ice_sign_mean, "
    "ice_violation_share_vs_model, ice_violation_share_vs_prior, is_ordered, "
    "monotone_constrained, prior {direction, note}, "
    "observed_oneway_B {grid, relativity, spearman_vs_level} (same key suffixed "
    "_A in fold_A_light), flags[]}. "
    "fold_A_light mirrors fold_B but only for each sub-model's top-12-ranked "
    "predictors (light pass). "
    "interactions[line][sub_name] -> list of {a, b, max_log_deviation_from_additive, "
    "joint_relativity (3x3), additive_relativity (3x3), flag}. "
    "surrogate[line] -> {features, r2_weighted, d2_tweedie, intercept_rate, "
    "band_references, relativities {term: relativity}}."
)

RNG = np.random.default_rng(0)
PD_SAMPLE = 20_000  # rows for PD/ALE evaluation, per line (owner budget)
ICE_SAMPLE = 1_500  # rows for the ICE-violation share (cheaper, still representative)
GRID_POINTS = 8

# ---------------------------------------------------------------------------
# Actuarial direction priors (owner reasoning, not fitted from anything). "U" =
# U-shaped (young/old or small/large extremes higher); None = no stated prior.
# Applied by feature name wherever it recurs across lines.
PRIORS: dict[str, tuple[str | None, str]] = {
    "ncb_years": ("down", "more no-claims years -> lower expected cost"),
    "annual_mileage": ("up", "more driving exposure -> higher expected cost"),
    "voluntary_excess": ("down", "higher excess selects/discourages small claims"),
    "compulsory_excess": ("down", "same mechanism as voluntary excess"),
    "sum_insured": ("up", "larger insured value -> larger claims, all else equal"),
    "flood_zone": ("up", "higher flood-zone band -> higher flood peril cost"),
    "prior_claims_5y": ("up", "claims history is the strongest classical risk signal"),
    "driver_age": ("U", "young and very old drivers carry higher risk than mid-career"),
    "policyholder_age": ("U", "same U-shape reasoning as driver age"),
    "customer_age": (None, "age proxies many things; no single-direction prior stated"),
    "vehicle_age": (None, "older/cheaper vs. mechanically worn: no stated prior"),
    "licence_points": ("up", "points are a direct driving-record risk signal"),
    "years_licensed": ("down", "more driving experience -> lower expected cost"),
    "missed_payments_12m": ("up", "payment stress correlates with claims propensity"),
    "credit_band": ("up", "worse credit band -> higher expected cost (as coded)"),
    "elevation_m": ("down", "higher elevation -> lower flood exposure"),
    "distance_to_river_km": ("down", "further from a river -> lower flood exposure"),
    "wildfire_index": ("up", "higher wildfire index -> higher peril cost"),
    "rainfall_mm": ("up", "wetter areas -> higher water-damage propensity"),
    "freezing_days": ("up", "more freeze days -> more burst-pipe claims"),
    "dry_summer_index": ("up", "drier summers -> more subsidence claims (clay soils)"),
    "trees_within_5m": ("up", "nearby trees -> more subsidence/root/storm claims"),
    "security_level": ("down", "better security -> lower theft/break-in cost"),
    "high_value_items": ("up", "more high-value items -> higher contents claim cost"),
    "lodgers": ("up", "more occupants -> more claim opportunity"),
    "property_age": ("up", "older building fabric -> higher maintenance-related risk"),
    "year_built": ("down", "newer build year is the mirror of property_age"),
    "urban_density": ("up", "denser areas -> higher theft/collision frequency"),
    "named_young_driver": ("up", "a young named driver raises household risk"),
    "vehicle_value_group": ("up", "higher vehicle value -> higher claim cost"),
    "engine_size_cc": (None, "weak, ambiguous prior (power vs. usage pattern)"),
    "hist_n_policy": ("up", "more prior claims on the policy -> higher expected cost"),
    "hist_cost_policy": ("up", "same mechanism, cost-weighted"),
    "hist_n_customer": ("up", "same mechanism at customer level"),
    "renewal_count": (None, "tenure proxies both loyalty and survivorship selection"),
    "telematics_score": (None, "scale/direction not documented here: no prior stated"),
    "works_from_home": (None, "occupancy effect is plausible but weak either way"),
    "solar_panels": (None, "no clear generic prior"),
    "bedrooms": (None, "mostly a size proxy; no stated prior"),
    "channel": (None, "distribution channel, not a risk characteristic per se"),
}


def prior_for(feature: str) -> tuple[str | None, str]:
    return PRIORS.get(feature, (None, "no prior catalogued for this feature"))


# ---------------------------------------------------------------------------
class _Rec:
    def __init__(self) -> None:
        self.events: list[tuple] = []


def _instrument(exp, rec: _Rec) -> None:
    class LGBM(lgb.LGBMRegressor):
        def predict(self, X, *a, **k):  # noqa: N803
            rec.events.append(("lgbm", self, X.copy()))
            return super().predict(X, *a, **k)

    class EBM(ExplainableBoostingRegressor):
        def predict(self, X, *a, **k):  # noqa: N803
            rec.events.append(("ebm", self, X.copy()))
            return super().predict(X, *a, **k)

    shim = types.SimpleNamespace(**{k: getattr(lgb, k) for k in dir(lgb) if not k.startswith("__")})
    shim.LGBMRegressor = LGBM
    exp.lgb = shim
    exp.ExplainableBoostingRegressor = EBM
    orig = exp._anchor

    def anchor(rate, target_lc, va):
        rec.events.append(("anchor", float(target_lc)))
        return orig(rate, target_lc, va)

    exp._anchor = anchor


def _segments(events: list[tuple]) -> dict[str, list[tuple]]:
    out, cur, need = {}, [], {"motor": 2, "buildings": 1, "contents": 1}
    lines = iter(need)
    line = next(lines)
    for e in events:
        cur.append(e)
        if e[0] == "anchor" and sum(x[0] == "anchor" for x in cur) == need[line]:
            out[line], cur = cur, []
            line = next(lines, None)
    return out


def _submodels(line: str, seg: list[tuple]) -> dict[str, list]:
    """name -> list of (model, X) events, one group per reported relativity curve."""
    lg = [(e[1], e[2]) for e in seg if e[0] == "lgbm"]
    eb = [(e[1], e[2]) for e in seg if e[0] == "ebm"]
    if line == "motor":
        half = len(lg) // 2
        return {"bi_frequency": lg[:half], "other_perils": lg[half:]}
    out = {"lgbm": lg}
    if eb:
        out["ebm"] = eb
    return out


def _mean_predict(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = []
    for m in models:
        if isinstance(m, ExplainableBoostingRegressor):
            preds.append(ExplainableBoostingRegressor.predict(m, X))
        else:
            preds.append(lgb.LGBMRegressor.predict(m, X))
    return np.mean(preds, axis=0)


def _per_seed_predict(models: list, X: pd.DataFrame) -> list[np.ndarray]:
    out = []
    for m in models:
        if isinstance(m, ExplainableBoostingRegressor):
            out.append(ExplainableBoostingRegressor.predict(m, X))
        else:
            out.append(lgb.LGBMRegressor.predict(m, X))
    return out


# ---------------------------------------------------------------------------
def _grid_for(raw: pd.Series, enc: pd.Series, is_cat_native: bool) -> list[dict]:
    """Returns [{"label": str, "value": <value to assign into the encoded column>}]."""
    if is_cat_native:
        counts = raw.value_counts()
        cats = list(counts.index[:12])
        return [{"label": str(c), "value": c} for c in cats]
    if pd.api.types.is_numeric_dtype(enc) and not isinstance(raw.dtype, pd.CategoricalDtype):
        qs = np.linspace(0.05, 0.95, GRID_POINTS)
        vals = np.unique(np.quantile(raw.dropna().to_numpy(), qs))
        return [{"label": f"{v:.3g}", "value": float(v)} for v in vals]
    # high-cardinality categorical, credibility-encoded to a float: grid over the
    # top raw levels, each mapped to its mean encoded value (regulator-readable labels)
    means = enc.groupby(raw).mean()
    counts = raw.value_counts()
    top = [c for c in counts.index[:12] if c in means.index]
    return [{"label": str(c), "value": float(means[c])} for c in top]


def _pd_ale_ice(
    predict_fn_each_seed,
    X: pd.DataFrame,
    raw: pd.Series,
    col: str,
    grid: list[dict],
    ice_idx: np.ndarray,
    prior_dir: str | None = None,
) -> dict:
    n_seeds = len(predict_fn_each_seed)
    obs_mean = float(np.mean([fn(X) for fn in predict_fn_each_seed]))  # unmodified-X base rate

    def set_col(Xb: pd.DataFrame, value) -> pd.DataFrame:
        Xm = Xb.copy()
        if isinstance(Xm[col].dtype, pd.CategoricalDtype):
            Xm[col] = pd.Categorical([value] * len(Xm), categories=Xm[col].cat.categories)
        else:
            Xm[col] = value
        return Xm

    # PD per seed, mean over all sampled rows
    pd_seed = np.zeros((n_seeds, len(grid)))
    ice_curves = np.zeros((len(ice_idx), len(grid)))
    for gi, g in enumerate(grid):
        Xm = set_col(X, g["value"])
        preds = [fn(Xm) for fn in predict_fn_each_seed]
        pd_seed[:, gi] = [p.mean() for p in preds]
        mean_pred = np.mean(preds, axis=0)
        ice_curves[:, gi] = mean_pred[ice_idx]
    pd_mean = pd_seed.mean(axis=0)

    # ALE: only meaningful with an ordering -> continuous / already-ordered grids
    ale = None
    is_ordered = not isinstance(raw.dtype, pd.CategoricalDtype)
    if is_ordered and len(grid) > 2:
        vals = np.array([g["value"] for g in grid])
        order = np.argsort(vals)
        vals, grid_o = vals[order], [grid[i] for i in order]
        bin_means = np.zeros(len(vals) - 1)
        for i in range(len(vals) - 1):
            lo, hi = vals[i], vals[i + 1]
            in_bin = (X[col].to_numpy() >= lo) & (X[col].to_numpy() <= hi)
            if in_bin.sum() < 20:
                bin_means[i] = 0.0
                continue
            Xb = X.loc[in_bin]
            lo_pred = np.mean([fn(set_col(Xb, lo)) for fn in predict_fn_each_seed], axis=0)
            hi_pred = np.mean([fn(set_col(Xb, hi)) for fn in predict_fn_each_seed], axis=0)
            bin_means[i] = (hi_pred - lo_pred).mean()
        uncentered = np.concatenate([[0.0], np.cumsum(bin_means)])
        weights = np.array(
            [max(((X[col] >= vals[i]) & (X[col] <= vals[i + 1])).sum(), 1) for i in range(len(vals) - 1)]
        )
        mid = 0.5 * (uncentered[:-1] + uncentered[1:])
        centered = uncentered - np.average(mid, weights=weights)
        # ALE is additive on the rate scale here (finite differences of predict_fn,
        # not a log-link GLM), so the 1.00-base relativity is the rate-scale ratio,
        # not a ratio of the centered effect itself (which can be negative).
        rate_curve = obs_mean + centered
        base_rate_ale = rate_curve[0] if rate_curve[0] != 0 else obs_mean
        ale = {
            "grid": [g["label"] for g in grid_o],
            "value": list(rate_curve / base_rate_ale),
            "raw_log_effect": list(centered),
        }

    base = pd_mean[0] if pd_mean[0] != 0 else pd_mean.mean()
    pd_rel = (pd_mean / base).tolist()
    seed_rel = (pd_seed / base).tolist()

    # ICE violation share: fraction of sampled rows whose curve direction disagrees
    # with the prior (Spearman sign); only computed when a prior with a sign exists
    grid_x = np.array([g["value"] for g in grid]) if is_ordered else np.arange(len(grid))
    ice_signs = []
    for r in range(ice_curves.shape[0]):
        if np.allclose(ice_curves[r], ice_curves[r, 0]):
            ice_signs.append(0.0)
            continue
        rho, _ = spearmanr(grid_x, ice_curves[r])
        ice_signs.append(rho if np.isfinite(rho) else 0.0)
    model_rho, _ = spearmanr(grid_x, pd_mean) if is_ordered else (np.nan, np.nan)
    prior_sign = {"up": 1.0, "down": -1.0}.get(prior_dir)

    return {
        "grid": [g["label"] for g in grid],
        "pd_relativity": pd_rel,
        "seed_relativity": seed_rel,
        "ale": ale,
        "spearman_vs_level": float(model_rho) if np.isfinite(model_rho) else None,
        "ice_sign_mean": float(np.mean(ice_signs)),
        "ice_violation_share_vs_model": float(
            sum(1 for s in ice_signs if np.isfinite(model_rho) and s * model_rho < -1e-6)
            / max(len(ice_signs), 1)
        )
        if is_ordered
        else None,
        "ice_violation_share_vs_prior": float(
            sum(1 for s in ice_signs if s * prior_sign < -1e-6) / max(len(ice_signs), 1)
        )
        if is_ordered and prior_sign is not None
        else None,
        "is_ordered": is_ordered,
    }


def _observed_oneway(tr: pd.DataFrame, col: str, grid: list[dict], is_cat_native: bool, numerator: np.ndarray) -> dict:
    """numerator/exposure, by band -- caller picks the numerator to match the
    sub-model (claim count for motor BI frequency, trended peril cost for motor
    other-perils, total developed cost for the property lines)."""
    y = numerator
    exposure = tr["exposure"].to_numpy(dtype=float)
    raw = tr[col]
    if is_cat_native or isinstance(raw.dtype, pd.CategoricalDtype):
        edges = None
    else:
        vals = np.array([g["value"] for g in grid])
        edges = np.concatenate([[-np.inf], 0.5 * (vals[:-1] + vals[1:]), [np.inf]])
    rates = []
    for i, g in enumerate(grid):
        if edges is None:
            m = (raw == g["value"]).to_numpy() if is_cat_native else None
            if m is None:  # high-card encoded: match by raw category label stored in grid
                m = (raw.astype(str) == g["label"]).to_numpy()
        else:
            m = (raw.to_numpy() >= edges[i]) & (raw.to_numpy() < edges[i + 1])
        if m.sum() < 30:
            rates.append(np.nan)
            continue
        rates.append(y[m].sum() / max(exposure[m].sum(), 1e-9))
    rates = np.array(rates, dtype=float)
    base = rates[np.isfinite(rates)][0] if np.isfinite(rates).any() else np.nan
    is_ordered = edges is not None
    rho = None
    if is_ordered and np.isfinite(rates).sum() > 2:
        idx = np.arange(len(rates))[np.isfinite(rates)]
        r, _ = spearmanr(idx, rates[np.isfinite(rates)])
        rho = float(r) if np.isfinite(r) else None
    return {
        "grid": [g["label"] for g in grid],
        "relativity": (rates / base).tolist() if base else [None] * len(grid),
        "spearman_vs_level": rho,
    }


# ---------------------------------------------------------------------------
def fit_fold(df: pd.DataFrame, cfg: dict, fold_name: str) -> dict:
    """Fit the current experiment once on a fold, instrumented. Reused by run_fold,
    interaction_check and surrogate_glm so each fold is fit exactly once."""
    exp = importlib.reload(importlib.import_module("iaa.experiment"))
    exp.PARAMS["num_threads"] = 3
    exp.EBM_PARAMS["n_jobs"] = 1  # avoid loky pool churn across thousands of .predict() calls in PD/ALE loops
    f = cfg["harness"]["folds"][fold_name]
    tr, va = harness.split(df, f["train"], f["valid"], cfg)
    rec = _Rec()
    _instrument(exp, rec)
    t0 = time.time()
    pred = exp.fit_predict(tr, harness.hide_target(va))
    print(f"fold {fold_name}: fit {time.time() - t0:.0f}s", flush=True)
    return {"exp": exp, "tr": tr, "va": va, "pred": pred, "segs": _segments(rec.events)}


def run_fold(fitted: dict, fold_name: str, light: bool, card: dict | None = None) -> dict:
    exp, tr, va, pred, segs = fitted["exp"], fitted["tr"], fitted["va"], fitted["pred"], fitted["segs"]

    out: dict = {"lines": {}}
    for line, seg in segs.items():
        subs = _submodels(line, seg)
        m_valid = (va["line"] == line).to_numpy()
        va_line = va[m_valid]
        tr_line = tr[tr["line"] == line]
        bi_cap = exp.BI_CAP if line == "motor" else None
        line_out: dict = {"sub_models": {}}
        for sub_name, models_X in subs.items():
            models = [mX[0] for mX in models_X]
            X_full = models_X[0][1]
            n = len(X_full)
            pd_idx = RNG.choice(n, size=min(PD_SAMPLE, n), replace=False)
            ice_idx_local = RNG.choice(len(pd_idx), size=min(ICE_SAMPLE, len(pd_idx)), replace=False)
            X = X_full.iloc[pd_idx].reset_index(drop=True)
            raw_df = va_line.iloc[pd_idx].reset_index(drop=True)
            cols = list(X.columns)
            predict_fns = [
                (lambda Xb, mm=m: ExplainableBoostingRegressor.predict(mm, Xb))
                if isinstance(m, ExplainableBoostingRegressor)
                else (lambda Xb, mm=m: lgb.LGBMRegressor.predict(mm, Xb))
                for m in models
            ]
            sub_out: dict = {"predictors": {}}
            if light and card:
                ranked = [r["feature"] for r in card["lines"][line]["features"]]
                use_cols = [c for c in ranked if c in cols][:12]
            elif light:
                use_cols = cols[:12]  # no ranking available: first columns in frame order
            else:
                use_cols = cols
            # observed one-way numerator: match each sub-model's own fitted target,
            # not the line's total cost (the BI sub-model is a claim-count model;
            # other-perils uses the group's own trended cost; property uses y).
            if sub_name == "bi_frequency":
                numerator = tr_line["n_bodily_injury"].to_numpy(dtype=float)
            elif sub_name == "other_perils":
                numerator = (tr_line["ult_other"] * tr_line["trend_factor"]).to_numpy(dtype=float)
            else:
                numerator = np.minimum(tr_line["y"].to_numpy(dtype=float), bi_cap) if bi_cap else tr_line["y"].to_numpy(dtype=float)
            for col in use_cols:
                is_cat_native = isinstance(raw_df[col].dtype, pd.CategoricalDtype) and not pd.api.types.is_numeric_dtype(X[col])
                grid = _grid_for(raw_df[col], X[col], is_cat_native)
                if len(grid) < 2:
                    continue
                prior_dir, prior_note = prior_for(col)
                res = _pd_ale_ice(predict_fns, X, raw_df[col], col, grid, ice_idx_local, prior_dir)
                res["prior"] = {"direction": prior_dir, "note": prior_note}
                res["monotone_constrained"] = col in exp.MONOTONE and sub_name != "ebm"
                observed = None
                if not light:
                    observed = _observed_oneway(tr_line, col, grid, is_cat_native, numerator)
                    res["observed_oneway_B" if fold_name == "B" else "observed_oneway_A"] = observed
                flags = []
                rho = res["spearman_vs_level"]
                if prior_dir == "up" and rho is not None and rho < -0.2:
                    flags.append("model direction opposes the 'up' prior")
                if prior_dir == "down" and rho is not None and rho > 0.2:
                    flags.append("model direction opposes the 'down' prior")
                if observed is not None and observed.get("spearman_vs_level") is not None and rho is not None:
                    rho_obs = observed["spearman_vs_level"]
                    if rho * rho_obs < 0 and abs(rho) > 0.2 and abs(rho_obs) > 0.2:
                        flags.append("model direction contradicts the observed one-way relativity on 2021-23")
                series = np.array(res["pd_relativity"])
                if len(series) > 3:
                    diffs = np.diff(series)
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                    if sign_changes >= 2 and not res["monotone_constrained"]:
                        flags.append(f"non-monotone wiggle ({sign_changes} sign changes) a regulator would query")
                if res.get("ice_violation_share_vs_model") and res["ice_violation_share_vs_model"] > 0.3:
                    flags.append(
                        f"{res['ice_violation_share_vs_model']:.0%} of sampled ICE curves disagree with the model's own average direction"
                    )
                if res.get("ice_violation_share_vs_prior") and res["ice_violation_share_vs_prior"] > 0.3 and not res["monotone_constrained"]:
                    flags.append(
                        f"{res['ice_violation_share_vs_prior']:.0%} of sampled ICE curves disagree with the actuarial prior direction"
                    )
                res["flags"] = flags
                sub_out["predictors"][col] = res
            line_out["sub_models"][sub_name] = sub_out
        out["lines"][line] = line_out
    out["pred_d2"] = score.d2(va["y_ult"].to_numpy(), pred)
    return out


def direction_stability(fold_b: dict, fold_a: dict) -> None:
    for line, lb in fold_b["lines"].items():
        la = fold_a["lines"].get(line, {})
        for sub_name, sb in lb["sub_models"].items():
            sa = la.get("sub_models", {}).get(sub_name, {})
            for col, rb in sb["predictors"].items():
                ra = sa.get("predictors", {}).get(col)
                rb["spearman_fold_A"] = ra["spearman_vs_level"] if ra else None
                if ra and rb["spearman_vs_level"] is not None and ra["spearman_vs_level"] is not None:
                    if rb["spearman_vs_level"] * ra["spearman_vs_level"] < -0.05:
                        rb["flags"].append("direction flips between fold A and fold B (unstable)")


# ---------------------------------------------------------------------------
def interaction_check(fitted: dict, card: dict) -> dict:
    """Coarse 3x3 two-way PD check on the top-5 predictors per line, for every
    LGBM sub-model (the EBM is skipped: interactions=0 by construction, so it
    contributes none). Tweedie/Poisson LightGBM is a log-link, multiplicative
    model, so additivity is tested on the log-rate scale, not the rate scale --
    on the rate scale even a model with zero interactions fails an additive test.
    Reuses the fold B fit already captured by fit_fold (no refit)."""
    segs = fitted["segs"]
    out = {}
    for line, seg in segs.items():
        top = [r["feature"] for r in card["lines"][line]["features"][:5]]
        subs = _submodels(line, seg)
        out[line] = {}
        for sub_name, models_X in subs.items():
            if sub_name == "ebm":
                continue
            models = [m for m, _ in models_X]
            X_full = models_X[0][1]
            idx = RNG.choice(len(X_full), size=min(2000, len(X_full)), replace=False)
            X = X_full.iloc[idx].reset_index(drop=True)

            def pfn(Xb, _models=models):
                return np.mean([lgb.LGBMRegressor.predict(m, Xb) for m in _models], axis=0)

            def setval(Xb: pd.DataFrame, col: str, value) -> pd.DataFrame:
                Xm = Xb.copy()
                if isinstance(Xm[col].dtype, pd.CategoricalDtype):
                    Xm[col] = pd.Categorical([value] * len(Xm), categories=Xm[col].cat.categories)
                else:
                    Xm[col] = value
                return Xm

            pairs = []
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    a, b = top[i], top[j]
                    if a not in X.columns or b not in X.columns:
                        continue
                    ga = (
                        np.quantile(X[a].dropna(), [0.1, 0.5, 0.9])
                        if pd.api.types.is_numeric_dtype(X[a])
                        else X[a].value_counts().index[:3]
                    )
                    gb = (
                        np.quantile(X[b].dropna(), [0.1, 0.5, 0.9])
                        if pd.api.types.is_numeric_dtype(X[b])
                        else X[b].value_counts().index[:3]
                    )
                    base_all = max(pfn(X).mean(), 1e-9)
                    log_a = np.zeros(3)
                    log_b = np.zeros(3)
                    log_joint = np.zeros((3, 3))
                    for ia, va_ in enumerate(ga):
                        log_a[ia] = np.log(max(pfn(setval(X, a, va_)).mean(), 1e-9) / base_all)
                    for ib, vb_ in enumerate(gb):
                        log_b[ib] = np.log(max(pfn(setval(X, b, vb_)).mean(), 1e-9) / base_all)
                    for ia, va_ in enumerate(ga):
                        for ib, vb_ in enumerate(gb):
                            Xj = setval(setval(X, a, va_), b, vb_)
                            log_joint[ia, ib] = np.log(max(pfn(Xj).mean(), 1e-9) / base_all)
                    additive = log_a[:, None] + log_b[None, :]
                    dev = float(np.abs(log_joint - additive).max())
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "max_log_deviation_from_additive": dev,
                            "joint_relativity": np.exp(log_joint).tolist(),
                            "additive_relativity": np.exp(additive).tolist(),
                            "flag": bool(dev > np.log(1.10)),
                        }
                    )
            out[line][sub_name] = pairs
    return out


# ---------------------------------------------------------------------------
def surrogate_glm(fitted: dict, card: dict) -> dict:
    """Surrogate GLM per line, fitted to the model's own final (anchored) line rate.
    This target deliberately differs from the per-sub-model PD/ALE curves above,
    where the anchor cancels in the 1.00-base normalisation; the surrogate instead
    reproduces what a regulator actually sees (the anchored rate per policy)."""
    va, pred = fitted["va"], fitted["pred"]
    out = {}
    for line in ("motor", "buildings", "contents"):
        m = (va["line"] == line).to_numpy()
        exposure = va.loc[m, "exposure"].to_numpy()
        rate = pred[m] / np.maximum(exposure, 1e-9)
        feats = [r["feature"] for r in card["lines"][line]["features"][:12]]
        X = va.loc[m, feats].copy()
        bands = {}
        design = pd.DataFrame(index=X.index)
        for c in feats:
            if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object:
                top = X[c].value_counts().index[:6]
                cat = X[c].astype(str).where(X[c].isin(top), other="_other_")
                dummies = pd.get_dummies(cat, prefix=c, drop_first=True)
                design = pd.concat([design, dummies], axis=1)
                bands[c] = {"type": "categorical", "reference": str(X[c].value_counts().index[0]) if X[c].value_counts().index[0] in top else "_other_"}
            else:
                q = X[c].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
                q = np.unique(q)
                band = pd.cut(X[c], bins=q, include_lowest=True, duplicates="drop")
                dummies = pd.get_dummies(band, prefix=c, drop_first=True)
                dummies.columns = [str(c) for c in dummies.columns]
                design = pd.concat([design, dummies], axis=1)
                bands[c] = {"type": "numeric_band", "reference": str(band.cat.categories[0])}
        design = design.fillna(0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=500)
            glm.fit(design, rate, sample_weight=exposure)
        surrogate_pred = glm.predict(design)
        r2 = r2_score(rate, surrogate_pred, sample_weight=exposure)
        d2_surrogate = score.d2(rate * exposure, surrogate_pred * exposure)
        coefs = {name: float(np.exp(c)) for name, c in zip(design.columns, glm.coef_, strict=True)}
        out[line] = {
            "features": feats,
            "r2_weighted": float(r2),
            "d2_tweedie": float(d2_surrogate),
            "intercept_rate": float(np.exp(glm.intercept_)),
            "band_references": bands,
            "relativities": coefs,
        }
    return out


def main() -> None:
    import argparse

    global JSON_OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(JSON_OUT), help="output JSON path (default: %(default)s)"
    )
    args = ap.parse_args()
    JSON_OUT = Path(args.out)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = ultimate.load_config()
    print("loading dataset (cached if fingerprint matches)...", flush=True)
    df = harness.dataset()
    print(f"dataset ready {time.time() - t0:.0f}s, {len(df)} rows", flush=True)

    with open(REPO / "data" / "cache" / "model_card.json") as fh:
        card = json.load(fh)

    result: dict = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "schema": SCHEMA,
            "pd_sample": PD_SAMPLE,
            "ice_sample": ICE_SAMPLE,
            "grid_points": GRID_POINTS,
            "num_threads": 3,
        }
    }

    print("=== fold B (full pack) ===", flush=True)
    fitted_b = fit_fold(df, cfg, "B")
    fold_b = run_fold(fitted_b, "B", light=False)
    result["fold_B"] = fold_b
    JSON_OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote partial {JSON_OUT} after fold B, {time.time() - t0:.0f}s", flush=True)

    print("=== interaction sanity check (reusing fold B fit) ===", flush=True)
    result["interactions"] = interaction_check(fitted_b, card)

    print("=== surrogate GLM (reusing fold B fit) ===", flush=True)
    result["surrogate"] = surrogate_glm(fitted_b, card)
    JSON_OUT.write_text(json.dumps(result, indent=1, default=float))
    del fitted_b  # drop the fold B models before fold A's fit to bound memory

    print("=== fold A (direction/stability only) ===", flush=True)
    fitted_a = fit_fold(df, cfg, "A")
    fold_a = run_fold(fitted_a, "A", light=True, card=card)
    result["fold_A_light"] = fold_a
    direction_stability(fold_b, fold_a)

    result["meta"]["total_seconds"] = round(time.time() - t0)
    JSON_OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"DONE {time.time() - t0:.0f}s -> {JSON_OUT}", flush=True)


if __name__ == "__main__":
    main()
