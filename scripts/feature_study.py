"""Variable study, stages 1-3: one-way analysis, contribution inside the model, and
combined power. Writes reports/features/feature_study.json for the dashboard.

Stage 1 (one-way, per line): exposure-weighted frequency and loss cost by level
(continuous variables in ten equal-exposure bands), 2021-23 vs 2024 stability, train vs
test drift, missing share, and standalone power (fold B D² of a one-variable LightGBM).
Stage 2 (in-model, per line): a LightGBM with the current best settings fitted on fold B
training rows; permutation importance on fold B (drop in D² when shuffled) and mean
|SHAP| (LightGBM pred_contrib).
Stage 3 (combined): group drop tests (refit without each variable group, fold B D²) and
SHAP interaction strength for the top pairs (XGBoost pred_interactions on a sample).

Usage: python -I scripts/feature_study.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from iaa import data, harness, score, ultimate

OUT = data.ROOT / "reports" / "features"
NON = {
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
PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 500,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
}
GROUPS = {
    "driver": [
        "driver_age",
        "years_licensed",
        "ncb_years",
        "licence_points",
        "named_young_driver",
        "telematics_opt_in",
        "telematics_score",
        "annual_mileage",
        "vehicle_use",
    ],
    "vehicle": [
        "vehicle_value_group",
        "vehicle_age",
        "make_model",
        "engine_size_cc",
        "body_type",
        "transmission",
        "seats",
        "vehicle_colour",
        "fuel_type",
        "driver_assist",
        "overnight_parking",
        "cover_type",
    ],
    "property": [
        "year_built",
        "property_age",
        "property_type",
        "roof_type",
        "basement",
        "trees_within_5m",
        "listed_building",
        "bedrooms",
        "energy_rating",
        "solar_panels",
        "sum_insured",
        "construction_type",
        "occupancy",
        "days_unoccupied",
        "smoke_alarms",
        "accidental_damage_cover",
        "mortgage",
        "tenure",
        "security_level",
        "high_value_items",
        "works_from_home",
        "lodgers",
        "bicycle_cover",
    ],
    "location_hazard": [
        "region",
        "urban_density",
        "flood_zone",
        "distance_to_river_km",
        "elevation_m",
        "soil_type",
    ],
    "weather": [
        "hail_days",
        "wind_exposure_index",
        "freezing_days",
        "dry_summer_index",
        "wildfire_index",
        "rainfall_mm",
        "mean_temperature_c",
    ],
    "policy_admin": [
        "voluntary_excess",
        "compulsory_excess",
        "channel",
        "payment_method",
        "missed_payments_12m",
        "new_business",
        "renewal_count",
        "days_quote_to_inception",
        "discount_applied",
        "written_premium",
        "start_month",
    ],
    "customer_marketing": [
        "occupation",
        "marital_status",
        "household_size",
        "credit_band",
        "prior_claims_5y",
        "years_at_address",
        "customer_age",
        "customer_tenure",
        "email_domain",
        "marketing_consent",
        "document_preference",
        "previous_insurer",
        "campaign_code",
        "branch_id",
        "marketing_segment",
        "n_lines_held",
        "policyholder_age",
    ],
    "claim_history": ["hist_n_policy", "hist_cost_policy", "hist_n_customer", "hist_cost_customer"],
}


def features(df: pd.DataFrame) -> list[str]:
    target_like = ("cost_", "n_", "ult_", "comp_")
    return [
        c
        for c in df.columns
        if c not in NON
        and df[c].notna().any()
        and (c == "n_lines_held" or not c.startswith(target_like))
    ]


def fit(tr: pd.DataFrame, cols: list[str], params=PARAMS) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**params)
    m.fit(tr[cols], tr["y"] / tr["exposure"], sample_weight=tr["exposure"])
    return m


def d2_rate(m, va: pd.DataFrame, cols: list[str], tr: pd.DataFrame) -> float:
    rate = m.predict(va[cols])
    rate *= (tr["y"].sum() / tr["exposure"].sum()) / np.average(rate, weights=va["exposure"])
    return score.d2(va["y_ult"], rate * va["exposure"])


def bands(s: pd.Series, e: pd.Series, k: int = 10) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype) or s.nunique() <= 12:
        top = s.value_counts().index[:15]
        return s.astype(object).where(s.isin(top), "other").fillna("missing").astype(str)
    order = s.rank(method="first")
    cum = e.groupby(order).sum().cumsum() / e.sum()
    b = np.minimum((order.map(cum) * k).astype(float).fillna(-1).astype(int), k - 1)
    lab = s.groupby(b).agg(["min", "max"]).apply(lambda r: f"{r['min']:.4g}-{r['max']:.4g}", axis=1)
    return b.map(lab).where(s.notna(), "missing").astype(str)


def one_way(line: str, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, cols, quick) -> dict:
    out = {}
    lc_all = tr["y"].sum() / tr["exposure"].sum()
    for i, c in enumerate(cols):
        t0 = time.time()
        res = {
            "missing_train": float(tr[c].isna().mean()),
            "missing_test": float(te[c].isna().mean()),
        }
        both = pd.concat([tr.assign(_p="2021-23"), va.assign(_p="2024", y=va["y_ult"])])
        both["_b"] = bands(both[c], both["exposure"])
        g = both.groupby(["_b", "_p"]).agg(
            e=("exposure", "sum"), y=("y", "sum"), n=("n_total", "sum")
        )
        g["rel"] = g["y"] / g["e"] / lc_all
        res["levels"] = {
            f"{b}|{p}": {"exposure": float(r.e), "claims": float(r.n), "relativity": float(r.rel)}
            for (b, p), r in g.iterrows()
        }
        if pd.api.types.is_numeric_dtype(tr[c]) and not pd.api.types.is_bool_dtype(tr[c]):
            sd = tr[c].std() or 1
            res["drift_test_sd"] = float((te[c].mean() - tr[c].mean()) / sd)
        if not quick:
            m = fit(tr, [c], {**PARAMS, "n_estimators": 60, "num_leaves": 7})
            res["d2_alone"] = d2_rate(m, va, [c], tr)
        out[c] = res
        print(f"  one-way {line} {i + 1}/{len(cols)} {c} {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = ultimate.load_config()
    fold = cfg["harness"]["folds"]["B"]
    df = harness.dataset()
    tr_all, va_all = harness.split(df, fold["train"], fold["valid"], cfg)
    te_all = df[df["split"] == "test"]
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "lines": {}}
    for line in ("motor", "buildings", "contents"):
        tr, va, te = (x[x["line"] == line] for x in (tr_all, va_all, te_all))
        if a.quick:
            tr, va = tr.sample(60_000, random_state=0), va.sample(30_000, random_state=0)
        cols = features(tr)
        rep: dict = {"n_train": len(tr), "n_valid": len(va)}
        rep["one_way"] = one_way(line, tr, va, te, cols, a.quick)
        # stage 2: permutation importance and SHAP on the full-feature model
        m = fit(tr, cols)
        base = d2_rate(m, va, cols, tr)
        rep["d2_full"] = base
        rng = np.random.default_rng(0)
        perm = {}
        for c in cols:
            v = va.copy()
            v[c] = v[c].to_numpy()[rng.permutation(len(v))]
            perm[c] = base - d2_rate(m, v, cols, tr)
        rep["permutation"] = perm
        sample = va.sample(min(50_000, len(va)), random_state=0)
        contrib = m.predict(sample[cols], pred_contrib=True)
        rep["shap_mean_abs"] = dict(
            zip(cols, np.abs(contrib[:, :-1]).mean(0).astype(float), strict=True)
        )
        print(f"  stage 2 {line} done", flush=True)
        # stage 3a: group drop tests
        drops = {}
        for gname, gcols in GROUPS.items():
            keep = [c for c in cols if c not in gcols]
            if len(keep) == len(cols):
                continue
            drops[gname] = base - d2_rate(fit(tr, keep), va, keep, tr)
        rep["group_drop"] = drops
        # stage 3b: SHAP interaction strength (XGBoost on a sample, top 15 features)
        top = sorted(perm, key=perm.get, reverse=True)[:15]
        s = tr.sample(min(150_000, len(tr)), random_state=1)
        X = s[top].copy()
        for c in top:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].cat.codes.where(X[c].notna(), np.nan).astype(float)
        dm = xgb.DMatrix(X.astype(float), label=s["y"] / s["exposure"], weight=s["exposure"])
        bst = xgb.train(
            {
                "objective": "reg:tweedie",
                "tweedie_variance_power": 1.3,
                "max_depth": 4,
                "eta": 0.1,
                "min_child_weight": 50,
                "subsample": 0.8,
            },
            dm,
            200,
        )
        xs = X.sample(min(5_000, len(X)), random_state=2).astype(float)
        inter = bst.predict(xgb.DMatrix(xs), pred_interactions=True)
        strength = np.abs(inter).mean(0)
        pairs = {
            f"{top[i]} x {top[j]}": float(strength[i, j] * 2)
            for i in range(len(top))
            for j in range(i + 1, len(top))
        }
        rep["interactions"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:20])
        report["lines"][line] = rep
        print(f"{line} done: full-model fold B D² {base:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("feature_study_quick.json" if a.quick else "feature_study.json")).write_text(
        json.dumps(report, indent=1, default=float)
    )
    print("written")


if __name__ == "__main__":
    main()
