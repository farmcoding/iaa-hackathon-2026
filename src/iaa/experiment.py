"""THE EXPERIMENT FILE. This is the only file the experiment agent may edit.

Contract (enforced by harness.py):
    fit_predict(train, valid) -> np.ndarray of expected ultimate net claim cost per
    valid row, finite and >= 0, in the order of valid.

train carries the developed, on-levelled target `y` (plus per-group developed costs
ult_<group>, reported counts n_<peril>, trend_factor). valid carries features only.
Both frames hold all three lines; `line` says which.
"""

from __future__ import annotations

import os
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.preprocessing import TargetEncoder

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
    # owner 2026-10-18: tested once restored (primary 0.1334 vs 0.1339), no gain
    "lines_held",
    # owner 2026-10-19 (Karl): not an admissible rating factor. Review found it independent
    # of every other customer field (Cramer's V < 0.01) and its relativities flip rank
    # between years: noise. reports/reviews/2026-10-19_features.md section 3
    "email_domain",
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
    # backlog 2.63 refinement: NCB earned on a brand-new policy is transferred from the
    # prior insurer and unverified, unlike renewal NCB; split it onto its own monotone
    # column instead of leaving the discount shared with renewals via one `ncb_years` curve.
    "ncb_years_new": -1,
}
# The harness re-runs a candidate keep with IAA_SEED_OFFSET set, to confirm the gain on
# fresh seeds. Every random choice in this file must derive from SEED_OFFSET or SEEDS.
SEED_OFFSET = int(os.environ.get("IAA_SEED_OFFSET", "0"))
SEEDS = tuple(SEED_OFFSET + s for s in (0, 1, 2))
# backlog 2.1: categoricals above this many levels (e.g. make_model, occupation,
# previous_insurer) get out-of-fold credibility encoding instead of LightGBM's native
# categorical splits; low-cardinality ones (e.g. the 12 regions) stay native. Detected
# by cardinality rather than named, since Part 2's market will have different columns.
HIGH_CARD_MIN = 15
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
# backlog 2.15: EBM blended with LightGBM on buildings and contents. results.tsv
# b54959b/2abf7b5 ran 1884-1889s against the 1800s budget; halving outer_bags (2->1,
# EBM's internal bagging) only saved 5s, so the external SEEDS loop was the cost. EBM
# now fits once, on SEEDS[0], with outer_bags back to 2 for internal diversity.
EBM_PARAMS = {
    "objective": "tweedie_deviance:variance_power=1.5",
    "interactions": 0,
    "outer_bags": 2,
}
# backlog 2.13 / Tier 0e.3: CatBoost's native Tweedie as a third equal-weight property
# blend member. Validated single-seed (results.tsv `b46f2be`, `89324f8`: +0.0014/+0.0018
# vs their day's champion, both folds up each time) and re-tested alone on the final
# pruned, no-premium, BI-Pareto-capped champion (results.tsv `9d37636`, +0.00067, both
# folds up) -- every other attempt stacked it with another change that scored lower than
# this single-seed solo form.
CATBOOST_PARAMS = {
    "loss_function": "Tweedie:variance_power=1.5",
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.08,
    "allow_writing_files": False,
    "verbose": False,
}


def _ebm_safe(x: pd.DataFrame) -> pd.DataFrame:
    # EBM's own missing-value handling is for continuous features; nominal (category
    # dtype) columns here can carry a real NaN level, so coerce those to the string
    # "missing" the same way CatBoost required, rather than assume EBM handles it.
    x = x.copy()
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            na = x[c].isna()
            x[c] = x[c].astype(str)
            x.loc[na, c] = "missing"
    return x


# owner 2026-10-19 (Karl): fields without a rating rationale on a line. written_premium is
# the incumbent's price, an output of someone else's tariff, so no line rates on it (motor
# renewal premium also carries experience re-rating on claims reported after renewal).
# licence_points is a driving record stored on the customer. Motor keeps the address and
# weather fields: hail and flood drive its glass/hail and own damage perils.
# reports/reviews/2026-10-19_features.md
LINE_DROP = {
    # distance to river: no rating rationale on motor (owner 2026-10-20, Karl, option 2b)
    "motor": {"written_premium", "distance_to_river_km"},
    "buildings": {"written_premium", "licence_points"},
    "contents": {"written_premium", "licence_points"},
}


# owner 2026-10-19 (Karl): (1) sales and admin fields that describe how the policy was
# sold, not the risk, dropped on every line: campaign_code, marketing_segment (a relabel of
# age and credit band), branch_id (not a geography proxy), days_quote_to_inception,
# marketing_consent; channel is kept (comparison-site business runs 5-14% high on every
# line in both periods, unrelated to age or region). (2) Pruning: predictors that added
# at most 0.0001 of within-line D2 when shuffled on 2024 (reports/features/model_card.json).
# Pruning alone scored 0.14033 vs 0.13867, confirmed on fresh seeds at 0.14002.
PRUNED = {
    "motor": [
        "basement",
        "bedrooms",
        "branch_id",
        "campaign_code",
        "compulsory_excess",
        "customer_age",
        "customer_tenure",
        "days_quote_to_inception",
        "discount_applied",
        "document_preference",
        "driver_assist",
        "elevation_m",
        "energy_rating",
        "flood_zone",
        "hail_days",
        "hist_cost_customer",
        "hist_n_customer",
        "household_size",
        "listed_building",
        "marital_status",
        "marketing_consent",
        "marketing_segment",
        # backlog 2.63: diagnostics.md flags new_business=True as a PERSISTENT miss
        # (9.0% of motor exposure, A/E 1.107 fold A / 1.136 fold B, both under-predicted)
        # that survived pruning for near-zero whole-model D2 on its own; restoring it
        # together with the ncb_years_new split below (see MONOTONE) targets that miss
        # directly instead of relying on the pruned-out raw flag alone.
        "occupation",
        "previous_insurer",
        "property_age",
        "property_type",
        "rainfall_mm",
        "renewal_count",
        "roof_type",
        "soil_type",
        "solar_panels",
        "telematics_opt_in",
        "transmission",
        "trees_within_5m",
        "vehicle_use",
        "wind_exposure_index",
        "year_built",
        "years_at_address",
    ],
    "buildings": [
        "hist_n_customer",  # owner 2026-10-20 (Karl, 3.45): no customer-level claim history
        "branch_id",
        "campaign_code",
        "compulsory_excess",
        "customer_age",
        "customer_tenure",
        "days_quote_to_inception",
        "days_unoccupied",
        "discount_applied",
        "document_preference",
        "freezing_days",
        "hail_days",
        "hist_cost_customer",
        "hist_cost_policy",
        "household_size",
        "marital_status",
        "marketing_consent",
        "marketing_segment",
        "mortgage",
        "new_business",
        "policyholder_age",
        "previous_insurer",
        "property_type",
        "region",
        "roof_type",
        "smoke_alarms",
        "solar_panels",
        "trees_within_5m",
        "urban_density",
    ],
    "contents": [
        "basement",
        "branch_id",
        "campaign_code",
        "compulsory_excess",
        "customer_tenure",
        "days_quote_to_inception",
        "distance_to_river_km",
        "document_preference",
        "dry_summer_index",
        "hail_days",
        "hist_cost_customer",
        "hist_cost_policy",
        "hist_n_customer",
        "hist_n_policy",
        "household_size",
        "listed_building",
        "marital_status",
        "marketing_consent",
        "marketing_segment",
        "mean_temperature_c",
        "new_business",
        "occupation",
        "previous_insurer",
        "property_age",
        "renewal_count",
        "roof_type",
        "soil_type",
        "start_month",
        "wind_exposure_index",
        "years_at_address",
    ],
}


def _feature_cols(df: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    drop = {c for c in df.columns if c.startswith(("cost_", "n_", "ult_", "comp_"))}
    drop |= LINE_DROP.get(str(df["line"].iloc[0]), set())
    # owner (3.30): config `harness.use_premium` (configs/market_dev.yaml), default
    # off, re-admits written_premium on every line for a market where it is not an
    # admissible rating factor concern; see the config comment for the re-test rule.
    if cfg is None:
        from iaa import ultimate

        cfg = ultimate.load_config()
    if cfg.get("harness", {}).get("use_premium", False):
        drop.discard("written_premium")
    drop |= set(PRUNED.get(str(df["line"].iloc[0]), []))
    cols = [c for c in df.columns if c not in NON_FEATURES and c not in drop]
    return [c for c in cols if df[c].notna().any()]


def _credibility_encode(
    tr: pd.DataFrame, va: pd.DataFrame, cols: list[str], target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_card = [
        c
        for c in cols
        if isinstance(tr[c].dtype, pd.CategoricalDtype) and tr[c].nunique() > HIGH_CARD_MIN
    ]
    tr_x, va_x = tr[cols].copy(), va[cols].copy()
    if not high_card:
        return tr_x, va_x
    # fit_transform cross-fits internally so a row's own target doesn't leak into its
    # own encoding; transform() on valid reuses the mapping fit on all of tr. Seeded so
    # runs reproduce (unseeded folds moved the primary score by up to 0.0007).
    enc = TargetEncoder(target_type="continuous", smooth="auto", random_state=SEED_OFFSET)
    tr_x[high_card] = enc.fit_transform(tr[high_card], target)
    va_x[high_card] = enc.transform(va[high_card])
    return tr_x, va_x


# owner 2026-10-19 (Karl, Q20 / backlog 3.37): one open 11.9m bodily-injury claim (AY2023)
# would otherwise load the BI level as if a claim that size came every four years. Rows are
# capped at BI_CAP and the expected excess is loaded from a Pareto tail fitted to the
# training rows above BI_TAIL (refitted per fold, so no look-ahead).
BI_CAP, BI_TAIL = 1_000_000.0, 250_000.0


@lru_cache(maxsize=1)
def _bi_fs_ratio() -> dict[int, float]:
    """Frequency-severity ultimate / chain-ladder ultimate per BI loss year, without the
    max(1, .) floor that ultimate.freq_sev_adjustment applies (owner 2026-10-20, Karl:
    taking the larger of two estimates biased BI up 0.5-2.4% in 2022-23)."""
    from iaa import data, ultimate

    cfg = ultimate.load_config()
    out: dict[int, float] = {}
    spec = (cfg["development"].get("freq_sev") or {}).get("motor.bodily_injury")
    if not spec:
        return out
    clm = data.claims()
    perils = cfg["development"]["peril_groups"]["motor"]["bodily_injury"]
    c = clm[clm["peril"].isin(perils) & (clm["line"] == "motor") & (clm["status"] != "repudiated")]
    base = c[c["ay"] == spec["base_year"]]["net"].mean()
    from iaa import target

    rows, _, _ = target.reported_targets("motor")
    ccdf = ultimate.completion_by_month(ultimate.delay_cdfs(clm, cfg, value="count"), cfg)
    acdf = ultimate.completion_by_month(ultimate.delay_cdfs(clm, cfg, value="amt"), cfg)
    cc = ultimate.row_completion(rows, ccdf, "motor", "bodily_injury")
    ac = ultimate.row_completion(rows, acdf, "motor", "bodily_injury")
    n = rows[[f"n_{q}" for q in perils]].sum(axis=1).to_numpy()
    cost = rows[[f"cost_{q}" for q in perils]].sum(axis=1).to_numpy()
    for yr in sorted(rows["year"].unique()):
        k = (rows["year"] == yr).to_numpy()
        sev = base * (1 + spec["sev_trend"]) ** (yr - spec["base_year"])
        r = (n[k] / cc[k]).sum() * sev / (cost[k] / ac[k]).sum()
        out[int(yr)] = r if yr > spec["base_year"] else 1.0
    return out


def _bi_level(tr: pd.DataFrame) -> float:
    # remove the upward floor: years where frequency x severity is below the chain ladder
    # take the lower estimate (the developed ult already carries max(1, r))
    fix = tr["year"].map({y: min(r, 1.0) for y, r in _bi_fs_ratio().items()}).fillna(1.0)
    ult, trend = tr["ult_bodily_injury"].fillna(0.0) * fix, tr["trend_factor"]
    capped = (ult.clip(upper=BI_CAP) * trend).sum()
    tail = ult[ult > BI_TAIL]
    excess = (ult - BI_CAP).clip(lower=0.0)
    if len(tail) >= 10:
        alpha = len(tail) / np.log(tail / BI_TAIL).sum()
        per_claim = (BI_TAIL / BI_CAP) ** alpha * BI_CAP / (alpha - 1) if alpha > 1.05 else np.nan
        load = len(tail) * per_claim if np.isfinite(per_claim) else excess.sum()
    else:
        load = excess.sum()  # too few large rows to fit a tail: keep the realised excess
    return (capped + load * np.average(trend, weights=ult.clip(upper=BI_CAP) + 1e-9)) / tr[
        "exposure"
    ].sum()


def _anchor(rate: np.ndarray, target_lc: float, va: pd.DataFrame) -> np.ndarray:
    # level anchoring: the model supplies relativities; the overall loss cost is the
    # on-levelled historical loss cost (trend assumption in the config)
    return rate * target_lc / np.average(rate, weights=va["exposure"])


# owner 2026-10-19 (Karl, meta-review): motor is where the gap is, and every motor experiment
# was paying for a full property refit. Lines listed here reuse the saved validation
# predictions of the run to beat (harness.best_kept) instead of refitting; the final 2025
# fit always refits every line. To experiment on property, set FROZEN_LINES = () in the
# same change.
# Part 2 (emerging): every line refits; the book is small and property is in scope.
FROZEN_LINES: tuple[str, ...] = (
    ("buildings", "contents") if os.environ.get("IAA_MARKET", "developed") == "developed" else ()
)


def _frozen_preds(valid: pd.DataFrame) -> pd.Series | None:
    """Saved predictions of the run to beat for these validation rows, or None if there
    are none (the 2025 final fit, or a fold the run did not score)."""
    from iaa import harness

    best = harness.best_kept()
    if not FROZEN_LINES or best is None or not best.get("run_id"):
        return None
    path = harness.PREDS / f"{best['run_id']}.parquet"
    if not path.exists():
        return None
    saved = pd.read_parquet(path, columns=["row_id", "line", "pred"])
    saved = saved[saved["line"].isin(FROZEN_LINES)].drop_duplicates("row_id")
    rows = valid.loc[valid["line"].isin(FROZEN_LINES), "row_id"].astype(str)
    hit = saved.set_index(saved["row_id"].astype(str))["pred"].reindex(rows)
    return None if hit.isna().any() else hit


def _fit_predict_core(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(valid))
    frozen = _frozen_preds(valid)
    for line in ("motor", "buildings", "contents"):
        tr = train[train["line"] == line]
        mask = (valid["line"] == line).to_numpy()
        va = valid[mask]
        if frozen is not None and line in FROZEN_LINES:
            pred[mask] = frozen.reindex(va["row_id"].astype(str)).to_numpy()
            continue
        if line == "motor":
            # backlog 2.63 refinement: NCB transferred onto a brand-new policy is
            # unverified (quoted by the customer, not earned on this book), unlike
            # renewal NCB, so give it its own monotone column instead of sharing one
            # `-1` curve across both; renewals and zero-NCB new business both get NaN
            # here, which LightGBM/EBM split on cleanly as a separate missing bin.
            is_new = tr["new_business"].astype(str) == "True"
            tr = tr.assign(ncb_years_new=tr["ncb_years"].where(is_new))
            is_new_va = va["new_business"].astype(str) == "True"
            va = va.assign(ncb_years_new=va["ncb_years"].where(is_new_va))
        if line == "motor" and "credit_band" in tr.columns:
            # owner 2026-10-20 (Karl, option 2b): motor credit band as an ordered code with a
            # monotone increasing constraint, so the relativity cannot wiggle between bands
            order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            tr = tr.assign(credit_band=tr["credit_band"].astype(str).map(order).astype(float))
            va = va.assign(credit_band=va["credit_band"].astype(str).map(order).astype(float))
        cols = _feature_cols(tr)
        mono = [
            ({**MONOTONE, "credit_band": 1} if line == "motor" else MONOTONE).get(c, 0)
            for c in cols
        ]
        tr_x, va_x = _credibility_encode(tr, va, cols, tr["y"] / tr["exposure"])

        # seed bagging: single-seed noise (sd 0.0026 on the primary metric) is larger
        # than most real improvements, so every model is an average over SEEDS
        def bag(target: pd.Series, weight: pd.Series, params: dict = PARAMS) -> np.ndarray:
            n_estimators = params["n_estimators"]
            if line != "motor":
                # backlog 2.34 / Tier 0c.3, restricted to property: rows 71 and 98
                # showed early stopping chosen on the pooled target cuts motor's
                # rounds and costs motor fold-B D2 (~-0.001) while raising
                # buildings/contents fold-B D2 (~+0.0024/+0.0025) -- the lines don't
                # share an optimum here either, so apply it to property only.
                last_year = tr["year"].max()
                inner_mask = (tr["year"] < last_year).to_numpy()
                if inner_mask.any() and (~inner_mask).any():
                    probe = lgb.LGBMRegressor(
                        **{**params, "n_estimators": 3000, "seed": SEEDS[0]},
                        monotone_constraints=mono,
                        monotone_constraints_method="advanced",
                    )
                    probe.fit(
                        tr_x[inner_mask],
                        target[inner_mask],
                        sample_weight=weight[inner_mask],
                        eval_set=[(tr_x[~inner_mask], target[~inner_mask])],
                        eval_sample_weight=[weight[~inner_mask]],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    n_estimators = max(100, int(probe.best_iteration_ * 1.1))
            out = np.zeros(len(va))
            for seed in SEEDS:
                model = lgb.LGBMRegressor(
                    **{**params, "n_estimators": n_estimators, "seed": seed},
                    monotone_constraints=mono,
                    monotone_constraints_method="advanced",
                )
                model.fit(tr_x, target, sample_weight=weight)
                out += model.predict(va_x) / len(SEEDS)
            return out

        def ebm_rate(
            target: pd.Series, weight: pd.Series, mono: list[int] | None = None
        ) -> np.ndarray:
            ebm_tr_x, ebm_va_x = _ebm_safe(tr_x), _ebm_safe(va_x)
            model = ExplainableBoostingRegressor(
                **EBM_PARAMS, random_state=SEEDS[0], monotone_constraints=mono
            )
            model.fit(ebm_tr_x, target, sample_weight=weight)
            return model.predict(ebm_va_x)

        def cb_rate(target: pd.Series, weight: pd.Series) -> np.ndarray:
            cat_cols = [c for c in tr_x.columns if isinstance(tr_x[c].dtype, pd.CategoricalDtype)]
            cb_tr_x, cb_va_x = _ebm_safe(tr_x), _ebm_safe(va_x)
            model = CatBoostRegressor(**CATBOOST_PARAMS, random_seed=SEEDS[0])
            model.fit(cb_tr_x, target, sample_weight=weight, cat_features=cat_cols)
            return model.predict(cb_va_x)

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
            bi_lc = _bi_level(tr)
            bi_rate = _anchor(freq, bi_lc, va)

            other = tr["ult_other"] * tr["trend_factor"]
            other_target = other / tr["exposure"]
            other_rate = 0.5 * bag(other_target, tr["exposure"]) + 0.5 * ebm_rate(
                other_target, tr["exposure"], mono
            )
            other_lc = other.sum() / tr["exposure"].sum()
            other_rate = _anchor(other_rate, other_lc, va)

            rate = bi_rate + other_rate
        else:
            rate_target = tr["y"] / tr["exposure"]
            rate = (
                bag(rate_target, tr["exposure"])
                + ebm_rate(rate_target, tr["exposure"])
                + cb_rate(rate_target, tr["exposure"])
            ) / 3
            rate = _anchor(rate, tr["y"].sum() / tr["exposure"].sum(), va)
        pred[mask] = rate * va["exposure"].to_numpy()
    return pred


# Owner data fix (Karl 2026-10-20): buildings construction_type is coded T/t in 2021-22 and
# W/w from 2023 (the same 11.5% / 0.6% of exposure: one category recoded between extracts),
# and carries case variants (BRICK/Brick/brick, stone/STONE, non_standard/NON_STANDARD).
# Harmonised to one label per category so each relativity uses all years.
def _fix_construction(df: pd.DataFrame) -> pd.Series:
    s = df["construction_type"].astype("string").replace({"T": "W", "t": "w"})
    return s.str.lower()


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    if "construction_type" in train.columns:
        a, b = _fix_construction(train), _fix_construction(valid)
        cats = sorted(set(a.dropna()) | set(b.dropna()))
        train = train.assign(construction_type=pd.Categorical(a, categories=cats))
        valid = valid.assign(construction_type=pd.Categorical(b, categories=cats))
    return _fit_predict_core(train, valid)
