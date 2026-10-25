"""Part 2 drift report: what changed between the developed and the emerging market.

Five checks, each feeding a decision in configs/market_emerging.yaml or the model:
1. schema: columns and category levels present in one market only, missing-rate shifts;
2. population shift: PSI per feature (developed train -> emerging train, and emerging
   train -> emerging 2025 test, the broker book);
3. adversarial validation: can a classifier tell the two samples apart, and on what;
4. experience at the same maturity: both markets are valued 31 Dec 2024, so loss years
   2023 and 2024 are at identical development ages in both, and report-delay shares,
   open shares, frequency and severity compare like for like;
5. climate: weather distributions by year.

    uv run python scripts/part2_drift.py DEV_CACHE EMG_CACHE [--out reports/part2_drift]

DEV_CACHE / EMG_CACHE are the harness dataset parquet files for each market
(IAA_MARKET=developed|emerging, harness.dataset()). Writes <out>.md and <out>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from iaa import data, experiment

NOT_FEATURES = {
    "row_id",
    "policy_id",
    "customer_id",
    "address_id",
    "period_start",
    "period_end",
    "split",
    "line",
    "year",
    "exposure",
}
TENURE = [
    "new_business",
    "renewal_count",
    "customer_tenure",
    "ncb_years",
    "hist_n_policy",
    "hist_cost_policy",
    "hist_n_customer",
    "hist_cost_customer",
    "lines_held",
]
WEATHER = [
    "hail_days",
    "wind_exposure_index",
    "freezing_days",
    "dry_summer_index",
    "wildfire_index",
    "rainfall_mm",
    "mean_temperature_c",
]
AGES = ["customer_age", "driver_age", "policyholder_age"]
# derived from the row's year, so with a fixed field they reveal the year itself
YEARISH = ["property_age", "customer_tenure"]
TARGETISH = ("n_", "cost_", "ult_", "comp_", "y_ult")
LINE_COLS = {
    "motor": [
        "years_licensed",
        "ncb_years",
        "vehicle_value_group",
        "vehicle_age",
        "make_model",
        "engine_size_cc",
        "body_type",
        "fuel_type",
        "driver_assist",
        "cover_type",
        "overnight_parking",
        "vehicle_use",
        "named_young_driver",
        "driver_age",
    ],
    "buildings": [
        "sum_insured",
        "construction_type",
        "occupancy",
        "days_unoccupied",
        "smoke_alarms",
        "accidental_damage_cover",
        "mortgage",
        "policyholder_age",
    ],
    "contents": [
        "sum_insured",
        "tenure",
        "security_level",
        "high_value_items",
        "accidental_damage_cover",
        "works_from_home",
        "lodgers",
        "bicycle_cover",
        "policyholder_age",
    ],
}
COMMON = [
    "voluntary_excess",
    "channel",
    "payment_method",
    "missed_payments_12m",
    "new_business",
    "renewal_count",
    "days_quote_to_inception",
    "discount_applied",
    "occupation",
    "marital_status",
    "household_size",
    "prior_claims_5y",
    "licence_points",
    "years_at_address",
    "previous_insurer",
    "urban_density",
    "flood_zone",
    "distance_to_river_km",
    "elevation_m",
    "soil_type",
    "year_built",
    "property_type",
    "roof_type",
    "basement",
    "trees_within_5m",
    "bedrooms",
    "energy_rating",
    "hail_days",
    "wind_exposure_index",
    "freezing_days",
    "dry_summer_index",
    "wildfire_index",
    "rainfall_mm",
    "mean_temperature_c",
    "customer_age",
    "customer_tenure",
    "property_age",
    "hist_n_policy",
    "hist_n_customer",
    "lines_held",
    "written_premium",
]


def psi(a: pd.Series, b: pd.Series, wa: np.ndarray, wb: np.ndarray, bins: int = 10) -> float:
    """Exposure-weighted population stability index of b against a (missing is a bin)."""
    if pd.api.types.is_numeric_dtype(a) and a.nunique() > bins:
        edges = np.unique(np.nanquantile(a.astype(float), np.linspace(0, 1, bins + 1)))
        ka = pd.cut(a.astype(float), edges, include_lowest=True).astype(str)
        kb = pd.cut(b.astype(float).clip(edges[0], edges[-1]), edges, include_lowest=True).astype(
            str
        )
    else:
        # numeric codes compare as numbers: an int column read back as float after a
        # join ("0" vs "0.0") must not land in different bins
        num = pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a)
        ka, kb = (
            (a.astype(float).astype(str), b.astype(float).astype(str))
            if num
            else (a.astype(str), b.astype(str))
        )
    pa = pd.Series(wa, index=a.index).groupby(ka.to_numpy()).sum() / wa.sum()
    pb = pd.Series(wb, index=b.index).groupby(kb.to_numpy()).sum() / wb.sum()
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx).fillna(0) + 1e-4, pb.reindex(idx).fillna(0) + 1e-4
    return float(((pb - pa) * np.log(pb / pa)).sum())


def adversarial(a: pd.DataFrame, b: pd.DataFrame, cols: list[str], seed: int = 0) -> dict:
    """Out-of-fold AUC of a classifier separating a (0) from b (1), plus top gain features."""
    x = pd.concat([a[cols], b[cols]], ignore_index=True)
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_bool_dtype(x[c]):
            x[c] = x[c].astype("string").astype("category")
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    oof, gain = np.zeros(len(y)), pd.Series(0.0, index=cols)
    params = dict(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.5,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        importance_type="gain",
    )
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y):
        m = lgb.LGBMClassifier(**params).fit(x.iloc[tr], y[tr])
        oof[va] = m.predict_proba(x.iloc[va])[:, 1]
        gain += pd.Series(m.booster_.feature_importance("gain"), index=cols)
    share = (gain / gain.sum()).sort_values(ascending=False)
    return {
        "auc": round(float(roc_auc_score(y, oof)), 4),
        "top": {k: round(float(v), 3) for k, v in share.head(8).items()},
    }


def same_maturity(
    dev_c: pd.DataFrame, emg_c: pd.DataFrame, dev: pd.DataFrame, emg: pd.DataFrame
) -> pd.DataFrame:
    """Per line, peril and loss year: frequency, severity, open share and the share of
    claims reported within 1 and 3 months, both markets at 31 Dec 2024."""
    rows = []
    for mkt, c, pol in (("developed", dev_c, dev), ("emerging", emg_c, emg)):
        c = c[(c["status"] != "repudiated") & c["ay"].isin([2023, 2024])].copy()
        c["delay_m"] = c["report_lag_days"] / 30.44
        expo = pol[pol["split"] == "train"].groupby(["line", "year"])["exposure"].sum()
        for (line, peril, ay), g in c.groupby(["line", "peril", "ay"]):
            e = expo.get((line, ay), np.nan)
            rows.append(
                {
                    "market": mkt,
                    "line": line,
                    "peril": peril,
                    "ay": ay,
                    "n": len(g),
                    "freq_per_1000": 1000 * len(g) / e,
                    "sev": g["net"].mean(),
                    "open": (g["status"] == "open").mean(),
                    "rep_1m": (g["delay_m"] <= 1).mean(),
                    "rep_3m": (g["delay_m"] <= 3).mean(),
                }
            )
    t = pd.DataFrame(rows).set_index(["line", "peril", "ay", "market"]).unstack("market")
    for k in ("freq_per_1000", "sev"):
        t[(k, "ratio")] = t[(k, "emerging")] / t[(k, "developed")]
    return t.sort_index(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dev_cache")
    ap.add_argument("emg_cache")
    ap.add_argument("--out", default=str(data.ROOT / "reports" / "part2_drift"))
    a = ap.parse_args()
    dev, emg = pd.read_parquet(a.dev_cache), pd.read_parquet(a.emg_cache)
    for df in (dev, emg):  # the model's own label clean-up (Part 1 owner fix)
        df["construction_type"] = experiment._fix_construction(df)
    dev_tr = dev[dev["split"] == "train"]
    emg_tr, emg_te = emg[emg["split"] == "train"], emg[emg["split"] == "test"]
    out: dict = {
        "rows": {"dev_train": len(dev_tr), "emg_train": len(emg_tr), "emg_test": len(emg_te)}
    }

    # 1. schema
    dcols, ecols = set(dev.columns), set(emg.columns)
    out["schema"] = {
        "developed_only": sorted(dcols - ecols),
        "emerging_only": sorted(ecols - dcols),
    }
    miss = {}
    for c in sorted(dcols & ecols - NOT_FEATURES):
        if c.startswith(TARGETISH):
            continue
        md, me = dev_tr[c].isna().mean(), emg_tr[c].isna().mean()
        if abs(me - md) > 0.02:
            miss[c] = [round(md, 3), round(me, 3)]
    out["schema"]["missing_rate_dev_vs_emg"] = miss

    # 2. PSI and 3. adversarial validation, per line
    out["psi"], out["adversarial"] = {}, {}
    for line in data.LINES:
        cols = [c for c in COMMON + LINE_COLS[line] if c in dcols & ecols]
        d, e, t = (x[x["line"] == line] for x in (dev_tr, emg_tr, emg_te))
        d = d.sample(min(len(d), 200_000), random_state=0)
        out["psi"][line] = {
            c: {
                "dev_to_emg": round(
                    psi(d[c], e[c], d["exposure"].to_numpy(), e["exposure"].to_numpy()), 3
                ),
                "emg_to_test": round(
                    psi(e[c], t[c], e["exposure"].to_numpy(), t["exposure"].to_numpy()), 3
                ),
            }
            for c in cols
        }
        no_ten = [c for c in cols if c not in TENURE and c != "written_premium"]
        # weather is year-specific (2025 vs 2023-24) and ages are banded in emerging, so
        # both separate the samples trivially; this variant shows the remaining mix shift
        core = [c for c in no_ten if c not in WEATHER + AGES + YEARISH]
        out["adversarial"][line] = {
            "dev_vs_emg_train": adversarial(
                d.sample(min(len(d), len(e) * 3), random_state=1), e, no_ten
            ),
            "emg_train_vs_test": adversarial(e, t, cols),
            "emg_train_vs_test_no_tenure": adversarial(e, t, no_ten),
            "dev_vs_emg_core": adversarial(
                d.sample(min(len(d), len(e) * 3), random_state=1), e, core
            ),
            "emg_train_vs_test_core": adversarial(e, t, core),
            # within the test book: renewing customers vs the broker's new business
            "test_renewal_vs_new": adversarial(
                t[t["new_business"].astype(str) == "False"],
                t[t["new_business"].astype(str) == "True"],
                [c for c in no_ten if c != "new_business"],
            ),
        }
        print(line, json.dumps(out["adversarial"][line]), flush=True)

    # make_model catalogue overlap (exposure share whose code was seen in the other sample)
    mm = {}
    dm, em, tm = (x[x["line"] == "motor"] for x in (dev_tr, emg_tr, emg_te))
    for name, a_, b_ in (
        ("emg_train_seen_in_dev", em, dm),
        ("emg_test_seen_in_emg_train", tm, em),
        ("emg_test_seen_in_dev", tm, dm),
    ):
        seen = a_["make_model"].astype(str).isin(set(b_["make_model"].astype(str)))
        mm[name] = round(float(a_.loc[seen, "exposure"].sum() / a_["exposure"].sum()), 4)
    mm["levels"] = {
        k: int(v["make_model"].nunique())
        for k, v in (("dev", dm), ("emg_train", em), ("emg_test", tm))
    }
    out["make_model_overlap"] = mm
    print("make_model", mm, flush=True)

    # 4. experience at the same maturity
    sm = same_maturity(data.claims("developed"), data.claims("emerging"), dev, emg)
    out["same_maturity"] = json.loads(sm.round(4).to_json(orient="index"))

    # 5. climate
    w = {
        m: pd.read_parquet(data.MARKET_DIRS[m] / "weather.parquet")
        for m in ("developed", "emerging")
    }
    wcols = [
        "hail_days",
        "wind_exposure_index",
        "freezing_days",
        "dry_summer_index",
        "wildfire_index",
        "rainfall_mm",
        "mean_temperature_c",
    ]
    out["climate"] = {m: w[m].groupby("year")[wcols].mean().round(3).to_dict("index") for m in w}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    write_md(out, sm, Path(a.out + ".md"))
    print("wrote", a.out + ".md / .json")


def write_md(out: dict, sm: pd.DataFrame, path: Path) -> None:
    L = [
        "# Part 2 drift report: developed -> emerging",
        "",
        "Generated by `scripts/part2_drift.py`. Numbers only; the decisions they feed are in "
        "`configs/market_emerging.yaml` and `reports/assumptions_log.md`.",
        "",
        f"Rows: {out['rows']}",
        "",
        "## 1. Schema",
        "",
        f"- Developed only: {', '.join(out['schema']['developed_only'])}",
        f"- Emerging only: {', '.join(out['schema']['emerging_only'])}",
        "- Missing rate, developed vs emerging train (shift > 2 points): "
        + ", ".join(
            f"{k} {v[0]:.1%} -> {v[1]:.1%}"
            for k, v in out["schema"]["missing_rate_dev_vs_emg"].items()
        ),
        "",
        "## 2. Population shift (PSI, exposure-weighted; > 0.25 is a major shift)",
        "",
    ]
    for line, p in out["psi"].items():
        top = sorted(p.items(), key=lambda kv: -max(kv[1].values()))[:10]
        L += [
            f"**{line}**",
            "",
            "| feature | developed -> emerging | emerging train -> 2025 test |",
            "|---|---|---|",
        ]
        L += [f"| {k} | {v['dev_to_emg']:.3f} | {v['emg_to_test']:.3f} |" for k, v in top]
        L.append("")
    L += [
        "make_model catalogue (motor exposure share whose code appears in the other sample): "
        + json.dumps(out["make_model_overlap"]),
        "",
        "## 3. Adversarial validation (out-of-fold AUC; 0.5 = indistinguishable)",
        "",
        "| line | test | AUC | top features (gain share) |",
        "|---|---|---|---|",
    ]
    for line, r in out["adversarial"].items():
        for k, v in r.items():
            L.append(
                f"| {line} | {k} | {v['auc']:.3f} | "
                + ", ".join(f"{a} {b:.2f}" for a, b in list(v["top"].items())[:5])
                + " |"
            )
    L += [
        "",
        "## 4. Experience at the same maturity (both markets valued 31 Dec 2024)",
        "",
        "| line | peril | AY | n dev / emg | freq per 1000, dev / emg (ratio) | severity dev / emg (ratio) | open dev / emg | reported within 3 m dev / emg |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (line, peril, ay), r in sm.iterrows():
        L.append(
            f"| {line} | {peril} | {ay} | {r[('n', 'developed')]:.0f} / {r[('n', 'emerging')]:.0f} | "
            f"{r[('freq_per_1000', 'developed')]:.1f} / {r[('freq_per_1000', 'emerging')]:.1f} ({r[('freq_per_1000', 'ratio')]:.2f}) | "
            f"{r[('sev', 'developed')]:.0f} / {r[('sev', 'emerging')]:.0f} ({r[('sev', 'ratio')]:.2f}) | "
            f"{r[('open', 'developed')]:.2f} / {r[('open', 'emerging')]:.2f} | {r[('rep_3m', 'developed')]:.2f} / {r[('rep_3m', 'emerging')]:.2f} |"
        )
    L += [
        "",
        "## 5. Climate (mean per address-year)",
        "",
        "| market | year | "
        + " | ".join(next(iter(out["climate"]["developed"].values())).keys())
        + " |",
        "|---|---|" + "---|" * 7,
    ]
    for m, yrs in out["climate"].items():
        for y, v in yrs.items():
            L.append(f"| {m} | {y} | " + " | ".join(f"{x:.2f}" for x in v.values()) + " |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
