"""Test-mix importance weights for the emerging validation years (Part 2 validation design).

The 2025 book is a broker transfer (76% new business, younger) unlike either training
year. Scoring a fold only on the natural 2024 book rewards what works for the old mix.
For each line and each training year, a classifier separates that year's rows from the
2025 test rows on policy, customer, vehicle and property characteristics; weather and
year-derived fields are excluded (they identify the year, not the book). The odds
p / (1 - p) reweight the year's rows towards the 2025 mix; weights are clipped at 10x the
mean and normalised to mean 1 per line and year. Effective sample size is reported.

    IAA_MARKET=emerging uv run python scripts/part2_weights.py

Writes data/cache/part2_testmix_weights.parquet (row_id, line, year, w) and prints ESS.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from iaa import data, harness

assert data.MARKET == "emerging"

EXCLUDE = {
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
    "written_premium",
    "start_month",
    "hail_days",
    "wind_exposure_index",
    "freezing_days",
    "dry_summer_index",
    "wildfire_index",
    "rainfall_mm",
    "mean_temperature_c",
    "property_age",
    "customer_tenure",
    "hist_n_policy",
    "hist_cost_policy",
    "hist_n_customer",
    "hist_cost_customer",
    "lines_held",
    "customer_age",
    "driver_age",
    "policyholder_age",
}
TARGETISH = ("cost_", "n_", "ult_", "comp_", "y_ult")


def main() -> None:
    df = harness.dataset()
    out = []
    for line in data.LINES:
        d = df[df["line"] == line]
        cols = [
            c
            for c in d.columns
            if c not in EXCLUDE and not c.startswith(TARGETISH) and d[c].notna().any()
        ]
        test = d[d["split"] == "test"]
        for yr in (2023, 2024):
            hist = d[(d["split"] == "train") & (d["year"] == yr)]
            x = pd.concat([hist[cols], test[cols]], ignore_index=True)
            y = np.r_[np.zeros(len(hist)), np.ones(len(test))]
            p = np.zeros(len(hist))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
                m = lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=200,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=0,
                    verbose=-1,
                )
                m.fit(x.iloc[tr], y[tr])
                hv = va[va < len(hist)]
                p[hv] = m.predict_proba(x.iloc[hv])[:, 1]
            odds = p / np.clip(1 - p, 1e-6, None)
            w = np.clip(odds / odds.mean(), 0, 10)
            w = w / w.mean()
            ess = w.sum() ** 2 / (w**2).sum()
            nb = (hist["new_business"].astype(str) == "True").to_numpy()
            print(
                f"{line} {yr}: rows {len(hist)}, ESS {ess:.0f} ({ess / len(hist):.0%}), "
                f"new-business share natural {nb.mean():.0%} -> weighted "
                f"{np.average(nb, weights=w):.0%} (2025: "
                f"{(test['new_business'].astype(str) == 'True').mean():.0%})"
            )
            out.append(
                pd.DataFrame(
                    {
                        "row_id": hist["row_id"].astype(str).to_numpy(),
                        "line": line,
                        "year": yr,
                        "w": w,
                    }
                )
            )
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(harness.CACHE / "part2_testmix_weights.parquet")
    print("wrote", harness.CACHE / "part2_testmix_weights.parquet", len(res))


if __name__ == "__main__":
    main()
