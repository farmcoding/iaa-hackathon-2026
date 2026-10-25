"""Part 2 baseline: one trended loss cost per exposure per line, emerging market.

Not a pricing candidate. It checks the submission plumbing and sets the floor every
emerging model must beat. Emerging claims (valued 31 Dec 2024, two loss years) are
developed with the developed market's report-delay patterns by line and peril group,
because two emerging years cannot support a tail. Each year is trended to 2025 at the
developed-market rates. Every test row gets line rate x exposure.

    uv run python scripts/part2_baseline.py [--out submissions/part2/baseline_rate.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, target, ultimate

ROOT = Path(__file__).resolve().parents[1]


def emerging_rows() -> pd.DataFrame:
    pol = pd.concat(
        [data.policies(ln, "train", "emerging") for ln in data.LINES], ignore_index=True
    )
    hits, stats = target.match_claims(pol, data.claims("emerging"))
    print("claim match:", stats)
    agg = hits.pivot_table(
        index="row_id", columns="peril", values="net", aggfunc="sum", fill_value=0
    )
    agg.columns = [f"cost_{p}" for p in agg.columns]
    rows = pol.merge(agg, left_on="row_id", right_index=True, how="left")
    rows[list(agg.columns)] = rows[list(agg.columns)].fillna(0.0)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submissions" / "part2" / "baseline_rate.csv"))
    args = ap.parse_args()

    cfg = ultimate.load_config()  # developed assumptions: peril groups, trends, max delay
    comp = ultimate.completion_by_month(ultimate.delay_cdfs(data.claims("developed"), cfg), cfg)
    rows = emerging_rows()
    trend, peril_trend = cfg["trend"], cfg.get("peril_trend", {})

    rates = {}
    for line, groups in cfg["development"]["peril_groups"].items():
        r = rows[rows["line"] == line]
        ult = np.zeros(len(r))
        for grp, perils in groups.items():
            cols = [f"cost_{p}" for p in perils if f"cost_{p}" in r.columns]
            t = peril_trend.get(line, {}).get(grp, trend[line])
            on_level = (1 + t) ** (2025 - r["year"].to_numpy())
            ult += (
                r[cols].sum(axis=1).to_numpy()
                / ultimate.row_completion(r, comp, line, grp)
                * on_level
            )
        by_year = pd.DataFrame({"y": r["year"], "ult": ult, "e": r["exposure"]}).groupby("y").sum()
        rates[line] = ult.sum() / r["exposure"].sum()
        print(
            line,
            "trended ultimate per exposure by year:",
            (by_year["ult"] / by_year["e"]).round(1).to_dict(),
            "pooled:",
            round(rates[line], 1),
        )

    sub = []
    for line in data.LINES:
        t = data.policies(line, "test", "emerging")
        sub.append(pd.DataFrame({"row_id": t["row_id"], "prediction": rates[line] * t["exposure"]}))
    sub = pd.concat(sub, ignore_index=True)
    sub["prediction"] = sub["prediction"].clip(lower=1.0)  # D2 is undefined at zero
    sample = pd.read_parquet(data.MARKET_DIRS["emerging"] / "sample_submission.parquet")
    assert len(sub) == len(sample) and set(sub["row_id"]) == set(sample["row_id"].astype("string"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {args.out}: {len(sub)} rows, mean {sub.prediction.mean():.2f}")


if __name__ == "__main__":
    main()
