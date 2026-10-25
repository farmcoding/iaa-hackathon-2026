"""Developed-market base predictions for the emerging experiments.

Fits the Part 1 final specification on developed years and predicts every emerging row
(train and test) plus every developed row. The emerging experiment module
(src/iaa/experiment_emerging.py) uses these as a prior: an offset, a feature or a blend
member. Fitting the developed model takes most of an hour, so it is done once here, not
inside each experiment.

Two fits, so the emerging fold stays out of time:
- tag B: developed 2021-23, used when the experiment scores emerging 2024;
- tag final: developed 2021-24, used for the 2025 file.

    uv run python scripts/part2_base.py EMERGING_CACHE --tag B --train-years 2021 2022 2023
    uv run python scripts/part2_base.py EMERGING_CACHE --tag final --train-years 2021 2022 2023 2024

Writes data/cache/part2_base_<tag>.parquet: market, row_id, line, split, year, exposure,
base (prediction for the row, CU). Run under the developed market (the default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, experiment, harness, ultimate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2_compat import compat  # noqa: E402
from part2_transfer import align  # noqa: E402

assert data.MARKET == "developed", "run with the developed market"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("emerging_cache")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-years", type=int, nargs="+", required=True)
    ap.add_argument("--compat", action="store_true", help="fit on the emerging schema (O2)")
    a = ap.parse_args()

    cfg = ultimate.load_config()
    dev = harness.dataset()
    if a.compat:
        dev = compat(dev)
        a.tag = "compat_" + a.tag
    # train on the chosen developed years, trended to 2025 (level is irrelevant downstream)
    tr, _ = harness.split(dev, a.train_years, cfg["predict_year"], cfg)
    emg = pd.read_parquet(a.emerging_cache)
    keep = ["row_id", "line", "split", "year", "exposure"]
    target = pd.concat(
        [align(emg, dev).assign(market="emerging"), dev.assign(market="developed")],
        ignore_index=True,
    )
    experiment.FROZEN_LINES = ()
    # fit_predict anchors each line's average to one level, so relativities across the
    # stacked frame are preserved; only within-line ratios are used downstream
    pred = harness.check_predictions(
        experiment.fit_predict(tr, harness.hide_target(target.drop(columns="market"))),
        len(target),
    )
    out = target[["market", *keep]].assign(base=pred)
    for col in ("row_id", "line", "split", "market"):
        out[col] = out[col].astype(str)
    path = harness.CACHE / f"part2_base_{a.tag}.parquet"
    out.to_parquet(path)
    rate = out.assign(r=out.base / out.exposure)
    idx = rate.groupby(["market", "line", "split"]).apply(
        lambda g: np.average(g.r, weights=g.exposure), include_groups=False
    )
    print("wrote", path, "\nmean base rate by market/line/split:\n", idx.round(1).to_string())


if __name__ == "__main__":
    main()
