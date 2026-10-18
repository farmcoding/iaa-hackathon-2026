"""Calibration submission: one pooled loss-cost rate x exposure.

Purpose: settle which D² baseline the leaderboard uses (see score.py), not to rank.
Rate = reported net cost 2021-2023 / exposure 2021-2023, all lines pooled.
"""

import json

import numpy as np
import pandas as pd

from iaa import data, score, target

rows, _, stats = target.reported_targets()
print(json.dumps(stats, default=str, indent=1))
hist = rows[rows["year"].between(2021, 2023)]
rate = hist["cost_total"].sum() / hist["exposure"].sum()
print(f"pooled rate 2021-2023: {rate:.2f} CU per exposure-year")

# local reading: same predictor fitted on 2021-2022, scored on 2023 reported
fit = rows[rows["year"].between(2021, 2022)]
val = rows[rows["year"] == 2023].copy()
val["pred"] = val["exposure"] * fit["cost_total"].sum() / fit["exposure"].sum()
val["y"] = val["cost_total"]
for b in ("constant", "rate"):
    print(f"local 2023 D2 ({b} baseline): {score.d2(val.y, val.pred, val.exposure, b):.5f}")

test = data.all_policies("test")
sub = pd.DataFrame(
    {"row_id": test["row_id"], "prediction": np.maximum(test["exposure"] * rate, 1.0)}
)
sample = pd.read_parquet(data.RAW / "sample_submission.parquet")
assert set(sub.row_id) == set(sample.row_id.astype("string")) and len(sub) == len(sample)
sub.to_csv("submissions/calib_pooled_rate.csv", index=False)
print(sub.describe())
