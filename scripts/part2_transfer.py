"""Part 2 transfer test: the Part 1 model, fitted on the developed market, scored on emerging.

Answers "do the Part 1 relativities hold in the new market?" before anything is changed.
The Part 1 final specification is fitted on developed 2021-24 and predicts emerging rows
(mapped to the developed vocabulary by features.harmonize_*). Emerging-only columns
(gender, age bands, self_employed) are dropped; developed-only columns (credit_band,
annual_mileage, telematics) are left missing. The level is set per line from emerging
history, so only the relativities are tested.

Outputs
- level-neutral D2 by line on emerging 2023 and 2024 (each line rescaled to its actual
  total, so the number measures ranking and spread, not level);
- with --submit-file, a 2025 file with each line at the baseline emerging level
  (scripts/part2_baseline.py) for a like-for-like leaderboard comparison.

    uv run python scripts/part2_transfer.py data/cache/dataset_<emerging fp>.parquet \
        [--submit-file submissions/part2/transfer_p1.csv]

Run under the default (developed) market: the training side reads the developed cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, experiment, harness, score, ultimate

assert data.MARKET == "developed", "run with the developed market; pass the emerging cache"


def align(emg: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=emg.index)
    for c in dev.columns:
        if c not in emg.columns:
            out[c] = pd.Series(np.nan, index=emg.index).astype(dev[c].dtype)
        elif isinstance(dev[c].dtype, pd.CategoricalDtype):
            out[c] = pd.Categorical(emg[c].astype("string"), categories=dev[c].cat.categories)
        else:
            out[c] = emg[c].astype(dev[c].dtype) if dev[c].dtype != object else emg[c]
    return out


def level_neutral(df: pd.DataFrame) -> dict[str, float]:
    df = df.copy()
    for line, g in df.groupby("line"):
        df.loc[g.index, "pred"] = g["pred"] * g["y"].sum() / g["pred"].sum()
    card = score.scorecard(df)
    return {k: round(v, 5) for k, v in card.items() if k.startswith("d2")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("emerging_cache")
    ap.add_argument("--submit-file")
    ap.add_argument("--native-preds", help="saved emerging fold-B predictions to compare")
    a = ap.parse_args()

    cfg = ultimate.load_config()
    dev = harness.dataset()
    tr, _ = harness.split(dev, cfg["harness"]["final"]["train"], cfg["predict_year"], cfg)
    emg = pd.read_parquet(a.emerging_cache)
    va = align(emg, dev)
    experiment.FROZEN_LINES = ()
    pred = harness.check_predictions(experiment.fit_predict(tr, harness.hide_target(va)), len(va))
    out = emg[["row_id", "line", "year", "split", "exposure", "y_ult"]].assign(pred=pred)
    out.to_parquet(Path(a.emerging_cache).with_name("transfer_p1_preds.parquet"))

    for yr in (2023, 2024):
        s = out[(out["split"] == "train") & (out["year"] == yr)].rename(columns={"y_ult": "y"})
        print(f"transfer, emerging {yr}, level-neutral:", level_neutral(s))
    if a.native_preds:
        nat = pd.read_parquet(a.native_preds)
        print("native refit (train 2023), emerging 2024, level-neutral:", level_neutral(nat))
        m = nat.merge(out[["row_id", "pred"]], on="row_id", suffixes=("_nat", ""))
        print("transfer on the same rows:", level_neutral(m.drop(columns="pred_nat")))

    if a.submit_file:
        # same level as the baseline: emerging history trended with the emerging config
        ecfg = ultimate.load_config(data.ROOT / "configs" / "market_emerging.yaml")
        hist, _ = harness.split(emg, ecfg["harness"]["final"]["train"], 2025, ecfg)
        te = out[out["split"] == "test"].copy()
        for line in data.LINES:
            h, t = hist["line"] == line, te["line"] == line
            rate = hist.loc[h, "y"].sum() / hist.loc[h, "exposure"].sum()
            cur = te.loc[t, "pred"].sum() / te.loc[t, "exposure"].sum()
            te.loc[t, "pred"] *= rate / cur
        sub = te[["row_id", "pred"]].rename(columns={"pred": "prediction"})
        sub["prediction"] = sub["prediction"].clip(lower=1.0)
        sub.to_csv(a.submit_file, index=False, float_format="%.4f")
        print(
            "wrote",
            a.submit_file,
            len(sub),
            "rate by line:",
            te.groupby("line").apply(lambda g: g.pred.sum() / g.exposure.sum()).round(1).to_dict(),
        )


if __name__ == "__main__":
    main()
