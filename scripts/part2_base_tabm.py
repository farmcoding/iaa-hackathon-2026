"""TabM developed-market base for the emerging rows (tier N15, owner, 27 Oct; GPU).

A second model class next to the LightGBM/EBM/CatBoost bases: TabM (an ensemble MLP)
fitted per line on developed rows with the Tweedie (p=1.5) rate loss of scripts/tabm_fit.py,
then scoring every emerging row in the developed schema (scripts/part2_transfer.align).
In Part 1 a TabM blend added +0.0039 on the public board. Neural nets extrapolate
differently from trees on emerging values outside the developed range (hail, wildfire).

Preprocessing differs from tabm_fit.prep in one way: missing-value indicators and
medians come from the training rows only, so emerging rows (credit band, mileage and
telematics absent) get the training layout instead of extra columns.

    uv run python scripts/part2_base_tabm.py EMERGING_CACHE --tag B --train-years 2021 2022 2023

Writes data/cache/part2_base_tabm_<tag>.parquet: market, row_id, line, base_tabm (seed
average). Run under the developed market.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iaa import data, harness, ultimate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabm_fit  # noqa: E402
from part2_transfer import align  # noqa: E402

assert data.MARKET == "developed", "run with the developed market"


def prep(d: pd.DataFrame, ref: pd.DataFrame, nums: list[str], cats: list[str]):
    cols = []
    for c in nums:
        r = ref[c].astype(float)
        med = float(np.nanmedian(r))
        lo, hi = np.nanquantile(r, [0.01, 0.99])
        x = d[c].astype(float).to_numpy()
        isna = np.isnan(x)
        cols.append((np.clip(np.where(isna, med, x), lo, hi) - med) / (hi - lo + 1e-9))
        if ref[c].isna().any():  # layout from the training rows only
            cols.append(isna.astype(float))
    xn = np.stack(cols, 1).astype(np.float32)
    xc = np.stack([d[c].cat.codes.to_numpy() + 1 for c in cats], 1).astype(np.int64)
    return xn, xc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("emerging_cache")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-years", type=int, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--sample", type=int, default=0, help="training rows per line (smoke)")
    a = ap.parse_args()
    tabm_fit.prep = prep
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    cfg = ultimate.load_config()
    dev = harness.dataset()
    tr_all, _ = harness.split(dev, a.train_years, cfg["predict_year"], cfg)
    emg = pd.read_parquet(a.emerging_cache)
    va_all = harness.hide_target(align(emg, dev))
    out = emg[["row_id", "line"]].astype(str).assign(market="emerging", base_tabm=np.nan)
    for line in ("motor", "buildings", "contents"):
        tr = tr_all[tr_all["line"] == line]
        if a.sample:
            tr = tr.sample(min(a.sample, len(tr)), random_state=0)
        rows = (va_all["line"].astype(str) == line).to_numpy()
        va = va_all[rows].reset_index(drop=True)
        preds = []
        for s in a.seeds:
            tabm_fit.SEED = s
            t0, log = time.time(), {}
            preds.append(tabm_fit.fit_line(tr, va, a.epochs, device, log))
            print(
                f"{line} seed {s}: {len(log['epochs'])} epochs, {time.time() - t0:.0f}s", flush=True
            )
        out.loc[rows, "base_tabm"] = np.mean(preds, axis=0)
    path = harness.CACHE / f"part2_base_tabm{'_sample' if a.sample else ''}_{a.tag}.parquet"
    out.to_parquet(path)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
