"""Developed-market base predictions by peril group, for the emerging market.

The emerging peril mix differs from the developed one at the same maturity: glass/hail
claims 1.6-1.8x as frequent, storm 1.3-1.7x, escape of water 0.6-0.7x
(reports/part2_drift.md, section 4). A base that predicts total cost carries developed
peril weights into every relativity. With one base prediction per peril group, each group
can take its own emerging level, which moves relativities towards the risks exposed to
the perils that are heavier here, with structure instead of fitted noise.

Peril groups follow the development config: motor bodily_injury / other (the Part 1 motor
model already prices them separately: its two level-anchoring calls are recorded);
buildings weather / water / fire / ad and contents all (the Part 1 property model refitted
once per group, with that group's trended developed cost as the target).

    uv run python scripts/part2_base_perils.py EMERGING_CACHE --tag B --train-years 2021 2022 2023

Writes data/cache/part2_base_perils_<tag>.parquet: market, row_id, line and one column
base_<group> per peril group (NaN on other lines). Run under the developed market.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, experiment, harness, ultimate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2_compat import compat  # noqa: E402
from part2_transfer import align  # noqa: E402

assert data.MARKET == "developed", "run with the developed market"


BANDS = {"17-24": (17, 24), "25-34": (25, 34), "35-49": (35, 49), "50-64": (50, 64)}
OLD_AGES, OLD_W = (65.0, 70.0, 75.0, 80.0, 85.0), (0.3, 0.25, 0.2, 0.15, 0.1)


def expand_ages(emg: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Tier N12 (owner, 27 Oct): k copies of each emerging row at ages inside its band,
    replacing the band-mean age, each carrying its share of the row's exposure. Ages are
    the midpoints of k equal slices of [max(band low, 17 + years_licensed), band high]
    (motor; property has no licensing constraint); 65+ uses fixed points and weights.
    Summing the copies' predictions gives the average prediction over the band."""
    out = []
    for j in range(k):
        c = emg.copy()
        for band, age in (
            ("driver_age_band", "driver_age"),
            ("policyholder_age_band", "policyholder_age"),
        ):
            if band not in c.columns:
                continue
            b = c[band].astype("string")
            lo = b.map({x: v[0] for x, v in BANDS.items()}).astype(float)
            hi = b.map({x: v[1] for x, v in BANDS.items()}).astype(float)
            if age == "driver_age" and "years_licensed" in c.columns:
                lo = np.minimum(
                    np.maximum(lo, 17 + c["years_licensed"].astype(float).fillna(0)), hi
                )
            a = lo + (hi - lo) * (j + 0.5) / k
            a = a.where(b != "65+", OLD_AGES[j])
            c[age] = a.where(b.notna(), c[age]).astype(float)
        w = np.where(
            c.get("driver_age_band", c.get("policyholder_age_band"))
            .astype("string")
            .fillna("")
            .where(c["line"] == "motor", c.get("policyholder_age_band").astype("string").fillna(""))
            == "65+",
            OLD_W[j],
            1.0 / k,
        )
        c["customer_age"] = c["driver_age"].where(c["line"] == "motor", c["policyholder_age"])
        c["exposure"] = c["exposure"] * w
        out.append(c)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("emerging_cache")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-years", type=int, nargs="+", required=True)
    ap.add_argument("--compat", action="store_true", help="fit on the emerging schema (O2)")
    ap.add_argument("--dry-run", action="store_true", help="20k training rows per line")
    ap.add_argument("--age-points", type=int, default=0, help="N12: average over in-band ages")
    ap.add_argument(
        "--fine",
        action="store_true",
        help="split groups into single perils (developed cost = reported / group completion)",
    )
    a = ap.parse_args()

    cfg = ultimate.load_config()
    dev = harness.dataset()
    if a.compat:
        dev = compat(dev)
        a.tag = "compat_" + a.tag
    tr, _ = harness.split(dev, a.train_years, cfg["predict_year"], cfg)
    if a.dry_run:
        tr = tr.groupby("line", group_keys=False).sample(20_000, random_state=0)
        a.tag += "_dryrun"
    emg = pd.read_parquet(a.emerging_cache)
    if a.age_points:
        emg = expand_ages(emg, a.age_points)
        a.tag = "age_" + a.tag
    if os.environ.get("IAA_WEIGHT_POWER", "1") != "1":
        a.tag = f"wp{os.environ['IAA_WEIGHT_POWER']}_" + a.tag
    va = harness.hide_target(align(emg, dev))
    experiment.FROZEN_LINES = ()
    out = emg[["row_id", "line"]].astype(str).assign(market="emerging")
    groups = cfg["development"]["peril_groups"]

    # motor: record the two anchored component rates (BI first, then other perils)
    calls: list[np.ndarray] = []
    anchor = experiment._anchor

    def recording_anchor(rate, target_lc, frame):
        res = anchor(rate, target_lc, frame)
        calls.append(np.asarray(res, dtype=float))
        return res

    experiment._anchor = recording_anchor
    m_tr, m_va = tr[tr["line"] == "motor"], va[va["line"] == "motor"].reset_index(drop=True)
    total = experiment.fit_predict(m_tr, m_va)
    experiment._anchor = anchor
    if len(calls) != 2:
        raise RuntimeError(f"expected 2 motor anchoring calls, saw {len(calls)}")
    expo = m_va["exposure"].to_numpy()
    bi, other = calls[0] * expo, calls[1] * expo
    assert np.allclose(bi + other, total, rtol=1e-6), "motor components do not add up"
    mot = (out["line"] == "motor").to_numpy()
    out.loc[mot, "base_bodily_injury"], out.loc[mot, "base_other"] = bi, other
    print("motor done", flush=True)

    peril_trend = cfg.get("peril_trend") or {}

    def developed(frame: pd.DataFrame, line: str, grp: str, peril: str | None) -> np.ndarray:
        """Trended developed cost of a group, or of one peril inside it (its reported cost
        over the group's completion: the peril shares the group's development pattern)."""
        rate = peril_trend.get(line, {}).get(grp, cfg["trend"][line])
        gap = cfg["predict_year"] - frame["year"].to_numpy()
        if peril is None:
            cost = frame[f"ult_{grp}"].fillna(0.0).to_numpy()
        else:
            comp = frame[f"comp_{grp}"].to_numpy()
            cost = np.where(comp > 0, frame[f"cost_{peril}"].fillna(0.0).to_numpy() / comp, 0.0)
        return cost * np.power(1 + rate, gap)

    if a.fine:
        # motor other perils: refit the motor model with ult_other replaced by one peril's
        # developed cost (untrended: the motor path trends it), recording its other-perils
        # component (the second anchoring call)
        for peril in groups["motor"]["other"]:
            calls.clear()
            experiment._anchor = recording_anchor
            c = m_tr["comp_other"].to_numpy()
            ult = np.where(c > 0, m_tr[f"cost_{peril}"].fillna(0.0).to_numpy() / c, 0.0)
            experiment.fit_predict(m_tr.assign(ult_other=ult), m_va)
            experiment._anchor = anchor
            out.loc[mot, f"base_{peril}"] = calls[1] * expo
            print("motor", peril, "done", flush=True)

    # property: one refit per peril group (or peril) on its trended developed cost
    for line in ("buildings", "contents"):
        l_tr = tr[tr["line"] == line]
        l_va = va[va["line"] == line].reset_index(drop=True)
        rows = (out["line"] == line).to_numpy()
        for grp, perils in groups[line].items():
            targets = [(p, p) for p in perils] if a.fine and len(perils) > 1 else [(grp, None)]
            for name, peril in targets:
                y = developed(l_tr, line, grp, peril)
                pred = experiment.fit_predict(l_tr.assign(y=y), l_va)
                out.loc[rows, f"base_{name}"] = pred
                print(line, name, "done", flush=True)

    if a.age_points:  # copies carry exposure shares: their predictions sum to the band average
        cols = [c for c in out.columns if c.startswith("base_")]
        out = out.groupby(["row_id", "line", "market"], as_index=False, sort=False)[cols].sum(
            min_count=1
        )
    path = harness.CACHE / f"part2_base_perils{'_fine' if a.fine else ''}_{a.tag}.parquet"
    out.to_parquet(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
