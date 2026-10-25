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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, experiment, harness, ultimate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2_transfer import align  # noqa: E402

assert data.MARKET == "developed", "run with the developed market"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("emerging_cache")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-years", type=int, nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true", help="20k training rows per line")
    ap.add_argument(
        "--fine",
        action="store_true",
        help="split groups into single perils (developed cost = reported / group completion)",
    )
    a = ap.parse_args()

    cfg = ultimate.load_config()
    dev = harness.dataset()
    tr, _ = harness.split(dev, a.train_years, cfg["predict_year"], cfg)
    if a.dry_run:
        tr = tr.groupby("line", group_keys=False).sample(20_000, random_state=0)
        a.tag += "_dryrun"
    emg = pd.read_parquet(a.emerging_cache)
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

    path = harness.CACHE / f"part2_base_perils{'_fine' if a.fine else ''}_{a.tag}.parquet"
    out.to_parquet(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
