"""Board level reads for the Part 2 2025 book (pre-registered in
research/experiment_backlog_emerging.md, tier P, 28 Oct).

The public board scores pooled Tweedie D2 (p=1.5) against the constant-mean baseline. If
one line's predictions are multiplied by c and everything else is fixed, the board score is
exactly

    S(c) = A - B c^(-1/2) - G c^(1/2)

(the p=1.5 unit deviance has a y*mu^(-1/2) term and a mu^(1/2) term; the baseline does not
move). Three scores on one line (c = 1 and two probes) fix A, B and G, and the line's
board-optimal multiplier is c* = B / G. Public rows are a random third of the same 2025
book as the private rows, so the 2025 year effect is shared and c* carries over, up to
sampling noise on the public third (sd about 4-5% on motor and contents and 5% on
buildings, from fold B subsamples of public size).

    write:  .venv/bin/python -I scripts/part2_level_probe.py write BASE.csv LINE C [C ...]
    solve:  .venv/bin/python -I scripts/part2_level_probe.py solve S1 C1 S2 C2 S3 C3

`write` scales LINE's rows (optionally only new business with --nb) and writes
submissions/part2/probe_<base>_<line>[_nb]_<c>.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "part2_raw" / "emerging"


def test_rows() -> pd.DataFrame:
    parts = []
    for line in ("motor", "buildings", "contents"):
        t = pd.read_parquet(RAW / f"test_{line}.parquet", columns=["row_id", "new_business"])
        parts.append(t.assign(line=line))
    return pd.concat(parts, ignore_index=True)


def write(base: Path, line: str, cs: list[float], nb_only: bool) -> None:
    sub = pd.read_csv(base)
    rows = test_rows()
    m = sub.merge(rows, on="row_id", how="left", validate="one_to_one")
    mask = m["line"].eq(line)
    if nb_only:
        mask &= m["new_business"].astype(str).eq("True")
    tag = f"{line}_nb" if nb_only else line
    for c in cs:
        out = sub.copy()
        out.loc[mask.to_numpy(), "prediction"] *= c
        dest = ROOT / "submissions" / "part2" / f"probe_{base.stem[:6]}_{tag}_{c:g}.csv"
        out.to_csv(dest, index=False)
        print(dest.relative_to(ROOT), f"rows scaled {int(mask.sum())}")


def solve(pairs: list[tuple[float, float]]) -> float:
    """c* from three (multiplier, board score) pairs on one line."""
    x = np.array([[1.0, -(c**-0.5), -(c**0.5)] for c, _ in pairs])
    a, b, g = np.linalg.solve(x, np.array([s for _, s in pairs]))
    cstar = b / g
    best = a - b * cstar**-0.5 - g * cstar**0.5
    print(f"A {a:.5f} B {b:.5f} G {g:.5f} -> c* {cstar:.3f}, board score at c* {best:.5f}")
    return cstar


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("write")
    w.add_argument("base", type=Path)
    w.add_argument("line", choices=["motor", "buildings", "contents"])
    w.add_argument("c", type=float, nargs="+")
    w.add_argument("--nb", action="store_true", help="scale new-business rows only")
    s = sp.add_parser("solve")
    s.add_argument("vals", type=float, nargs=6, help="S1 C1 S2 C2 S3 C3")
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.base, a.line, a.c, a.nb)
    else:
        v = a.vals
        solve([(v[1], v[0]), (v[3], v[2]), (v[5], v[4])])


if __name__ == "__main__":
    main()
