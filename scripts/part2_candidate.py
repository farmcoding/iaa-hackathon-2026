"""Rebuild the 28 Oct Part 2 candidate `o4_mech_lvl_20261028.csv` from the repo.

Step 1 (mechanisms, from 2023-24 experience): O4 refitted with the contents trend at 0% and
a 12% bodily-injury trend on motor (other motor perils stay at 3%):

    IAA_MARKET=emerging .venv/bin/python -I scripts/part2_trend_bench.py \
        contents=0.0 motor_bi=0.12 write=1

writes submissions/part2/o4_trend_contents0.0_motor_bi0.12.csv.

Step 2 (this script): the shrunk board residual on motor, under the selection rule amended
28 Oct (docs/SIGNOFF.md). Board-optimal levels against O4 (tier P, backlog): motor new
business 1.206, renewals 1.096; shrunk to 1.185 and 1.07; divided by what step 1 already
moved (1.054 and 1.050) gives the residuals below. Buildings and contents are unchanged.

    .venv/bin/python -I scripts/part2_candidate.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # run with -I: the script dir is not on the path
from part2_level_probe import test_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions" / "part2"
RESIDUAL = {True: 1.185 / 1.054, False: 1.07 / 1.050}  # motor, by new_business
EXPECTED_SHA = "dabd894e8644f416"


def main() -> None:
    s = pd.read_csv(SUB / "o4_trend_contents0.0_motor_bi0.12.csv")
    x = s.merge(test_rows(), on="row_id", validate="one_to_one")
    nb = x["new_business"].astype(str).eq("True")
    for flag, f in RESIDUAL.items():
        x.loc[x["line"].eq("motor") & (nb == flag), "prediction"] *= f
    out = SUB / "o4_mech_lvl_20261028.csv"
    x[["row_id", "prediction"]].to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(out.relative_to(ROOT), sha)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"sha256 {sha} does not match the submitted file {EXPECTED_SHA}")


if __name__ == "__main__":
    main()
