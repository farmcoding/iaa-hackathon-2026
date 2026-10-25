"""Pooled two-market base: the Part 1 specification fitted on developed AND emerging rows.

The developed-only base cannot learn what the developed market never shows: the emerging
regions (E01-E06), hail and wildfire beyond the developed range, gender. Pooling lets the
trees learn those from the emerging rows while borrowing the developed market's volume
for everything shared. A `market` field lets them split the two where they differ.
Developed rows use the emerging schema (scripts/part2_compat.py) so both markets carry
the same information on shared fields.

One fit per emerging validation year, so a fold never trains on the rows it scores:
- tag B (scores 2024): developed 2021-23 + emerging 2023
- tag R (scores 2023): developed 2021-23 + emerging 2024
- tag final (2025): developed 2021-24 + emerging 2023-24

    uv run python scripts/part2_pooled.py EMERGING_CACHE --tag B [--emerging-weight 5]

Writes data/cache/part2_pooled[_w<k>]_<tag>.parquet: market, row_id, line, base_pooled
(prediction for every emerging row, CU). Run under the developed market (the default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from iaa import data, experiment, features, harness, ultimate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2_compat import compat  # noqa: E402

assert data.MARKET == "developed", "run with the developed market"

FITS = {
    "B": ([2021, 2022, 2023], [2023]),
    "R": ([2021, 2022, 2023], [2024]),
    "final": ([2021, 2022, 2023, 2024], [2023, 2024]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("emerging_cache")
    ap.add_argument("--tag", required=True, choices=list(FITS))
    ap.add_argument(
        "--emerging-weight",
        type=int,
        default=1,
        help="repeat emerging training rows k times (sample weight via duplication)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dev_years, emg_years = FITS[a.tag]

    cfg = ultimate.load_config()
    ecfg = ultimate.load_config(data.ROOT / "configs" / "market_emerging.yaml")
    dev = compat(harness.dataset())
    emg = pd.read_parquet(a.emerging_cache)
    tr_d, _ = harness.split(dev, dev_years, cfg["predict_year"], cfg)
    ecfg = {**ecfg, "harness": {}}  # trend only; no base attachment here
    tr_e, _ = harness.split(emg, emg_years, cfg["predict_year"], ecfg)
    if a.dry_run:
        tr_d = tr_d.groupby("line", group_keys=False).sample(20_000, random_state=0)
    if a.emerging_weight > 1:
        tr_e = pd.concat([tr_e] * a.emerging_weight, ignore_index=True)

    # stack, then re-type categoricals on the union of levels (E-regions, age bands, gender)
    tr_d, tr_e = tr_d.assign(market="developed"), tr_e.assign(market="emerging")
    target = emg.assign(market="emerging")
    n_tr = len(tr_d) + len(tr_e)
    allf = pd.concat([tr_d, tr_e, target], ignore_index=True)
    for c in allf.columns:
        if c in features.ID_COLS or c in ("split", "line"):
            continue
        dt = allf[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            continue
        if (
            isinstance(dt, pd.CategoricalDtype)
            or not pd.api.types.is_numeric_dtype(dt)
            or (pd.api.types.is_bool_dtype(dt))
        ):
            allf[c] = allf[c].astype("string").astype("category")
    # parquet round trip: in-memory string categoricals carry pd.NA, which the target
    # encoder rejects; read back they match the harness dataset's typing (see harness.dataset)
    tmp = harness.CACHE / f"_pooled_tmp_{a.tag}.parquet"
    allf.to_parquet(tmp)
    allf = pd.read_parquet(tmp)
    tmp.unlink()
    tr, va = allf.iloc[:n_tr].copy(), allf.iloc[n_tr:].copy()
    experiment.FROZEN_LINES = ()
    # Part 1's pruning lists were data-driven on the developed market, where hail, region and
    # tenure carried little signal; here they are what the emerging rows teach. Admissibility
    # drops (NON_FEATURES, LINE_DROP) stay.
    experiment.PRUNED = {}
    pred = harness.check_predictions(experiment.fit_predict(tr, harness.hide_target(va)), len(va))
    out = emg[["row_id", "line"]].astype(str).assign(market="emerging", base_pooled=pred)
    w = f"_w{a.emerging_weight}" if a.emerging_weight > 1 else ""
    tag = a.tag + ("_dryrun" if a.dry_run else "")
    path = harness.CACHE / f"part2_pooled{w}_{tag}.parquet"
    out.to_parquet(path)
    r = out.assign(e=emg["exposure"].to_numpy(), s=emg["split"].astype(str).to_numpy())
    print(
        "wrote",
        path,
        r.groupby(["line", "s"])
        .apply(lambda g: g.base_pooled.sum() / g.e.sum(), include_groups=False)
        .round(1)
        .to_dict(),
    )


if __name__ == "__main__":
    main()
