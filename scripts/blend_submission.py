"""Owner script: blend the best LightGBM run with TabM, score it on both folds, and
optionally write the blended submission.

The blend is a fixed-weight average of the two models' predictions, re-anchored per line
so each line keeps the LightGBM total (the level comes from the frozen level method, not
from the neural net). The weight is set by the owner, not tuned per fold.

Validation rows go to results.tsv with verdict "blend_pass" or "blend_fail" (the
mechanical keep rule against the best kept LightGBM run). They are not "keep" rows, so the
agent loop keeps competing against the best single LightGBM model.

Usage:
  python -I scripts/blend_submission.py --lgbm-run 20261018T094703 --tabm-dir ~/.cache/iaa-gpu/out
  python -I scripts/blend_submission.py ... --final --lgbm-sub submissions/<file>.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, harness, score, ultimate


def blend(lgbm: pd.Series, tabm: pd.Series, line: pd.Series, w: float) -> pd.Series:
    r = (1 - w) * lgbm + w * tabm
    return r * line.map(lgbm.groupby(line).sum() / r.groupby(line).sum())


def tabm_preds(tabm_dir: Path, fold: str) -> pd.DataFrame:
    path = tabm_dir / f"tabm_{fold}_{harness._fingerprint()}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}: run tabm_foldB.py --fold {fold} on the current dataset")
    return pd.read_parquet(path)[["row_id", "pred"]].rename(columns={"pred": "tabm"})


def validate(a: argparse.Namespace, cfg: dict) -> dict:
    lg = pd.read_parquet(harness.PREDS / f"{a.lgbm_run}.parquet")
    res: dict = {}
    saved = []
    for fold in cfg["harness"]["folds"]:
        g = lg[lg["fold"] == fold]
        m = g.merge(tabm_preds(a.tabm_dir, fold), on="row_id", how="inner")
        if len(m) != len(g):
            raise SystemExit(
                f"fold {fold}: {len(g) - len(m)} LightGBM rows have no TabM prediction"
            )
        m["pred"] = blend(m["pred"], m["tabm"], m["line"], a.weight)
        saved.append(m[["row_id", "line", "exposure", "y", "pred", "fold"]])
        card = score.scorecard(m)
        res[f"d2_{fold}"] = card["d2"]
        if fold == "B":
            for k in ("motor", "buildings", "contents"):
                res[f"d2_{k}_B"] = card[f"d2_{k}"]
                res[f"ae_{k}_B"] = card[f"ae_{k}"]
            res["gini_B"] = card["gini"]
        print(f"fold {fold}: " + "  ".join(f"{k}={v:.5f}" for k, v in card.items()))
    folds = [res[f"d2_{f}"] for f in cfg["harness"]["folds"]]
    res["primary"] = harness.primary_of({f: res[f"d2_{f}"] for f in cfg["harness"]["folds"]}, cfg)
    res["min_fold"] = float(np.min(folds))
    res["verdict"] = "blend_pass" if harness.verdict(res, cfg) == "keep" else "blend_fail"
    res["run_id"] = datetime.now().strftime("%Y%m%dT%H%M%S")
    res.update(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        commit=harness._commit(),
        seconds=0,
        description=f"owner blend: LightGBM {a.lgbm_run} + TabM, weight {a.weight}",
    )
    if not a.no_log:
        pd.concat(saved).to_parquet(harness.PREDS / f"{res['run_id']}.parquet")
        with harness.RESULTS.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=harness.FIELDS, delimiter="\t", extrasaction="ignore")
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in res.items()})
    print(
        f"RESULT primary={res['primary']:.5f} min_fold={res['min_fold']:.5f} verdict={res['verdict']}"
    )
    return res


def final(a: argparse.Namespace) -> Path:
    lg = pd.read_csv(a.lgbm_sub, dtype={"row_id": str})
    df = harness.dataset()
    te = df[df["split"] == "test"][["row_id", "line", "exposure"]].astype({"row_id": str})
    m = lg.merge(te, on="row_id").merge(
        tabm_preds(a.tabm_dir, "final").astype({"row_id": str}), on="row_id"
    )
    if len(m) != len(lg):
        raise SystemExit(f"{len(lg) - len(m)} submission rows have no TabM prediction")
    pred = harness.check_predictions(blend(m["prediction"], m["tabm"], m["line"], a.weight), len(m))
    sub = pd.DataFrame({"row_id": m["row_id"], "prediction": pred})
    sample = pd.read_parquet(data.RAW / "sample_submission.parquet")
    if set(sub["row_id"]) != set(sample["row_id"].astype(str)) or len(sub) != len(sample):
        raise SystemExit("submission rows do not match sample_submission")
    path = (
        harness.SUBMISSIONS
        / f"blend_{Path(a.lgbm_sub).stem}_w{a.weight}_{datetime.now():%Y%m%d_%H%M}.csv"
    )
    sub.to_csv(path, index=False)
    by_line = m.assign(p=pred).groupby("line").apply(lambda g: g.p.sum() / g.exposure.sum())
    print(f"wrote {path}  rate per exposure by line: {by_line.round(1).to_dict()}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lgbm-run", required=True, help="run_id of the LightGBM validation predictions"
    )
    ap.add_argument("--tabm-dir", type=Path, default=Path.home() / ".cache" / "iaa-gpu" / "out")
    ap.add_argument("--weight", type=float, default=0.5, help="TabM weight, fixed by the owner")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument(
        "--lgbm-sub", help="LightGBM submission CSV from harness --final (with --final)"
    )
    a = ap.parse_args()
    a.tabm_dir = a.tabm_dir.expanduser()
    validate(a, ultimate.load_config())
    if a.final:
        if not a.lgbm_sub:
            raise SystemExit("--final needs --lgbm-sub")
        final(a)


if __name__ == "__main__":
    main()
