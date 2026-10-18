"""Re-score earlier kept runs from their exact experiment.py in git and save their fold
predictions, named so the dashboard matches them to their ledger rows (one second before
the row's timestamp). Read-only on the ledger."""

import csv
import subprocess
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from iaa import harness, ultimate

root = Path(sys.argv[1])
cfg = ultimate.load_config()
allrows = list(csv.DictReader((root / "results.tsv").open(), delimiter="\t"))
prev_ts = {}
for i, r in enumerate(allrows):
    prev_ts[r["timestamp"]] = allrows[i - 1]["timestamp"] if i else "2000-01-01T00:00:00"
rows = [r for r in allrows if r["verdict"] in ("keep", "keep_owner")]
log = subprocess.run(
    ["git", "log", "--format=%h%x09%s", "--", "src/iaa/experiment.py"],
    cwd=root,
    capture_output=True,
    text=True,
    check=True,
    timeout=30,
).stdout.splitlines()
commits = {s: h for h, s in (ln.split("\t", 1) for ln in log)}
df = None
for r in rows:
    ts = datetime.fromisoformat(r["timestamp"])
    name = (ts - timedelta(seconds=1)).strftime("%Y%m%dT%H%M%S")
    out = harness.PREDS / f"{name}.parquet"
    lo = datetime.fromisoformat(prev_ts[r["timestamp"]]).strftime("%Y%m%dT%H%M%S")
    hi = ts.strftime("%Y%m%dT%H%M%S")
    existing = list(harness.PREDS.glob("*.parquet")) if harness.PREDS.exists() else []
    if any(lo <= p.stem <= hi for p in existing):
        print("has preds:", r["description"][:60])
        continue
    desc = r["description"]
    if desc.startswith("baseline"):
        h = "afcb387"
    else:
        h = next(
            (h for s, h in commits.items() if s.startswith("exp(keep)") and desc[:60] in s), None
        )
    if h is None:
        print("no commit for:", desc[:60])
        continue
    src = subprocess.run(
        ["git", "show", f"{h}:src/iaa/experiment.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    mod = types.ModuleType(f"exp_{h}")
    exec(compile(src, f"exp_{h}", "exec"), mod.__dict__)
    df = harness.dataset() if df is None else df
    saved = []
    for fname, fold in cfg["harness"]["folds"].items():
        tr, va = harness.split(df, fold["train"], fold["valid"], cfg)
        pred = harness.check_predictions(mod.fit_predict(tr, harness.hide_target(va)), len(va))
        saved.append(
            va[["row_id", "line", "exposure"]].assign(y=va["y_ult"], pred=pred, fold=fname)
        )
    harness.PREDS.mkdir(parents=True, exist_ok=True)
    pd.concat(saved).to_parquet(out)
    print("backfilled", h, "->", out.name, desc[:60])
