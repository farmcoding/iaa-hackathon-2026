"""Owner bench (28 Oct): O4 with emerging-observed trends and NB credibility k.

Scores folds B and R against the best keep (same as 1027/bench.py), then fits the final
(2023-24 -> 2025) and reports the 2025 rate per exposure by line and by new business,
against the tier P board optima, and writes the submission file.

    IAA_MARKET=emerging .venv/bin/python -I scripts/part2_trend_bench.py \
        motor=0.075 contents=0.0 nbk=200 [write=1]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from iaa import experiment_emerging as e
from iaa import harness, ultimate

ROOT = Path.home() / "Projects/iaa-hackathon-2026"
kw = dict(a.split("=", 1) for a in sys.argv[1:])
df = harness.dataset()  # before the override: the dataset fingerprint hashes the config
cfg = ultimate.load_config()
for line in ("motor", "buildings", "contents"):
    if line in kw:
        cfg["trend"][line] = float(kw[line])
if "motor_bi" in kw:  # bodily-injury group trend on motor (config peril_trend)
    cfg.setdefault("peril_trend", {})["motor"] = {"bodily_injury": float(kw["motor_bi"])}
e.NB_FACTOR_K = float(kw.get("nbk", e.NB_FACTOR_K))
if "nbbasis" in kw:  # basis of the new-business factor only ("count" in O4, or "cost")
    _cf = e.credibility_factors
    e.credibility_factors = lambda tr, col, k=200.0, basis="cost": _cf(
        tr, col, k, kw["nbbasis"] if col == "new_business" else basis
    )
ultimate.load_config = lambda *a, **k: cfg  # peril_target re-reads the trend from config
t0 = time.time()
res, saved = harness._score_folds(e, df, cfg)
best = harness.best_kept(cfg)
q = harness.paired_bootstrap(saved, best, cfg)
print(
    f"RESULT trend {kw}: primary {res['primary']:.5f} ({res['primary'] - best['_primary']:+.5f}) "
    f"B {res['d2_B']:.5f} R {res['d2_R']:.5f} lnB {res['d2_ln_B']:.5f} lnR {res.get('d2_ln_R', float('nan')):.5f} "
    f"motorB {res.get('d2_motor_B', float('nan')):.4f} motorR {res.get('d2_motor_R', float('nan')):.4f} boot_q10 {q:+.5f}",
    flush=True,
)
sv = pd.concat(saved).merge(df[["row_id", "new_business"]], on="row_id")
sv["nb"] = sv["new_business"].astype(str).eq("True")
for (fold, nb), g in sv[sv.line == "motor"].groupby(["fold", "nb"]):
    print(f"  fold {fold} motor nb={nb!s:5s} cost A/E {g.y.sum() / g.pred.sum():.3f}", flush=True)
sv.drop(columns="new_business").to_parquet(harness.CACHE / "trend_bench_folds.parquet")
tr, te = harness.split(df, cfg["harness"]["final"]["train"], cfg["predict_year"], cfg)
pred = harness.predict(e, tr, te, cfg)
o4 = pd.read_csv(ROOT / "submissions/part2/exp_o4_0c4e403_20261025_131648.csv")
x = te[["row_id", "line", "exposure", "new_business"]].assign(
    p=pred, row_id=te["row_id"].astype(str)
)
x = x.merge(o4.assign(row_id=o4["row_id"].astype(str)), on="row_id")
x["nb"] = x["new_business"].astype(str).eq("True")
target = {("motor", True): 1.206, ("motor", False): 1.096}
for (line, nb), g in x.groupby(["line", "nb"]):
    ratio = g.p.sum() / g.prediction.sum()
    tgt = target.get((line, nb), {"buildings": 0.996, "contents": 0.897}.get(line))
    print(
        f"  {line:9s} nb={nb!s:5s} rate {g.p.sum() / g.exposure.sum():7.1f}  vs O4 x{ratio:.3f}"
        f"  (board optimum x{tgt:.3f})",
        flush=True,
    )
if kw.get("write") == "1":
    tag = "_".join(f"{k}{v}" for k, v in kw.items() if k != "write")
    path = ROOT / f"submissions/part2/o4_trend_{tag}.csv"
    pd.DataFrame({"row_id": x["row_id"], "prediction": x["p"]}).to_csv(path, index=False)
    print("wrote", path, flush=True)
print(f"[{time.time() - t0:.0f}s]")
