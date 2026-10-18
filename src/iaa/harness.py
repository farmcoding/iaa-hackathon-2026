"""Experiment harness: builds the dataset once, runs an experiment on the frozen
folds, scores it exactly like the leaderboard, applies guardrails and logs the result.

FROZEN: the experiment agent must not edit this file.

Usage:
    python -m iaa.harness "short description"        # validate experiment.py
    python -m iaa.harness --final "description"      # fit 2021-2024, write submission
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, features, score, target, ultimate

CACHE = data.ROOT / "data" / "cache"
RESULTS = data.ROOT / "results.tsv"
PREDS = CACHE / "preds"  # per-run validation predictions for lift charts (gitignored)
SUBMISSIONS = data.ROOT / "submissions"
FROZEN = ["data.py", "target.py", "ultimate.py", "features.py", "score.py", "harness.py"]
TARGET_COLS_PREFIX = ("cost_", "n_", "ult_", "comp_")
FIELDS = [
    "timestamp",
    "commit",
    "verdict",
    "primary",
    "d2_A",
    "d2_B",
    "min_fold",
    "d2_motor_B",
    "d2_buildings_B",
    "d2_contents_B",
    "ae_motor_B",
    "ae_buildings_B",
    "ae_contents_B",
    "gini_B",
    "seconds",
    "description",
]


def _fingerprint() -> str:
    h = hashlib.sha256()
    for name in FROZEN:
        h.update((data.ROOT / "src" / "iaa" / name).read_bytes())
    h.update(ultimate.CONFIG.read_bytes())
    return h.hexdigest()[:12]


def dataset() -> pd.DataFrame:
    """Train (2021-2024, with developed targets) and test (2025) rows, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"dataset_{_fingerprint()}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows, _, _ = target.reported_targets()
    dev = ultimate.developed_targets(rows)
    tcols = ["row_id"] + [c for c in dev.columns if c.startswith(TARGET_COLS_PREFIX)] + ["y_ult"]
    tr = features.build("train").merge(dev[tcols], on="row_id", how="left")
    te = features.build("test")
    tr["split"], te["split"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    df = features.categorize(df)
    for col in features.ID_COLS:
        df[col] = df[col].astype("string")
    df.to_parquet(path)
    return df


def on_level(df: pd.DataFrame, to_year: int, cfg: dict) -> np.ndarray:
    rate = df["line"].map(cfg["trend"]).astype(float).to_numpy()
    return np.power(1 + rate, to_year - df["year"].to_numpy())


def split(df: pd.DataFrame, train_years, valid_year, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[(df["split"] == "train") & df["year"].isin(train_years)].copy()
    tr["y"] = tr["y_ult"] * on_level(tr, valid_year, cfg)
    tr["trend_factor"] = on_level(tr, valid_year, cfg)
    if valid_year == cfg["predict_year"]:
        va = df[df["split"] == "test"].copy()
    else:
        va = df[(df["split"] == "train") & (df["year"] == valid_year)].copy()
    return tr, va


def hide_target(va: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in va.columns if c.startswith(TARGET_COLS_PREFIX) or c in ("y_ult", "y")]
    return va.drop(columns=drop)


def check_predictions(pred: np.ndarray, n: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (n,):
        raise ValueError(f"prediction shape {pred.shape}, expected ({n},)")
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("predictions must be finite and non-negative")
    return np.maximum(pred, score.FLOOR)


def _commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=data.ROOT,
        check=False,
        timeout=30,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "src", "configs"],
        capture_output=True,
        text=True,
        cwd=data.ROOT,
        check=False,
        timeout=30,
    )
    return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def best_kept() -> dict | None:
    if not RESULTS.exists():
        return None
    kept = ("keep", "keep_owner")
    rows = [r for r in csv.DictReader(RESULTS.open(), delimiter="\t") if r["verdict"] in kept]
    return max(rows, key=lambda r: float(r["primary"])) if rows else None


def verdict(res: dict, cfg: dict) -> str:
    floor = cfg["harness"]["noise_floor"] or 0.0
    best = best_kept()
    if best is None:
        return "keep"
    better = res["primary"] > float(best["primary"]) + floor
    stable = res["min_fold"] >= float(best["min_fold"]) - floor
    return "keep" if better and stable else "discard"


def validate(
    description: str, module: str = "iaa.experiment", log: bool = True, owner_keep: bool = False
) -> dict:
    cfg = ultimate.load_config()
    df = dataset()
    exp = importlib.import_module(module)
    t0 = time.time()
    res: dict = {}
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    saved = []
    for name, fold in cfg["harness"]["folds"].items():
        tr, va = split(df, fold["train"], fold["valid"], cfg)
        pred = check_predictions(exp.fit_predict(tr, hide_target(va)), len(va))
        va = va.assign(pred=pred, y=va["y_ult"])
        saved.append(va[["row_id", "line", "exposure", "y", "pred"]].assign(fold=name))
        card = score.scorecard(va)
        res[f"d2_{name}"] = card["d2"]
        if name == "B":
            for k in ("motor", "buildings", "contents"):
                res[f"d2_{k}_B"] = card[f"d2_{k}"]
                res[f"ae_{k}_B"] = card[f"ae_{k}"]
            res["gini_B"] = card["gini"]
        print(f"fold {name}: " + "  ".join(f"{k}={v:.5f}" for k, v in card.items()), flush=True)
    res["seconds"] = round(time.time() - t0)
    folds = [res[f"d2_{n}"] for n in cfg["harness"]["folds"]]
    res["primary"], res["min_fold"] = float(np.mean(folds)), float(np.min(folds))
    if res["seconds"] > cfg["harness"]["time_budget_seconds"]:
        res["verdict"] = "over_budget"
    elif owner_keep:
        res["verdict"] = "keep_owner"  # human decision, recorded in assumptions_log.md
    else:
        res["verdict"] = verdict(res, cfg)
    res["run_id"] = run_id
    if log:
        PREDS.mkdir(parents=True, exist_ok=True)
        pd.concat(saved).to_parquet(PREDS / f"{run_id}.parquet")
    res.update(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        commit=_commit(),
        description=description,
    )
    if log:
        new = not RESULTS.exists()
        with RESULTS.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in res.items()})
    print(
        f"RESULT primary={res['primary']:.5f} min_fold={res['min_fold']:.5f} "
        f"verdict={res['verdict']} seconds={res['seconds']}"
    )
    return res


def final(description: str, module: str = "iaa.experiment") -> Path:
    cfg = ultimate.load_config()
    df = dataset()
    exp = importlib.import_module(module)
    tr, te = split(df, cfg["harness"]["final"]["train"], cfg["predict_year"], cfg)
    pred = check_predictions(exp.fit_predict(tr, hide_target(te)), len(te))
    sub = pd.DataFrame({"row_id": te["row_id"].astype(str), "prediction": pred})
    sample = pd.read_parquet(data.RAW / "sample_submission.parquet")
    if set(sub["row_id"]) != set(sample["row_id"].astype(str)) or len(sub) != len(sample):
        raise ValueError("submission rows do not match sample_submission")
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"{_commit()}_{datetime.now():%Y%m%d_%H%M}.csv"
    sub.to_csv(path, index=False)
    by_line = te.assign(p=pred).groupby("line").apply(lambda g: g.p.sum() / g.exposure.sum())
    print(f"wrote {path}  rate per exposure by line: {by_line.round(1).to_dict()}  ({description})")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("description")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--module", default="iaa.experiment")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--owner-keep", action="store_true", help="human decision to keep")
    a = ap.parse_args()
    if a.final:
        final(a.description, a.module)
    else:
        validate(a.description, a.module, log=not a.no_log, owner_keep=a.owner_keep)


if __name__ == "__main__":
    main()
