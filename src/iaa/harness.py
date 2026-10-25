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
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iaa import data, features, score, target, ultimate

CACHE = data.ROOT / "data" / "cache"
RESULTS = data.ROOT / (
    "results.tsv" if data.MARKET == "developed" else f"results_{data.MARKET}.tsv"
)
PREDS = CACHE / "preds"  # per-run validation predictions for lift charts (gitignored)
SUBMISSIONS = data.ROOT / "submissions" / ("" if data.MARKET == "developed" else "part2")
FROZEN = ["data.py", "target.py", "ultimate.py", "features.py", "score.py", "harness.py"]
# The dataset cache depends only on these files and on the config outside `harness:`.
# Harness logic (folds, keep rule, logging) can change without rebuilding the dataset or
# orphaning artefacts tagged with the fingerprint (TabM predictions).
DATA_CODE = ["data.py", "target.py", "ultimate.py", "features.py"]
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
    "run_id",
    "confirm_primary",
    "confirm_min_fold",
    # owner (3.23): 1 when any line's fold-B A/E falls outside the 0.97-1.03 band set
    # in program.md, else 0. A ledger flag, not a gate: it does not change the verdict.
    "ae_flag",
]
AE_BAND = (0.97, 1.03)


def _fingerprint() -> str:
    h = hashlib.sha256()
    for name in DATA_CODE:
        h.update((data.ROOT / "src" / "iaa" / name).read_bytes())
    cfg = {k: v for k, v in ultimate.load_config().items() if k != "harness"}
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
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
    df = features.finalize(df)
    df.to_parquet(path)
    # read back, so the first run after a rebuild sees exactly what later runs see
    # (the in-memory frame's categoricals differ from the parquet round trip)
    return pd.read_parquet(path)


def on_level(df: pd.DataFrame, to_year: int, cfg: dict) -> np.ndarray:
    rate = df["line"].map(cfg["trend"]).astype(float).to_numpy()
    return np.power(1 + rate, to_year - df["year"].to_numpy())


def split(df: pd.DataFrame, train_years, valid_year, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[(df["split"] == "train") & df["year"].isin(train_years)].copy()
    tr["trend_factor"] = on_level(tr, valid_year, cfg)
    tr["y"] = tr["y_ult"] * tr["trend_factor"]
    # peril-group trend overrides (config peril_trend): re-trend that group's developed
    # cost at its own rate; y_ult is the sum of the line's ult_<group> columns
    for line, groups in (cfg.get("peril_trend") or {}).items():
        on_line = (tr["line"] == line).to_numpy()
        gap = tr["year"].to_numpy()
        for group, rate in groups.items():
            own = np.power(1 + rate, valid_year - gap)
            extra = tr[f"ult_{group}"].fillna(0).to_numpy() * (own - tr["trend_factor"].to_numpy())
            tr["y"] = tr["y"] + np.where(on_line, extra, 0.0)
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


def primary_of(fold_scores: dict[str, float], cfg: dict) -> float:
    """Weighted mean of fold D2. Fold B (train 2021-23, score 2024) tracks the public
    leaderboard; fold A inflated gains about fourfold, so B carries double weight."""
    w = cfg["harness"]["primary_weights"]
    return sum(w[k] * fold_scores[k] for k in w) / sum(w.values())


def best_kept(cfg: dict | None = None) -> dict | None:
    if not RESULTS.exists():
        return None
    cfg = cfg or ultimate.load_config()
    kept = ("keep", "keep_owner")
    # owner can retire a kept row from the comparison bar (e.g. a favourable unseeded
    # draw later re-baselined); listed by timestamp under harness.superseded
    superseded = set(cfg["harness"].get("superseded") or [])
    rows = [
        r
        for r in csv.DictReader(RESULTS.open(), delimiter="\t")
        if r["verdict"] in kept and r["timestamp"] not in superseded
    ]
    # an owner keep re-bases the model on purpose (an admissibility or assumption change can
    # lower the score), so the bar is the best keep from the latest owner keep onward
    owner = [i for i, r in enumerate(rows) if r["verdict"] == "keep_owner"]
    rows = rows[owner[-1] :] if owner else rows
    for r in rows:
        r["_primary"] = primary_of(
            {k: float(r[f"d2_{k}"]) for k in cfg["harness"]["primary_weights"]}, cfg
        )
    return max(rows, key=lambda r: r["_primary"]) if rows else None


def verdict(res: dict, cfg: dict) -> str:
    floor = cfg["harness"]["noise_floor"] or 0.0
    best = best_kept(cfg)
    if best is None:
        return "keep"
    fold_floor = cfg["harness"].get("fold_floor", floor)
    better = res["primary"] > best["_primary"] + floor
    stable = res["min_fold"] >= float(best["min_fold"]) - fold_floor
    return "keep" if better and stable else "discard"


def _ensure_header(path: Path, fields: list[str]) -> None:
    """Rewrite the ledger's header line in place when FIELDS has grown (append-only:
    new columns land at the end). Data rows are untouched; DictReader fills a row that
    is now short a trailing value with its restval, so old rows still parse. Reads and
    writes with newline="" so a CRLF ledger keeps its line endings instead of every
    data row being silently rewritten as LF (a large, spurious git diff)."""
    if not path.exists():
        return
    with path.open("r", newline="") as f:
        text = f.read()
    if not text:
        return
    first_line, _, rest = text.partition("\n")
    line_end = "\r\n" if first_line.endswith("\r") else "\n"
    header_text = first_line[: -len("\r")] if line_end == "\r\n" else first_line
    existing = header_text.split("\t")
    if existing == fields:
        return
    if fields[: len(existing)] != existing:
        return  # unrecognized header shape; leave it rather than guess
    with path.open("w", newline="") as f:
        f.write("\t".join(fields) + line_end + rest)


def validate(
    description: str, module: str = "iaa.experiment", log: bool = True, owner_keep: bool = False
) -> dict:
    cfg = ultimate.load_config()
    df = dataset()
    exp = importlib.import_module(module)
    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    res, saved = _score_folds(exp, df, cfg)
    res["seconds"] = round(time.time() - t0)
    if res["seconds"] > cfg["harness"]["time_budget_seconds"]:
        res["verdict"] = "over_budget"
    elif owner_keep:
        res["verdict"] = "keep_owner"  # human decision, recorded in assumptions_log.md
    else:
        res["verdict"] = verdict(res, cfg)
        offset = cfg["harness"].get("confirm_seed_offset")
        if res["verdict"] == "keep" and offset:
            _confirm(res, saved, module, df, cfg, offset)
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
        if not new:
            _ensure_header(RESULTS, FIELDS)
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


def _confirm(res: dict, saved: list, module: str, df: pd.DataFrame, cfg: dict, offset: int) -> None:
    """A candidate keep must pass the keep rule a second time on fresh seeds (owner
    decision 2026-10-18): it protects against a lucky draw on either side."""
    os.environ["IAA_SEED_OFFSET"] = str(offset)
    try:
        exp = importlib.reload(importlib.import_module(module))
        res2, saved2 = _score_folds(exp, df, cfg)
    finally:
        os.environ.pop("IAA_SEED_OFFSET", None)
        importlib.reload(importlib.import_module(module))
    res["confirm_primary"], res["confirm_min_fold"] = res2["primary"], res2["min_fold"]
    same = np.allclose(pd.concat(saved)["pred"].to_numpy(), pd.concat(saved2)["pred"].to_numpy())
    if same:
        print("confirmation run reproduced the first run exactly: the seed hook is not used")
    ok = not same and verdict(res2, cfg) == "keep"
    print(
        f"confirmation (seed offset {offset}): primary={res2['primary']:.5f} "
        f"min_fold={res2['min_fold']:.5f} -> {'confirmed' if ok else 'not confirmed'}"
    )
    if not ok:
        res["verdict"] = "unconfirmed"


def _score_folds(exp, df: pd.DataFrame, cfg: dict) -> tuple[dict, list]:
    res: dict = {}
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
    folds = [res[f"d2_{n}"] for n in cfg["harness"]["folds"]]
    res["primary"] = primary_of({n: res[f"d2_{n}"] for n in cfg["harness"]["folds"]}, cfg)
    res["min_fold"] = float(np.min(folds))
    lo, hi = AE_BAND
    res["ae_flag"] = int(
        any(not (lo <= res[f"ae_{k}_B"] <= hi) for k in ("motor", "buildings", "contents"))
    )
    return res, saved


def smoke(module: str = "iaa.experiment", rows_per_line: int = 50_000) -> None:
    """Run the experiment on a small subsample of both folds to catch crashes fast."""
    cfg = ultimate.load_config()
    df = dataset()
    df = df.groupby(["split", "line", "year"], group_keys=False).head(rows_per_line // 4)
    exp = importlib.import_module(module)
    for name, fold in cfg["harness"]["folds"].items():
        tr, va = split(df, fold["train"], fold["valid"], cfg)
        check_predictions(exp.fit_predict(tr, hide_target(va)), len(va))
        print(f"smoke fold {name}: ok ({len(tr)} train rows, {len(va)} valid rows)")


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
    ap.add_argument("--smoke", action="store_true", help="fast crash check on a subsample")
    a = ap.parse_args()
    if a.smoke:
        smoke(a.module)
    elif a.final:
        final(a.description, a.module)
    else:
        validate(a.description, a.module, log=not a.no_log, owner_keep=a.owner_keep)


if __name__ == "__main__":
    main()
