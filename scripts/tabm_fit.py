"""TabM per-line fits on fold A, B or the final fit (train 2021-2024, predict 2025), on Apple Metal.

Same frozen targets, trend and level anchoring as the LightGBM loop (iaa.harness.split).
Per line: TabM with a Tweedie (power 1.5) loss on the rate target, exposure-weighted,
early stopping on a 5% random holdout of training rows. Saves the fold's predictions to
~/.cache/iaa-gpu/out/tabm_<fold>_<dataset fingerprint>.parquet (outside the repo: they are
derived from competition data). With --lgbm-preds, also reports a blend-weight sweep.
Training on Metal is not bit-for-bit reproducible, so the saved predictions (sha256 in
docs/SIGNOFF.md) are the artefact of record. scripts/blend_submission.py consumes them.

Usage (separate GPU venv, `uv pip install -e '.[gpu]'`):
  python -I scripts/tabm_fit.py [--fold A|B|final] [--sample N] [--epochs E] [--lgbm-preds PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tabm
import torch

from iaa import harness, score, ultimate

P = 1.5
OUT = Path.home() / ".cache" / "iaa-gpu" / "out"
NON_FEATURES = {
    "row_id",
    "policy_id",
    "customer_id",
    "address_id",
    "period_start",
    "period_end",
    "split",
    "y",
    "y_ult",
    "trend_factor",
    "year",
    "line",
    "exposure",
}


def load_line(line: str) -> pd.DataFrame:
    # the dataset for the current frozen code, not the newest file on disk
    path = harness.CACHE / f"dataset_{harness._fingerprint()}.parquet"
    if not path.exists():
        harness.dataset()
    return pd.read_parquet(path, filters=[("line", "==", line)])


def prep(d: pd.DataFrame, ref: pd.DataFrame, nums: list[str], cats: list[str]):
    cols = []
    for c in nums:
        r = ref[c].astype(float)
        med = float(np.nanmedian(r))
        lo, hi = np.nanquantile(r, [0.01, 0.99])
        x = d[c].astype(float).to_numpy()
        isna = np.isnan(x)
        x = np.where(isna, med, x)
        cols.append((np.clip(x, lo, hi) - med) / (hi - lo + 1e-9))
        if isna.any() or ref[c].isna().any():
            cols.append(isna.astype(float))
    xn = np.stack(cols, 1).astype(np.float32)
    xc = np.stack([d[c].cat.codes.to_numpy() + 1 for c in cats], 1).astype(np.int64)
    return xn, xc


def tweedie_loss(eta: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (
        w[:, None]
        * (-y[:, None] * torch.exp((1 - P) * eta) / (1 - P) + torch.exp((2 - P) * eta) / (2 - P))
    ).sum() / w.sum()


def fit_line(tr: pd.DataFrame, va: pd.DataFrame, epochs: int, device: str, log: dict) -> np.ndarray:
    feats = [
        c
        for c in tr.columns
        if c not in NON_FEATURES
        and not c.startswith(("cost_", "n_", "ult_", "comp_"))
        and tr[c].notna().any()
    ]
    cats = [c for c in feats if isinstance(tr[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in feats if c not in cats]
    rng = np.random.default_rng(0)
    hold = rng.random(len(tr)) < 0.05
    fit, es = tr[~hold], tr[hold]
    xn_f, xc_f = prep(fit, fit, nums, cats)
    xn_e, xc_e = prep(es, fit, nums, cats)
    xn_v, xc_v = prep(va, fit, nums, cats)
    card = [int(tr[c].cat.categories.size) + 1 for c in cats]

    def t(a, dt=torch.float32):
        return torch.as_tensor(a, dtype=dt, device=device)

    yf, wf = (fit["y"] / fit["exposure"]).to_numpy(np.float32), fit["exposure"].to_numpy(np.float32)
    ye, we = (es["y"] / es["exposure"]).to_numpy(np.float32), es["exposure"].to_numpy(np.float32)
    base = float(np.log(np.average(yf, weights=wf)))
    torch.manual_seed(0)
    model = tabm.TabM.make(n_num_features=xn_f.shape[1], cat_cardinalities=card, d_out=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    Xn, Xc, Y, W = t(xn_f), t(xc_f, torch.long), t(yf), t(wf)

    def predict_eta(xn, xc):
        model.eval()
        with torch.no_grad():
            out = [
                model(t(xn[i : i + 16384]), t(xc[i : i + 16384], torch.long)).squeeze(-1)
                for i in range(0, len(xn), 16384)
            ]
        model.train()
        return torch.cat(out) + base

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(len(Y), device=device)
        for i in range(0, len(Y), 2048):
            b = perm[i : i + 2048]
            loss = tweedie_loss(model(Xn[b], Xc[b]).squeeze(-1) + base, Y[b], W[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        es_loss = float(tweedie_loss(predict_eta(xn_e, xc_e), t(ye), t(we)))
        hist.append((ep + 1, round(time.time() - t0, 1), round(es_loss, 5)))
        print(f"   epoch {ep + 1}: {hist[-1][1]} s, holdout loss {es_loss:.5f}", flush=True)
        if es_loss < best - 1e-5:
            best, bad = es_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    rate = torch.exp(predict_eta(xn_v, xc_v)).mean(1).cpu().numpy()
    # level anchoring, as in the LightGBM experiment
    target_lc = tr["y"].sum() / tr["exposure"].sum()
    rate *= target_lc / np.average(rate, weights=va["exposure"])
    log["epochs"] = hist
    return rate * va["exposure"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="B", choices=["A", "B", "final"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lgbm-preds", default="")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = ultimate.load_config()
    if a.fold == "final":
        train_years, valid_year = cfg["harness"]["final"]["train"], cfg["predict_year"]
    else:
        f = cfg["harness"]["folds"][a.fold]
        train_years, valid_year = f["train"], f["valid"]
    tag = f"{a.fold}_{harness._fingerprint()}{'_sample' if a.sample else ''}"
    OUT.mkdir(parents=True, exist_ok=True)
    report, frames = {"device": device, "fold": a.fold, "sample": a.sample, "lines": {}}, []
    t_all = time.time()
    for line in ("motor", "buildings", "contents"):
        df = load_line(line)
        tr, va = harness.split(df, train_years, valid_year, cfg)
        if a.sample:
            tr = tr.sample(min(a.sample, len(tr)), random_state=0)
            va = va.sample(min(a.sample // 2, len(va)), random_state=0)
        print(f"{line}: {len(tr):,} train, {len(va):,} valid", flush=True)
        t0, log = time.time(), {}
        pred = fit_line(tr, va, a.epochs, device, log)
        log["minutes"] = round((time.time() - t0) / 60, 1)
        frames.append(va[["row_id", "line", "exposure"]].assign(y=va["y_ult"], pred=pred))
        report["lines"][line] = log
        del df, tr, va
    p = pd.concat(frames, ignore_index=True)
    p.to_parquet(OUT / f"tabm_{tag}.parquet")
    report["minutes_total"] = round((time.time() - t_all) / 60, 1)
    if a.fold == "final":
        (OUT / f"report_{tag}.json").write_text(json.dumps(report, indent=1, default=float))
        print(json.dumps(report, indent=1, default=float))
        return
    report["d2_tabm"] = score.d2(p["y"], p["pred"])
    for line, g in p.groupby("line"):
        report["lines"][line]["d2"] = score.d2(g["y"], g["pred"])
    if a.lgbm_preds:
        g = pd.read_parquet(a.lgbm_preds)
        g = g[g["fold"] == a.fold][["row_id", "pred"]].rename(columns={"pred": "lgbm"})
        m = p.merge(g, on="row_id", how="inner")
        report["d2_lgbm_same_rows"] = score.d2(m["y"], m["lgbm"])
        blends = {}
        for w in np.round(np.arange(0, 1.01, 0.1), 1):
            r = (1 - w) * m["lgbm"] + w * m["pred"]
            # re-anchor each line so the blend keeps the line levels
            r = r * m["line"].map(m.groupby("line")["lgbm"].sum() / r.groupby(m["line"]).sum())
            blends[float(w)] = score.d2(m["y"], r)
        report["blend_d2_by_tabm_weight"] = blends
    (OUT / f"report_{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "lines"}, indent=1, default=float))
    print(
        {k: {kk: vv for kk, vv in v.items() if kk != "epochs"} for k, v in report["lines"].items()}
    )


if __name__ == "__main__":
    main()
