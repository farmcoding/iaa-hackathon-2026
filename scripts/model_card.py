"""Owner script: what the current model uses, and how much each predictor adds.

Fits the current experiment once on fold B (train 2021-23, score 2024), capturing every
fitted sub-model and its scoring inputs. Each predictor is then shuffled across the 2024
rows and the line's predictions rebuilt through the same sub-models, blend and level
anchoring. The drop in within-line D² is the predictor's contribution to ranking (the
level is anchored, so shuffling cannot move it). High-cardinality fields enter as
credibility encodings; shuffling the encoded column is the same as shuffling the field.

Also lists every available column that the model does not use, with the reason, and the
EBM's own term importances (including the pairwise interactions it found).

Writes data/cache/model_card.json (gitignored, so it can run while the loop holds the
tree); copy it to reports/features/ when the loop is stopped.

Usage: python -I scripts/model_card.py [--module iaa.experiment] [--repeats 1]
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import types

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor

from iaa import data, harness, score, ultimate

OUT = data.ROOT / "data" / "cache" / "model_card.json"

REASONS = {
    "row_id": "Identifier",
    "policy_id": "Identifier",
    "customer_id": "Identifier",
    "address_id": "Identifier",
    "period_start": "Date; enters through start month and ages",
    "period_end": "Date; exposure carries the period length",
    "split": "Structural",
    "line": "Structural; each line has its own model",
    "year": "Calendar year: trees cannot extrapolate it; trend is set by on-levelling",
    "exposure": "Weight and level base, not a predictor (as a feature, run 40: -0.003)",
    "trend_factor": "Derived from year (on-levelling)",
    "y": "Target",
    "y_ult": "Target",
    "lines_held": "Tested (lines a customer holds): no gain, 0.1334 vs 0.1339",
    "email_domain": "Owner decision 19 Oct: inadmissible and pure noise (no relation to any other field)",
    "written_premium": "Owner decision 19 Oct: the incumbent's price, not a risk characteristic",
    "licence_points": "Owner decision 19 Oct: a driving record, no rating rationale for property",
}
PREFIX_REASON = "Claims outcome for the row (target side), hidden from the model"
BEFORE_DATASET = {
    "renewal_decision": "Leak: recorded after the period; training data only",
    "customer_service_contacts": "Leak: includes claim-related contacts in training data",
    "customer_since_year": "Drifts between training and 2025 by construction; tenure replaces it",
    "exposure_raw": "Data-quality working column (exposure in days fixed)",
    "dq_exposure_fixed": "Data-quality flag",
    "date_of_birth": "Replaced by customer age at period start",
    "surname": "Personal identifier; never used",
}


class _Rec:
    def __init__(self) -> None:
        self.events: list[tuple] = []


def _instrument(exp, rec: _Rec) -> None:
    class LGBM(lgb.LGBMRegressor):
        def predict(self, X, *a, **k):  # noqa: N803
            rec.events.append(("lgbm", self, X))
            return super().predict(X, *a, **k)

    class EBM(ExplainableBoostingRegressor):
        def predict(self, X, *a, **k):  # noqa: N803
            rec.events.append(("ebm", self, X))
            return super().predict(X, *a, **k)

    class CB(CatBoostRegressor):
        def predict(self, X, *a, **k):  # noqa: N803
            rec.events.append(("cb", self, X))
            return super().predict(X, *a, **k)

    shim = types.SimpleNamespace(**{k: getattr(lgb, k) for k in dir(lgb) if not k.startswith("__")})
    shim.LGBMRegressor = LGBM
    exp.lgb = shim
    exp.ExplainableBoostingRegressor = EBM
    exp.CatBoostRegressor = CB
    orig = exp._anchor

    def anchor(rate, target_lc, va):
        rec.events.append(("anchor", float(target_lc)))
        return orig(rate, target_lc, va)

    exp._anchor = anchor


def _anchor(rate: np.ndarray, lc: float, w: np.ndarray) -> np.ndarray:
    return rate * lc / np.average(rate, weights=w)


def _segments(events: list[tuple]) -> dict[str, list[tuple]]:
    # fit_predict runs motor (two anchors), then buildings and contents (one each)
    out, cur, need = {}, [], {"motor": 2, "buildings": 1, "contents": 1}
    lines = iter(need)
    line = next(lines)
    for e in events:
        cur.append(e)
        if e[0] == "anchor" and sum(x[0] == "anchor" for x in cur) == need[line]:
            out[line], cur = cur, []
            line = next(lines, None)
    return out


def _rebuild(line: str, seg: list[tuple], w: np.ndarray, perm: dict | None = None) -> dict:
    """Predicted rate per row, optionally with columns shuffled in some sub-models.
    perm: {"cols": [...], "idx": array, "parts": {"bi", "other", "prop"}}"""

    def pred(kind, model, X, part):
        if perm and part in perm["parts"]:
            X = X.copy()
            for c in perm["cols"]:
                if c in X.columns:
                    X[c] = X[c].iloc[perm["idx"]].set_axis(X.index)
        if kind == "lgbm":
            return lgb.LGBMRegressor.predict(model, X)
        if kind == "ebm":
            return ExplainableBoostingRegressor.predict(model, X)
        return CatBoostRegressor.predict(model, X)

    lg = [e for e in seg if e[0] == "lgbm"]
    eb = [e for e in seg if e[0] == "ebm"]
    cb = [e for e in seg if e[0] == "cb"]
    if line == "motor":
        # BI: LightGBM Poisson seeds; other perils: LightGBM Tweedie seeds averaged 50/50 with an EBM
        an = [e[1] for e in seg if e[0] == "anchor"]
        k = len(lg) // 2
        bi = np.mean([pred("lgbm", m, X, "bi") for _, m, X in lg[:k]], axis=0)
        ot = np.mean([pred("lgbm", m, X, "other") for _, m, X in lg[k:]], axis=0)
        if eb:
            ot = 0.5 * ot + 0.5 * pred("ebm", eb[0][1], eb[0][2], "other")
        return _anchor(bi, an[0], w) + _anchor(ot, an[1], w)
    # property: LightGBM seeds, EBM and CatBoost, equal weights
    lc = [e[1] for e in seg if e[0] == "anchor"][0]
    members = [np.mean([pred("lgbm", m, X, "prop") for _, m, X in lg], axis=0)]
    members += [pred("ebm", e[1], e[2], "prop") for e in eb]
    members += [pred("cb", e[1], e[2], "prop") for e in cb]
    return _anchor(np.mean(members, axis=0), lc, w)


def _unused(df: pd.DataFrame, used: dict[str, list[str]]) -> list[dict]:
    rows = []
    for c in df.columns:
        lines_present = [ln for ln in used if df.loc[df["line"] == ln, c].notna().any()]
        unused_on = [ln for ln in lines_present if c not in used[ln]]
        if not unused_on:
            continue
        if c.startswith(harness.TARGET_COLS_PREFIX):
            reason = PREFIX_REASON
        elif c in REASONS:
            reason = REASONS[c]
        else:
            reason = "Not used (no rule found; check)"
        rows.append({"column": c, "lines": unused_on, "reason": reason})
    for c, reason in BEFORE_DATASET.items():
        rows.append({"column": c, "lines": ["removed before modelling"], "reason": reason})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="iaa.experiment")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    cfg = ultimate.load_config()
    df = harness.dataset()
    f = cfg["harness"]["folds"]["B"]
    tr, va = harness.split(df, f["train"], f["valid"], cfg)
    exp = importlib.import_module(a.module)
    exp.FROZEN_LINES = ()  # refit every line: frozen lines reuse saved predictions and would not be captured
    rec = _Rec()
    _instrument(exp, rec)
    pred = exp.fit_predict(tr, harness.hide_target(va))
    segs = _segments(rec.events)
    print(f"fitted in {time.time() - t0:.0f} s", flush=True)

    card: dict = {"fold": "B", "train": f["train"], "valid": f["valid"], "lines": {}}
    card["pooled_d2"] = score.d2(va["y_ult"].to_numpy(), pred)
    used: dict[str, list[str]] = {}
    rng = np.random.default_rng(0)
    for line, seg in segs.items():
        m = (va["line"] == line).to_numpy()
        y, w = va.loc[m, "y_ult"].to_numpy(), va.loc[m, "exposure"].to_numpy()
        lgbm_X = next(e[2] for e in seg if e[0] == "lgbm")
        cols = list(lgbm_X.columns)
        used[line] = cols
        base_rate = _rebuild(line, seg, w)
        base = score.d2(y, base_rate * w)
        assert np.allclose(base_rate * w, pred[m], rtol=1e-6), (
            f"{line}: rebuild differs from the model"
        )
        base_gini = score.gini(y, base_rate * w, w)
        parts = [{"bi", "other"}] if line == "motor" else [{"prop"}]
        rows = []
        for c in cols:
            drops, gdrops, split = [], [], {}
            for _ in range(a.repeats):
                idx = rng.permutation(m.sum())
                r = _rebuild(line, seg, w, {"cols": [c], "idx": idx, "parts": parts[0]})
                drops.append(base - score.d2(y, r * w))
                gdrops.append(base_gini - score.gini(y, r * w, w))
                if line == "motor":
                    for p in ("bi", "other"):
                        rp = _rebuild(line, seg, w, {"cols": [c], "idx": idx, "parts": {p}})
                        split.setdefault(p, []).append(base - score.d2(y, rp * w))
            rows.append(
                {
                    "feature": c,
                    "d2_drop": float(np.mean(drops)),
                    "gini_drop": float(np.mean(gdrops)),
                    "encoded": bool(
                        c in tr.columns
                        and isinstance(tr[c].dtype, pd.CategoricalDtype)
                        and pd.api.types.is_float_dtype(lgbm_X[c].dtype)
                    ),
                    **{f"d2_drop_{p}": float(np.mean(v)) for p, v in split.items()},
                }
            )
        rows.sort(key=lambda r: -r["d2_drop"])
        info = {"d2": base, "gini": base_gini, "n_features": len(cols), "features": rows}
        eb = [e for e in seg if e[0] == "ebm"]
        if eb:
            model = eb[0][1]
            imp = model.term_importances()
            info["ebm_terms"] = sorted(
                (
                    {"term": n, "importance": float(v)}
                    for n, v in zip(model.term_names_, imp, strict=True)
                ),
                key=lambda t: -t["importance"],
            )
        n_lgbm = sum(e[0] == "lgbm" for e in seg)
        info["components"] = (
            f"BI claim frequency: LightGBM Poisson ({n_lgbm // 2} seeds); other perils: LightGBM Tweedie ({n_lgbm // 2} seeds) averaged 50/50 with an EBM"
            if line == "motor"
            else f"LightGBM Tweedie ({n_lgbm} seeds, rounds by early stopping), an EBM and CatBoost Tweedie, equal weights"
        )
        card["lines"][line] = info
        print(f"{line}: D2 {base:.4f}, {len(cols)} features, {time.time() - t0:.0f} s", flush=True)

    for ln, cols in getattr(exp, "PRUNED", {}).items():
        for c in cols:
            REASONS.setdefault(
                c,
                "Pruned 19 Oct (owner-approved): no measurable contribution on that line, or a sales/admin field",
            )
    card["unused"] = _unused(df, used)
    card["seconds"] = round(time.time() - t0)
    card["dataset"] = harness._fingerprint()
    OUT.write_text(json.dumps(card, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
