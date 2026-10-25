"""Where the current model misses, for the loop agent to read before picking a hypothesis.

Reads the validation predictions of the run that sets today's bar (harness.best_kept) and
writes data/cache/diagnostics.md (gitignored, so it never dirties the tree):

- actual/expected by predicted-risk decile, per line and fold (calibration of the ranking)
- actual/expected by segment for every predictor of interest, per line, fold A and fold B
  side by side; a segment is flagged PERSISTENT when both folds miss in the same direction
  by more than 5% with at least 2% of the line's exposure. Persistent misses are the
  data-grounded hypotheses; one-fold misses are usually noise.

Usage: python -I scripts/diagnostics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iaa import data, harness, ultimate

OUT = data.ROOT / "data" / "cache" / "diagnostics.md"
SKIP = {
    "exposure",
    "row_id",
    "policy_id",
    "customer_id",
    "address_id",
    "period_start",
    "period_end",
    "split",
    "line",
    "year",
    "y",
    "y_ult",
    "trend_factor",
}


def _band(s: pd.Series) -> pd.Series:
    if s.notna().sum() == 0:  # e.g. claim history in the first emerging year
        return pd.Series("missing", index=s.index)
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.nunique() <= 8:
        s = s.astype("string").fillna("missing")
        top = s.value_counts().index[:10]
        return s.where(s.isin(top), "other")
    q = pd.qcut(s.rank(method="first"), 5, labels=False)
    edges = s.groupby(q).agg(["min", "max"])
    lab = {i: f"{r['min']:.4g}-{r['max']:.4g}" for i, r in edges.iterrows()}
    return q.map(lab).fillna("missing")


def main() -> None:
    best = harness.best_kept()
    run = best["run_id"]
    p = pd.read_parquet(harness.PREDS / f"{run}.parquet")
    # one 11.9m BI claim (2023) swamps any segment it lands in; cap actuals at 1m per row here
    p["y"] = p["y"].clip(upper=1_000_000)
    cfg = ultimate.load_config()
    folds = list(cfg["harness"]["folds"])[:2]
    F1, F2 = folds if len(folds) == 2 else (folds[0], folds[0])
    if cfg["harness"].get("metric") == "level_neutral":
        # Part 2: the keep metric is level-neutral, so show relativity misses only
        p = pd.concat([harness.level_neutral(g) for _, g in p.groupby("fold")])
    df = harness.dataset()
    df = df[df["split"] == "train"]
    lines = [f"# Diagnostics for the run to beat: {run} (primary {float(best['primary']):.5f})", ""]
    lines += [
        f"A/E = actual (capped at 1m per row) / predicted cost. Folds: {', '.join(f'{k} scores {v["valid"]}' for k, v in cfg['harness']['folds'].items())}. PERSISTENT = both folds miss the same way by >5% on >=2% of the line's exposure.",
        "",
    ]
    for line in ("motor", "buildings", "contents"):
        pl = p[p["line"] == line]
        feats = df[df["line"] == line].set_index("row_id")
        feats = feats[
            [
                c
                for c in feats.columns
                if c not in SKIP
                and not c.startswith(harness.TARGET_COLS_PREFIX)
                and feats[c].notna().any()
            ]
        ]
        lines += [f"## {line.title()}", "", "Decile of predicted rate: A/E by fold", ""]
        dec = []
        for fold in (F1, F2):
            g = pl[pl["fold"] == fold].copy()
            g["d"] = pd.qcut((g["pred"] / g["exposure"]).rank(method="first"), 10, labels=False) + 1
            dec.append(
                g.groupby("d").apply(lambda x: x["y"].sum() / x["pred"].sum(), include_groups=False)
            )
        lines.append("| decile | " + " | ".join(str(i) for i in range(1, 11)) + " |")
        lines.append("|---" * 11 + "|")
        for fold, d in zip((F1, F2), dec, strict=True):
            lines.append(f"| {fold} | " + " | ".join(f"{v:.2f}" for v in d) + " |")
        lines += [
            "",
            "Segments (worst persistent misses first):",
            "",
            f"| field | level | exposure share | A/E fold {F1} | A/E fold {F2} | flag |",
            "|---|---|---|---|---|---|",
        ]
        rows = []
        for c in feats.columns:
            res = {}
            for fold in (F1, F2):
                g = pl[pl["fold"] == fold].join(feats[[c]], on="row_id")
                g["lvl"] = _band(g[c])
                agg = g.groupby("lvl", observed=True).agg(
                    y=("y", "sum"), pred=("pred", "sum"), e=("exposure", "sum")
                )
                agg["ae"] = agg["y"] / agg["pred"]
                agg["share"] = agg["e"] / agg["e"].sum()
                res[fold] = agg
            both = res[F1][["ae", "share"]].join(
                res[F2][["ae", "share"]], lsuffix="_A", rsuffix="_B", how="inner"
            )
            for lvl, r in both.iterrows():
                share = min(r["share_A"], r["share_B"])
                same = np.sign(r["ae_A"] - 1) == np.sign(r["ae_B"] - 1)
                big = min(abs(r["ae_A"] - 1), abs(r["ae_B"] - 1)) > 0.05
                flag = "PERSISTENT" if same and big and share >= 0.02 else ""
                rows.append(
                    (
                        flag,
                        min(abs(r["ae_A"] - 1), abs(r["ae_B"] - 1)) if flag else 0,
                        c,
                        lvl,
                        share,
                        r["ae_A"],
                        r["ae_B"],
                    )
                )
        rows.sort(key=lambda r: (r[0] != "PERSISTENT", -r[1]))
        for flag, _, c, lvl, share, a, b in rows[:25]:
            lines.append(f"| {c} | {lvl} | {share:.1%} | {a:.3f} | {b:.3f} | {flag} |")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
