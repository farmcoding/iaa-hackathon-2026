"""Diagnostic figures for the Part 2 report (report v2, 28 Oct).

Reads saved fold predictions only (no dataset load, no refit):

- the ledger runs P0, L1 and O4 (data/cache/preds/<run_id>.parquet), and
- the selected relativity model with its 2023-24 trends (data/cache/trend_bench_folds.parquet,
  written by `scripts/part2_trend_bench.py contents=0.0 motor_bi=0.12`).

Fold B trains 2023 and scores 2024; fold R trains 2024 and scores 2023.

    .venv/bin/python -I scripts/part2_report_figures.py

Writes reports/figures/part2_lift.png and reports/figures/part2_steps_ae.png, and prints
the numbers behind them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "reports" / "figures"
POWER = 1.5

# reference palette (dataviz skill, light mode, validated): fold B, fold R; ink for the rest
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LINES = ("motor", "buildings", "contents")
STEPS = [  # label, prediction file, public board score
    ("P0\nper-peril\nbase", CACHE / "preds" / "20261025T111149.parquet", 0.15195),
    ("L1\n+ motor\ngender", CACHE / "preds" / "20261025T111615.parquet", 0.15933),
    ("O4\n+ emerging\nlevels", CACHE / "preds" / "20261025T125844.parquet", 0.16326),
    ("Selected\nO4 +\ntrends", CACHE / "trend_bench_folds.parquet", 0.16621),
]


def deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    p = POWER
    return 2 * (
        np.power(y, 2 - p) / ((1 - p) * (2 - p))
        - y * np.power(mu, 1 - p) / (1 - p)
        + np.power(mu, 2 - p) / (2 - p)
    )


def d2(y: np.ndarray, mu: np.ndarray) -> float:
    return float(1 - deviance(y, mu).sum() / deviance(y, np.full_like(y, y.mean())).sum())


def level_neutral(f: pd.DataFrame) -> pd.DataFrame:
    """Each line rescaled to its actual total within the fold: ranking and spread only."""
    f = f.copy()
    for _, g in f.groupby(["fold", "line"]):
        f.loc[g.index, "pred"] = g["pred"] * g["y"].sum() / g["pred"].sum()
    return f


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7, length=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_lift(sel: pd.DataFrame) -> None:
    f = level_neutral(sel)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    for ax, line in zip(axes, LINES, strict=True):
        g = f[f.line == line].copy()
        g["rate"] = g["pred"] / g["exposure"]
        rows = []
        for fold, h in g.groupby("fold"):
            h = h.sort_values("rate")
            cum = h["exposure"].cumsum() / h["exposure"].sum()
            h["dec"] = np.minimum((cum * 10).apply(np.ceil).astype(int), 10)
            mean = h["y"].sum() / h["exposure"].sum()
            t = h.groupby("dec")[["y", "pred", "exposure"]].sum()
            rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "dec": t.index.to_numpy(),
                        "act": t.y / t.exposure / mean,
                        "pred": t.pred / t.exposure / mean,
                    }
                )
            )
        r = pd.concat(rows, ignore_index=True)
        pr = r.groupby("dec")["pred"].mean()
        ax.plot(pr.index, pr.values, color=INK, linewidth=1.6, label="Predicted")
        for fold, col, mk in (("B", BLUE, "o"), ("R", ORANGE, "s")):
            q = r[r.fold == fold]
            ax.plot(q.dec, q.act, mk, color=col, markersize=4, label=f"Actual, fold {fold}")
        ax.set_title(line.capitalize(), fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Predicted risk decile", fontsize=7, color=INK2)
        style(ax)
        print(f"lift {line}: top/bottom predicted {pr.iloc[-1] / pr.iloc[0]:.1f}x", end="  ")
        for fold in ("B", "R"):
            q = r[r.fold == fold].set_index("dec")["act"]
            print(f"actual {fold} {q.iloc[-1] / q.iloc[0]:.1f}x", end="  ")
        print()
    axes[0].set_ylabel("Loss cost / average", fontsize=7, color=INK2)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "part2_lift.png", dpi=220)
    plt.close(fig)


def fig_steps_ae(sel: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: level-neutral D2 by fold for each step, public board beside it
    xs = np.arange(len(STEPS))
    scores = []
    for label, path, pub in STEPS:
        f = level_neutral(pd.read_parquet(path))
        b = d2(f[f.fold == "B"].y.to_numpy(), f[f.fold == "B"].pred.to_numpy())
        r = d2(f[f.fold == "R"].y.to_numpy(), f[f.fold == "R"].pred.to_numpy())
        scores.append((b, r, pub))
        print(f"step {label.replace(chr(10), ' ')}: fold B {b:.4f} fold R {r:.4f} public {pub}")
    s = np.array(scores)
    for i, (name, col, mk) in enumerate(
        (("Fold B", BLUE, "o"), ("Fold R", ORANGE, "s"), ("Public board", INK2, "D"))
    ):
        a1.plot(xs, s[:, i], marker=mk, color=col, linewidth=1.4, markersize=4, label=name)
        a1.annotate(
            f"{s[-1, i]:.3f}",
            (xs[-1], s[-1, i]),
            xytext=(5, -2),
            textcoords="offset points",
            fontsize=6.5,
            color=INK2,
        )
    a1.set_xticks(xs, [lbl for lbl, _, _ in STEPS], fontsize=6)
    a1.set_xlim(-0.3, len(STEPS) - 0.45)
    a1.set_ylabel("Tweedie D²", fontsize=7, color=INK2)
    a1.set_title("Score by model step", fontsize=8.5, color=INK, loc="left")
    a1.legend(fontsize=6.5, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.4), ncol=3)
    style(a1)
    # right: actual / expected at the production level, line x new business, per fold
    groups = [(ln, nb) for ln in LINES for nb in (False, True)]
    ys = np.arange(len(groups))[::-1]
    for fold, col, mk, dy in (("B", BLUE, "o", 0.12), ("R", ORANGE, "s", -0.12)):
        vals = []
        for ln, nb in groups:
            g = sel[(sel.fold == fold) & (sel.line == ln) & (sel.nb == nb)]
            vals.append(g.y.sum() / g.pred.sum())
        a2.plot(np.minimum(vals, 2.0), ys + dy, mk, color=col, markersize=4, label=f"Fold {fold}")
        for v, yy in zip(vals, ys + dy, strict=True):
            if v > 2.0:
                a2.annotate(
                    f"{v:.2f}",
                    (2.0, yy),
                    xytext=(-22, -2),
                    textcoords="offset points",
                    fontsize=6,
                    color=INK2,
                )
        print(
            f"A/E fold {fold}:",
            {
                f"{ln} {'NB' if nb else 'renewal'}": round(v, 2)
                for (ln, nb), v in zip(groups, vals, strict=True)
            },
        )
    a2.axvline(1.0, color=MUTED, linewidth=0.8)
    a2.set_yticks(
        ys,
        [f"{ln.capitalize()}, {'new business' if nb else 'renewal'}" for ln, nb in groups],
        fontsize=6.5,
    )
    a2.set_xlim(0.3, 2.05)
    a2.set_xlabel("Actual / expected", fontsize=7, color=INK2)
    a2.set_title("Actual / expected by fold", fontsize=8.5, color=INK, loc="left")
    a2.legend(fontsize=6.5, frameon=False, loc="lower right")
    style(a2)
    a2.grid(axis="x", color=GRID, linewidth=0.6)
    a2.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "part2_steps_ae.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_parquet(STEPS[-1][1])
    fig_lift(sel)
    fig_steps_ae(sel)


if __name__ == "__main__":
    main()
