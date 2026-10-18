"""Scoring, matched to the Kaggle leaderboard metric.

FROZEN: the experiment agent must not edit this file.

The leaderboard uses Tweedie D² with power 1.5, "relative to a constant-rate baseline".
Two readings are implemented until the calibration submission settles which one the
host uses:

- "constant": baseline is the single mean of y over all rows (sklearn d2_tweedie_score)
- "rate": baseline is one claim rate per unit of exposure, times each row's exposure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POWER = 1.5
FLOOR = 1e-3  # predictions of exactly zero give infinite deviance when y > 0


def tweedie_deviance(y: np.ndarray, mu: np.ndarray, power: float = POWER) -> np.ndarray:
    """Unit Tweedie deviance per row, for 1 < power < 2."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), FLOOR)
    p = power
    term_y = np.where(y > 0, np.power(y, 2 - p) / ((1 - p) * (2 - p)), 0.0)
    return 2 * (term_y - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def d2(
    y: np.ndarray,
    pred: np.ndarray,
    exposure: np.ndarray | None = None,
    baseline: str = "constant",
    power: float = POWER,
) -> float:
    y = np.asarray(y, dtype=float)
    if baseline == "constant":
        base = np.full_like(y, y.mean())
    elif baseline == "rate":
        if exposure is None:
            raise ValueError("rate baseline needs exposure")
        exposure = np.asarray(exposure, dtype=float)
        base = exposure * y.sum() / exposure.sum()
    else:
        raise ValueError(baseline)
    return 1.0 - tweedie_deviance(y, pred, power).sum() / tweedie_deviance(y, base, power).sum()


def scorecard(df: pd.DataFrame, pred_col: str = "pred", y_col: str = "y") -> dict[str, float]:
    """Pooled and per-line metrics for one validation fold.

    df needs columns line, exposure, y and pred.
    """
    out: dict[str, float] = {}
    y, p, e = df[y_col].to_numpy(), df[pred_col].to_numpy(), df["exposure"].to_numpy()
    out["d2"] = d2(y, p)
    out["d2_rate"] = d2(y, p, e, baseline="rate")
    out["ae"] = y.sum() / p.sum()
    out["gini"] = gini(y, p, e)
    for line, g in df.groupby("line"):
        out[f"d2_{line}"] = d2(g[y_col], g[pred_col])
        out[f"ae_{line}"] = g[y_col].sum() / g[pred_col].sum()
    return out


def gini(y: np.ndarray, pred: np.ndarray, exposure: np.ndarray) -> float:
    """Exposure-weighted Gini of the loss-cost ranking (ordered Lorenz curve)."""
    rate = pred / np.maximum(exposure, 1e-9)
    order = np.argsort(rate)
    e = np.cumsum(exposure[order]) / exposure.sum()
    loss = np.cumsum(np.asarray(y)[order]) / np.sum(y)
    return float(1 - 2 * np.trapezoid(loss, e))
