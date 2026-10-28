"""O5 target sensitivity (owner, 27 Oct): how far each target assumption moves the 2025 level.

The target is a frozen owner assumption, so a target variant cannot be judged by the keep
rule: folds B and R would score against a different truth. Under `level: balance_on_train`
each line's 2025 prediction sums to its trended training actuals, so a target change
passes through to the 2025 line level almost one for one. This script recomputes the
developed targets under each variant (policy rows and claims only, no feature build, no
dataset cache) and reports the implied 2025 level per line against v1.

Decision rule, declared before the numbers were seen: a variant that moves a line's 2025
level by less than 2% is a sensitivity row in reports/assumptions_log.md, with no refit
and no submission (Part 1 level probes: motor +/-10% moved the private score about
0.0013-0.0027; board noise between different models is about 0.002). A larger move is a
track question: evidence first, then one harness --final refit of O4 on that target and
at most one board submission (SIGNOFF.md two-track amendment).

    IAA_MARKET=emerging .venv/bin/python -I scripts/part2_o5_levels.py
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from iaa import data, target, ultimate

PRED_YEAR = 2025
_claims = data.claims  # the cached loader, kept before any patch


def patched_claims(fn):
    """Point the loaders at a modified claims table: fn(claims) -> claims."""

    def claims(market: str | None = None) -> pd.DataFrame:
        return fn(_claims(market).copy())

    return claims


def scale_open(perils: set[str] | None, factor: float):
    """Case reserves on open claims understated or overstated: net x factor (both the
    borrowed pattern and the emerging rows)."""

    def fn(c: pd.DataFrame) -> pd.DataFrame:
        m = c["status"].eq("open")
        if perils is not None:
            m &= c["peril"].isin(perils)
        c.loc[m, "net"] = c.loc[m, "net"] * factor
        return c

    return fn


def od_open_recoveries(c: pd.DataFrame) -> pd.DataFrame:
    """Open own-damage claims recover at the closed claims' ratio by the end."""
    od = c["peril"].eq("own_damage") & c["line"].eq("motor")
    closed, opn = od & c["status"].eq("closed"), od & c["status"].eq("open")
    r_closed = c.loc[closed, "recoveries"].sum() / c.loc[closed, "incurred_to_date"].sum()
    gap = np.maximum(r_closed * c.loc[opn, "incurred_to_date"] - c.loc[opn, "recoveries"], 0.0)
    c.loc[opn, "net"] = np.maximum(c.loc[opn, "net"] - gap, 0.0)
    return c


def developed(cfg: dict, claims_fn=None) -> pd.DataFrame:
    if claims_fn is not None:
        data.claims = claims_fn
    try:
        rows, _, _ = target.reported_targets()
        return ultimate.developed_targets(rows, cfg)
    finally:
        data.claims = _claims


def levels(dev: pd.DataFrame, cfg: dict, bi_scale: float = 1.0) -> pd.DataFrame:
    """Per line: developed cost by year and the trended 2023+2024 total (the 2025 level)."""
    out = []
    peril_trend = cfg.get("peril_trend") or {}
    for line, groups in cfg["development"]["peril_groups"].items():
        r = dev[dev["line"] == line]
        gap = PRED_YEAR - r["year"].to_numpy()
        tot = np.zeros(len(r))
        trended = np.zeros(len(r))
        for grp in groups:
            u = r[f"ult_{grp}"].fillna(0).to_numpy() * (bi_scale if grp == "bodily_injury" else 1.0)
            rate = peril_trend.get(line, {}).get(grp, cfg["trend"][line])
            tot += u
            trended += u * (1 + rate) ** gap
        rec = {"line": line, "level_2025": trended.sum()}
        for yr in sorted(r["year"].unique()):
            rec[f"ult_{yr}"] = tot[r["year"].to_numpy() == yr].sum()
        if line == "motor":
            bi = r["ult_bodily_injury"].fillna(0).to_numpy() * bi_scale
            rec["bi_share"] = bi.sum() / tot.sum()
        out.append(rec)
    return pd.DataFrame(out).set_index("line")


def main() -> None:
    cfg = ultimate.load_config()
    assert cfg["market"] == "emerging", "run with IAA_MARKET=emerging"
    fs = copy.deepcopy(cfg)
    fs["development"]["freq_sev"] = {"motor.bodily_injury": {"base_year": 2023, "sev_trend": 0.035}}
    base = developed(cfg)
    variants = {
        "v1 (current)": levels(base, cfg),
        "BI tail +5%": levels(base, cfg, 1.05),
        "BI tail -5%": levels(base, cfg, 0.95),
        "open BI case reserves +10%": levels(
            developed(cfg, patched_claims(scale_open({"bodily_injury"}, 1.10))), cfg
        ),
        "open BI case reserves -10%": levels(
            developed(cfg, patched_claims(scale_open({"bodily_injury"}, 0.90))), cfg
        ),
        "all open case reserves +10%": levels(
            developed(cfg, patched_claims(scale_open(None, 1.10))), cfg
        ),
        "all open case reserves -10%": levels(
            developed(cfg, patched_claims(scale_open(None, 0.90))), cfg
        ),
        "open OD recoveries at closed ratio": levels(
            developed(cfg, patched_claims(od_open_recoveries)), cfg
        ),
        "BI count x severity, base 2023, 3.5%": levels(developed(fs), fs),
    }
    v1 = variants["v1 (current)"]
    rows = []
    for name, lv in variants.items():
        rec = {"variant": name}
        for line in lv.index:
            rec[f"{line}_2025"] = lv.loc[line, "level_2025"] / v1.loc[line, "level_2025"]
        rec["motor_2023"] = lv.loc["motor", "ult_2023"] / v1.loc["motor", "ult_2023"]
        rec["motor_2024"] = lv.loc["motor", "ult_2024"] / v1.loc["motor", "ult_2024"]
        rec["motor_bi_share"] = lv.loc["motor", "bi_share"]
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200)
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
