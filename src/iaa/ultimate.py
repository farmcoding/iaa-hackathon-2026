"""Develop reported claims to ultimate and on-level to the prediction year.

FROZEN: the experiment agent must not edit this file.

Development: chain ladder on reported net cost, arranged by loss month and report
delay in months, one triangle per (line, peril group). This captures IBNR, including
the tendency of late-reported claims to be larger. Open-claim case estimates are taken
as unbiased (single valuation date, so IBNER cannot be estimated; it is an assumption).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from iaa import data

CONFIG = data.ROOT / "configs" / "market_dev.yaml"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def peril_group_map(cfg: dict) -> dict[tuple[str, str], str]:
    out = {}
    for line, groups in cfg["development"]["peril_groups"].items():
        for grp, perils in groups.items():
            for p in perils:
                out[(line, p)] = grp
    return out


def _months(d: pd.Series) -> pd.Series:
    return d.dt.year * 12 + d.dt.month - 1


def bi_cap(cfg: dict) -> float | None:
    """Per-claim cap on motor bodily injury (large_loss.caps), or None."""
    return ((cfg.get("large_loss") or {}).get("caps") or {}).get("motor_bodily_injury")


def capped_claims(cfg: dict) -> pd.DataFrame:
    """data.claims() with motor bodily-injury net cost capped per claim at large_loss.caps.
    The excess above the cap is carried separately (target.reported_targets, ult_bi_excess)
    and reloaded as an expected large-loss load in the model's level, not in the target."""
    clm = data.claims()
    cap = bi_cap(cfg)
    if not cap:
        return clm
    perils = cfg["development"]["peril_groups"]["motor"]["bodily_injury"]
    m = (clm["line"] == "motor") & clm["peril"].isin(perils)
    clm = clm.copy()
    clm.loc[m, "net"] = clm.loc[m, "net"].clip(upper=cap)
    return clm


def delay_cdfs(claims: pd.DataFrame, cfg: dict, value: str = "amt") -> pd.DataFrame:
    """Cumulative development pattern (share of ultimate reported) by delay month.

    Returns a frame indexed by (line, group) with columns 0..max_delay: the fraction of
    ultimate net cost reported within that many months of the loss month.
    """
    val_m = _months(pd.Series([pd.Timestamp(cfg["valuation_date"])])).iloc[0]
    max_d = cfg["development"]["max_delay_months"]
    gmap = peril_group_map(cfg)
    c = claims[(claims["status"] != "repudiated") & (claims["ay"] >= 2021)].copy()
    c["group"] = [gmap[(ln, p)] for ln, p in zip(c["line"], c["peril"], strict=True)]
    c["lm"] = _months(c["loss_date"])
    c["delay"] = (_months(c["report_date"]) - c["lm"]).clip(upper=max_d)
    c["ibner"] = np.where(c["status"] == "open", cfg["development"]["ibner_open_factor"], 1.0)
    c["amt"] = c["net"] * c["ibner"]
    c["count"] = 1.0
    rows = {}
    for (line, grp), g in c.groupby(["line", "group"]):
        tri = g.pivot_table(index="lm", columns="delay", values=value, aggfunc="sum")
        tri = tri.reindex(columns=range(max_d + 1), fill_value=0).fillna(0).cumsum(axis=1)
        age = val_m - tri.index.to_numpy()  # last observed delay per loss month
        f = np.ones(max_d)
        for d in range(max_d):
            ok = age >= d + 1  # loss months observed at both d and d+1
            num, den = tri.loc[ok, d + 1].sum(), tri.loc[ok, d].sum()
            f[d] = num / den if den > 0 else 1.0
        # cumulative: share reported by delay d = 1 / prod(f[d:])
        cdf_to_ult = np.cumprod(f[::-1])[::-1]
        rows[(line, grp)] = np.append(1.0 / cdf_to_ult, 1.0)
    return pd.DataFrame.from_dict(rows, orient="index", columns=range(max_d + 1))


def completion_by_month(cdfs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Share of ultimate reported at the valuation date, by (line, group, loss month)."""
    val_m = _months(pd.Series([pd.Timestamp(cfg["valuation_date"])])).iloc[0]
    max_d = cfg["development"]["max_delay_months"]
    out = []
    for (line, grp), cdf in cdfs.iterrows():
        for lm in range(2021 * 12, val_m + 1):
            out.append((line, grp, lm, cdf[min(val_m - lm, max_d)]))
    return pd.DataFrame(out, columns=["line", "group", "lm", "completion"])


def row_completion(rows: pd.DataFrame, comp: pd.DataFrame, line: str, grp: str) -> np.ndarray:
    """Exposure-weighted average completion over each row's period (by month)."""
    table = comp[(comp["line"] == line) & (comp["group"] == grp)].set_index("lm")["completion"]
    sm, em = _months(rows["period_start"]).to_numpy(), _months(rows["period_end"]).to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(table.to_numpy())])
    base = table.index.min()
    lo = np.clip(sm - base, 0, len(table) - 1)
    hi = np.clip(em - base, 0, len(table) - 1)
    return (cum[hi + 1] - cum[lo]) / (hi - lo + 1)


def freq_sev_adjustment(rows: pd.DataFrame, cfg: dict) -> dict[tuple[str, str, int], float]:
    """Year-level rescaling for groups developed by frequency x severity.

    Ultimate count per year comes from a chain ladder on reported counts. Ultimate
    severity is the base cohort's mean net cost trended to each year. The returned
    factor rescales the cost-chain-ladder ultimate of that (line, group, year) to the
    frequency-severity ultimate.
    """
    out: dict[tuple[str, str, int], float] = {}
    fs = cfg["development"].get("freq_sev", {}) or {}
    if not fs:
        return out
    clm = capped_claims(cfg)
    ccdf = completion_by_month(delay_cdfs(clm, cfg, value="count"), cfg)
    acdf = completion_by_month(delay_cdfs(clm, cfg, value="amt"), cfg)
    gmap = peril_group_map(cfg)
    for key, spec in fs.items():
        line, grp = key.split(".")
        perils = cfg["development"]["peril_groups"][line][grp]
        c = clm[(clm["line"] == line) & clm["peril"].isin(perils) & (clm["status"] != "repudiated")]
        base = c[c["ay"] == spec["base_year"]]["net"].mean()
        m = rows["line"] == line
        r = rows.loc[m]
        cnt_cols = [f"n_{p}" for p in perils if f"n_{p}" in r.columns]
        cost_cols = [f"cost_{p}" for p in perils if f"cost_{p}" in r.columns]
        cc = row_completion(r, ccdf, line, grp)
        ac = row_completion(r, acdf, line, grp)
        for yr in sorted(r["year"].unique()):
            k = (r["year"] == yr).to_numpy()
            ult_n = (r.loc[k, cnt_cols].sum(axis=1).to_numpy() / cc[k]).sum()
            sev = base * (1 + spec["sev_trend"]) ** (yr - spec["base_year"])
            cl_ult = (r.loc[k, cost_cols].sum(axis=1).to_numpy() / ac[k]).sum()
            out[(line, grp, int(yr))] = (
                max(1.0, ult_n * sev / cl_ult) if yr > spec["base_year"] else 1.0
            )
        del gmap
    return out


def developed_targets(rows: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add y_ult (developed net cost) and per-group developed costs to training rows.

    rows comes from target.reported_targets(). Group cost / row completion gives an
    unbiased (per group) estimate of the row's ultimate cost; groups listed under
    development.freq_sev are then rescaled by year to the frequency-severity ultimate.
    """
    cfg = cfg or load_config()
    cdfs = delay_cdfs(capped_claims(cfg), cfg)
    comp = completion_by_month(cdfs, cfg)
    adj = freq_sev_adjustment(rows, cfg)
    rows = rows.copy()
    rows["y_ult"] = 0.0
    for line, groups in cfg["development"]["peril_groups"].items():
        m = rows["line"] == line
        for grp, perils in groups.items():
            cols = [f"cost_{p}" for p in perils if f"cost_{p}" in rows.columns]
            cost = rows.loc[m, cols].sum(axis=1).to_numpy()
            c = row_completion(rows.loc[m], comp, line, grp)
            k = np.array([adj.get((line, grp, int(y)), 1.0) for y in rows.loc[m, "year"]])
            ult = cost / c * k
            rows.loc[m, f"ult_{grp}"] = ult
            rows.loc[m, f"comp_{grp}"] = c / k
            rows.loc[m, "y_ult"] += ult
            if (line, grp) == ("motor", "bodily_injury") and "cost_bi_excess" in rows.columns:
                # developed excess above the BI cap: kept out of y_ult, used only for the
                # large-loss load in the model's level
                rows.loc[m, "ult_bi_excess"] = rows.loc[m, "cost_bi_excess"].to_numpy() / c
    return rows
