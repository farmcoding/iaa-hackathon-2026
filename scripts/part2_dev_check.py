"""Does emerging claim experience carry its own development signal? (owner question, 28 Oct)

The emerging target develops reported net cost with report-delay chain-ladder patterns
borrowed from the developed market (configs/market_emerging.yaml, `pattern_market`). The
claims file holds one valuation (31 Dec 2024), so incurred development (IBNER) cannot be
observed, but report dates are history: emerging claims give their own report-delay
triangle for the first 24 months. This script fits that triangle (`ultimate.delay_cdfs`
on emerging claims), splices it onto the developed tail at delay D, and compares the
implied ultimate of each loss year with the borrowed pattern's, for cost and for counts.
A claim-level bootstrap gives the sampling error of the emerging pattern.

    IAA_MARKET=emerging .venv/bin/python -I scripts/part2_dev_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iaa import data, ultimate

SPLICE = (6, 12, 18)
DRAWS = 200


def ratios(f_dev: pd.DataFrame, f_emg: pd.DataFrame, splice: int) -> pd.DataFrame:
    """Spliced cdf: emerging age-to-age factors below `splice`, developed from there on."""
    out = {}
    emg_rows = dict(f_emg.iterrows())
    for key, row in f_dev.iterrows():
        dev, emg = row.to_numpy(), emg_rows[key].to_numpy()
        a_dev = dev[1:] / dev[:-1]
        a_emg = np.where(np.arange(len(a_dev)) < splice, emg[1:] / emg[:-1], a_dev)
        out[key] = np.append(1.0 / np.cumprod(a_emg[::-1])[::-1], 1.0)
    return pd.DataFrame.from_dict(out, orient="index", columns=f_dev.columns)


def ultimates(claims: pd.DataFrame, cdfs: pd.DataFrame, cfg: dict, value: str) -> pd.Series:
    """Developed ultimate by (line, group, loss year): reported-to-date / completion."""
    gmap = ultimate.peril_group_map(cfg)
    c = claims[(claims["status"] != "repudiated") & (claims["ay"].isin([2023, 2024]))].copy()
    c["group"] = [gmap[(ln, p)] for ln, p in zip(c["line"], c["peril"], strict=True)]
    c["lm"] = ultimate._months(c["loss_date"])
    c["v"] = c["net"] if value == "amt" else 1.0
    comp = ultimate.completion_by_month(cdfs, cfg).set_index(["line", "group", "lm"])
    rep = c.groupby(["line", "group", "lm", "ay"])["v"].sum().reset_index()
    rep["ult"] = (
        rep["v"]
        / comp.loc[list(zip(rep.line, rep.group, rep.lm, strict=True)), "completion"].to_numpy()
    )
    return rep.groupby(["line", "group", "ay"])["ult"].sum()


def main() -> None:
    cfg = ultimate.load_config()
    dev_claims, emg_claims = data.claims("developed"), data.claims()
    rng = np.random.default_rng(0)
    for value in ("amt", "count"):
        f_dev = ultimate.delay_cdfs(dev_claims, cfg, value=value)
        f_emg = ultimate.delay_cdfs(emg_claims, cfg, value=value)
        base = ultimates(emg_claims, f_dev, cfg, value)
        print(f"\n== {value}: emerging-pattern ultimate / borrowed-pattern ultimate ==")
        table = {}
        for s in SPLICE:
            table[f"splice {s}m"] = (
                ultimates(emg_claims, ratios(f_dev, f_emg, s), cfg, value) / base
            )
        # bootstrap the 12-month splice: resample emerging claims, refit the emerging pattern
        boots = []
        for _ in range(DRAWS):
            b = emg_claims.sample(len(emg_claims), replace=True, random_state=rng.integers(1 << 31))
            fb = ultimate.delay_cdfs(b, cfg, value=value)
            boots.append(ultimates(emg_claims, ratios(f_dev, fb, 12), cfg, value) / base)
        bt = pd.concat(boots, axis=1)
        table["12m p10"], table["12m p90"] = bt.quantile(0.1, axis=1), bt.quantile(0.9, axis=1)
        out = pd.DataFrame(table)
        out["borrowed ult"] = base.round(0 if value == "count" else -3)
        print(out.round(3).to_string())
        # line totals by loss year at the 12-month splice
        w = pd.DataFrame({"b": base, "e": table["splice 12m"] * base})
        lt = w.groupby(level=["line", "ay"]).sum()
        print("line x year, 12m splice:", (lt.e / lt.b).round(3).to_dict())


if __name__ == "__main__":
    main()
