"""Join claims to policy-period rows and build reported (undeveloped) targets.

FROZEN: the experiment agent must not edit this file.

A claim belongs to the row of the same line and policy_id whose period contains the
loss date. Repudiated claims are excluded; net cost is incurred minus recoveries.
Development to ultimate is applied separately in ultimate.py.
"""

from __future__ import annotations

import pandas as pd

from iaa import data


def match_claims(pol: pd.DataFrame, clm: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return claims with the matched row_id, plus integrity counts.

    Integrity: every non-repudiated claim should match exactly one row.
    """
    c = clm[clm["status"] != "repudiated"]
    m = c.merge(
        pol[["row_id", "policy_id", "line", "period_start", "period_end"]],
        on=["policy_id", "line"],
        how="left",
    )
    inside = (m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])
    hits = m[inside]
    n_match = hits.groupby("claim_id").size()
    stats = {
        "claims_non_repudiated": len(c),
        "matched_once": int((n_match == 1).sum()),
        "matched_multiple": int((n_match > 1).sum()),
        "orphans": int(len(c) - len(n_match)),
    }
    # A claim matching overlapping periods is assigned to the earliest-starting row.
    hits = hits.sort_values(["claim_id", "period_start"]).drop_duplicates("claim_id")
    return hits.drop(columns=["period_start", "period_end"]), stats


def reported_targets(kind_line: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Training rows with reported claim count and net cost, by peril and in total.

    Returns (rows, matched_claims, integrity_stats). Columns added to rows:
    n_<peril>, cost_<peril>, n_total, cost_total.
    """
    lines = data.LINES if kind_line is None else (kind_line,)
    pol = pd.concat([data.policies(ln, "train") for ln in lines], ignore_index=True)
    clm = data.claims()
    clm = clm[clm["line"].isin(lines)]
    hits, stats = match_claims(pol, clm)
    stats["orphans_by_ay"] = (
        clm[(clm["status"] != "repudiated") & ~clm["claim_id"].isin(hits["claim_id"])]
        .groupby("ay")
        .size()
        .to_dict()
    )
    from iaa import ultimate

    cfg = ultimate.load_config()
    cap = ultimate.bi_cap(cfg)
    if cap:
        # large_loss.caps: motor bodily injury net cost capped per claim; excess kept apart
        perils = cfg["development"]["peril_groups"]["motor"]["bodily_injury"]
        m = (hits["line"] == "motor") & hits["peril"].isin(perils)
        hits = hits.assign(bi_excess=0.0)
        hits.loc[m, "bi_excess"] = (hits.loc[m, "net"] - cap).clip(lower=0.0)
        hits.loc[m, "net"] = hits.loc[m, "net"].clip(upper=cap)
    agg = hits.pivot_table(
        index="row_id", columns="peril", values="net", aggfunc=["size", "sum"], fill_value=0
    )
    agg.columns = [f"{'n' if a == 'size' else 'cost'}_{p}" for a, p in agg.columns]
    rows = pol.merge(agg, left_on="row_id", right_index=True, how="left")
    peril_cols = [c for c in agg.columns]
    rows[peril_cols] = rows[peril_cols].fillna(0)
    rows["n_total"] = rows[[c for c in peril_cols if c.startswith("n_")]].sum(axis=1)
    rows["cost_total"] = rows[[c for c in peril_cols if c.startswith("cost_")]].sum(axis=1)
    if cap:
        xs = hits.groupby("row_id")["bi_excess"].sum()
        rows["cost_bi_excess"] = rows["row_id"].map(xs).fillna(0.0)
    return rows, hits, stats
