"""Memory-light EDA aggregates for the dashboard, cached to JSON."""

import json
import sys

import pandas as pd

from iaa import data, target, ultimate

out = {}
cols = [
    "row_id",
    "policy_id",
    "period_start",
    "period_end",
    "exposure",
    "customer_service_contacts",
]
pols, tests = [], []
for ln in data.LINES:
    p = pd.read_parquet(data.RAW / f"policies_{ln}.parquet", columns=cols)
    t = pd.read_parquet(data.RAW / f"test_{ln}.parquet", columns=cols)
    for df in (p, t):
        for c in ("row_id", "policy_id"):
            df[c] = df[c].astype("string")
        df["period_start"] = pd.to_datetime(df.period_start)
        df["period_end"] = pd.to_datetime(df.period_end)
        df["exposure_raw"] = df["exposure"]
        df["exposure"] = ((df.period_end - df.period_start).dt.days + 1) / 365
        df["line"] = ln
        df["year"] = df.period_start.dt.year
    pols.append(p)
    tests.append(t)
pol, test = pd.concat(pols, ignore_index=True), pd.concat(tests, ignore_index=True)
out["exposure_fix"] = {
    "days_like": int((pol.exposure_raw > 1.5).sum()),
    "near_1_01": int(((pol.exposure_raw - 1.01).abs() < 0.005).sum()),
    "train_rows": len(pol),
    "test_rows": len(test),
}
clm = data.claims()
hits, stats = target.match_claims(pol, clm)
out["join"] = stats
agg = hits.pivot_table(
    index="row_id", columns="peril", values="net", aggfunc=["size", "sum"], fill_value=0
)
agg.columns = [f"{'n' if a == 'size' else 'cost'}_{p}" for a, p in agg.columns]
rows = pol.merge(agg, left_on="row_id", right_index=True, how="left").fillna(
    {c: 0 for c in agg.columns}
)
rows["cost_total"] = rows[[c for c in agg.columns if c.startswith("cost_")]].sum(axis=1)
rows["n_total"] = rows[[c for c in agg.columns if c.startswith("n_")]].sum(axis=1)
dev = ultimate.developed_targets(rows)
g = (
    dev.groupby(["line", "year"])
    .agg(
        rows=("row_id", "size"),
        exposure=("exposure", "sum"),
        claims=("n_total", "sum"),
        reported=("cost_total", "sum"),
        ultimate=("y_ult", "sum"),
    )
    .reset_index()
)
out["line_year"] = g.to_dict("records")
out["test_by_line"] = (
    test.groupby("line")
    .agg(rows=("row_id", "size"), exposure=("exposure", "sum"))
    .reset_index()
    .to_dict("records")
)
has = rows.n_total > 0
out["contacts"] = {
    "no_claim": float(
        rows.loc[~has, "customer_service_contacts"].sum() / rows.loc[~has, "exposure"].sum()
    ),
    "with_claim": float(
        rows.loc[has, "customer_service_contacts"].sum() / rows.loc[has, "exposure"].sum()
    ),
    "test": float(test.customer_service_contacts.sum() / test.exposure.sum()),
    "dist_no_claim": rows.loc[~has, "customer_service_contacts"]
    .clip(upper=5)
    .value_counts(normalize=True)
    .sort_index()
    .to_dict(),
    "dist_with_claim": rows.loc[has, "customer_service_contacts"]
    .clip(upper=5)
    .value_counts(normalize=True)
    .sort_index()
    .to_dict(),
    "dist_test": test.customer_service_contacts.clip(upper=5)
    .value_counts(normalize=True)
    .sort_index()
    .to_dict(),
}
b = clm[(clm.peril == "bodily_injury") & (clm.status != "repudiated") & (clm.ay >= 2021)].copy()
b["dm"] = (b.report_date.dt.year * 12 + b.report_date.dt.month) - (
    b.loss_date.dt.year * 12 + b.loss_date.dt.month
)
b["bucket"] = pd.cut(
    b.dm, [-1, 0, 2, 5, 11, 23, 60], labels=["0", "1-2", "3-5", "6-11", "12-23", "24+"]
)
sev = b.assign(n=b.net.clip(upper=250000)).pivot_table(
    index="ay", columns="bucket", values="n", aggfunc="mean", observed=False
)
opn = b.assign(o=b.status.eq("open")).pivot_table(
    index="ay", columns="bucket", values="o", aggfunc="mean", observed=False
)
out["bi_sev"] = {
    str(k): {str(c): (None if pd.isna(v) else float(v)) for c, v in r.items()}
    for k, r in sev.iterrows()
}
out["bi_open"] = {
    str(k): {str(c): (None if pd.isna(v) else float(v)) for c, v in r.items()}
    for k, r in opn.iterrows()
}
cdf = ultimate.delay_cdfs(clm, ultimate.load_config())
out["delay_cdf"] = {
    f"{a} / {b_}": [float(x) for x in r.values[:25]] for (a, b_), r in cdf.iterrows()
}
miss = {}
for ln in data.LINES:
    p = pd.read_parquet(data.RAW / f"policies_{ln}.parquet")
    t = pd.read_parquet(data.RAW / f"test_{ln}.parquet")
    for c in p.columns:
        if c in t.columns and p[c].isna().mean() > 0.001:
            miss[f"{ln}: {c}"] = {
                "train": float(p[c].isna().mean()),
                "test": float(t[c].isna().mean()),
            }
    del p, t
out["missing"] = miss
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, default=float, indent=1)
print("ok", len(out["line_year"]))
