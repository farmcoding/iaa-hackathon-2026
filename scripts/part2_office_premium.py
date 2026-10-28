"""Office premium build-up for the 20 emerging reference policies (Part 2 written report).

The data page asks, for each reference policy, how the office premium is built from the
predicted risk premium with the `expenses` assumptions (judged by the panel; no Kaggle
effect). Risk premium = the selected submission's prediction for the row (expected
ultimate claim cost for the row's period). Emerging-market items:

    fixed expense per policy  F  (CU, charged in full per policy, whatever its exposure)
    cat reinsurance           c  x risk premium, property lines only (a cost on top of claims)
    variable expense          v  share of premium
    commission                k  share of premium, by the policy's channel
    premium tax               t  share of premium
    target profit margin      m  share of premium

    office premium = (risk premium + F + c x property risk premium) / (1 - v - k - t - m)

so every share-of-premium item is a share of the office premium itself, and the
build-up adds back to the office premium exactly.

    .venv/bin/python -I scripts/part2_office_premium.py SUBMISSION.csv

Writes reports/part2_office_premium.md (and .csv next to it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "part2_raw" / "emerging"
COMMISSION = {
    "direct": "commission_direct",
    "broker": "commission_broker",
    "comparison_site": "commission_comparison_site",
}


def md(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
        for r in df.itertuples(index=False)
    ]
    return "\n".join([head, rule, *body])


def build(sub: pd.DataFrame) -> pd.DataFrame:
    ex = pd.read_parquet(RAW / "expenses.parquet").set_index("item")["emerging"]
    ref = pd.read_parquet(RAW / "reference_policies.parquet")
    df = ref.merge(sub, on="row_id", how="left", validate="one_to_one")
    if df["prediction"].isna().any():
        raise SystemExit("a reference policy has no prediction in the submission")
    k = df["channel"].astype(str).map(COMMISSION).map(ex)
    if k.isna().any():  # a blank emerging value: the channel has no assumption in this market
        raise SystemExit(
            f"no emerging commission for channels {sorted(df.loc[k.isna(), 'channel'].unique())}"
        )
    out = pd.DataFrame(
        {
            "row_id": df["row_id"],
            "line": df["line"],
            "channel": df["channel"],
            "new_business": df["new_business"],
            "exposure": df["exposure"].round(4),
            "risk_premium": df["prediction"],
        }
    )
    prop = df["line"].isin(["buildings", "contents"])
    out["cat_reinsurance"] = (ex["cat_reinsurance"] * df["prediction"]).where(prop, 0.0)
    out["fixed_expense"] = ex["fixed_expense_per_policy"]
    shares = ex["variable_expense"] + k + ex["premium_tax"] + ex["target_profit_margin"]
    out["office_premium"] = (
        out["risk_premium"] + out["cat_reinsurance"] + out["fixed_expense"]
    ) / (1 - shares)
    for name, s in (
        ("variable_expense", ex["variable_expense"]),
        ("commission", k),
        ("premium_tax", ex["premium_tax"]),
        ("profit", ex["target_profit_margin"]),
    ):
        out[name] = s * out["office_premium"]
    parts = [
        "risk_premium",
        "cat_reinsurance",
        "fixed_expense",
        "variable_expense",
        "commission",
        "premium_tax",
        "profit",
    ]
    assert (out[parts].sum(axis=1) - out["office_premium"]).abs().max() < 1e-6
    out["loss_ratio"] = out["risk_premium"] / out["office_premium"]
    out["incumbent_premium"] = df["written_premium"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    a = ap.parse_args()
    out = build(pd.read_csv(a.submission))
    dest = ROOT / "reports" / "part2_office_premium"
    out.to_csv(dest.with_suffix(".csv"), index=False)
    ex = pd.read_parquet(RAW / "expenses.parquet")
    cols = [
        "row_id",
        "line",
        "channel",
        "exposure",
        "risk_premium",
        "cat_reinsurance",
        "fixed_expense",
        "variable_expense",
        "commission",
        "premium_tax",
        "profit",
        "office_premium",
        "loss_ratio",
        "incumbent_premium",
    ]
    fmt = out[cols].copy()
    for c in cols[4:-2] + ["incumbent_premium"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:,.2f}")
    fmt["loss_ratio"] = out["loss_ratio"].map(lambda x: f"{x:.1%}")
    lines = [
        "# Office premium build-up, 20 emerging reference policies",
        "",
        f"Risk premium: `{a.submission.name}` (expected ultimate claim cost for the row's period).",
        "Emerging expense assumptions (`expenses.parquet`):",
        "",
        md(ex),
        "",
        "Office premium = (risk premium + fixed expense + cat reinsurance) / (1 - variable expense -",
        "commission - premium tax - profit margin). Cat reinsurance is 12% of the risk premium on",
        "buildings and contents only. Share-of-premium items are shares of the office premium, so each",
        "row adds back exactly. The fixed expense is charged in full per policy, so short-period",
        "policies (exposure 0.07 and 0.11 here) carry it in full; pro-rating it by exposure is the",
        "alternative if the expense is incurred over the policy year. `incumbent_premium` is the",
        "broker book's existing written premium, for comparison only (not a model input).",
        "",
        md(fmt),
        "",
    ]
    dest.with_suffix(".md").write_text("\n".join(lines))
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
