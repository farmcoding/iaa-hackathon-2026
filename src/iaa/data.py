"""Load and clean the competition tables.

FROZEN: the experiment agent must not edit this file. Every cleaning rule here is a
documented data-quality decision (see reports/data_quality.md).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# one folder per market; tables are joined only within a market (ids repeat across them).
# IAA_MARKET picks the market the cached loaders read; developed is the Part 1 default.
MARKET_DIRS = {
    "developed": ROOT / "data" / "raw" / "public",
    "emerging": ROOT / "data" / "part2_raw" / "emerging",
}
MARKET = os.environ.get("IAA_MARKET", "developed")
RAW = MARKET_DIRS[MARKET]
LINES = ("motor", "buildings", "contents")
ID_COLS = ("customer_id", "address_id", "policy_id", "row_id", "claim_id")
DAYS_PER_YEAR = 365.0


def _read(name: str, raw: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet((RAW if raw is None else raw) / f"{name}.parquet")
    for col in ID_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


@cache
def customers() -> pd.DataFrame:
    return _read("customers")


@cache
def addresses() -> pd.DataFrame:
    return _read("addresses")


@cache
def weather() -> pd.DataFrame:
    return _read("weather")


@cache
def claims(market: str | None = None) -> pd.DataFrame:
    """Claims with dates parsed, net cost and accident year.

    net = incurred_to_date - recoveries. Repudiated claims are kept here (flagged) so the
    data-quality log can count them; target construction drops them. market reads another
    market's claims (used to borrow development patterns); default is IAA_MARKET.
    """
    c = _read("claims", None if market is None else MARKET_DIRS[market])
    c["loss_date"] = pd.to_datetime(c["loss_date"])
    c["report_date"] = pd.to_datetime(c["report_date"])
    c["recoveries"] = c["recoveries"].fillna(0.0)
    c["net"] = c["incurred_to_date"] - c["recoveries"]
    c["ay"] = c["loss_date"].dt.year
    c["report_lag_days"] = (c["report_date"] - c["loss_date"]).dt.days
    return c


@cache
def policies(line: str, kind: str = "train", market: str | None = None) -> pd.DataFrame:
    """Policy-period rows for one line. kind is 'train' (2021-2024) or 'test' (2025).

    Exposure is recomputed from the period dates. About 0.4% of training rows carry
    exposure in days (365, 89, ...) and about 0.3% carry 1.01; the test files are clean.
    The raw value is kept in exposure_raw and the defect is flagged.
    """
    if line not in LINES:
        raise ValueError(f"unknown line {line!r}")
    name = f"policies_{line}" if kind == "train" else f"test_{line}"
    df = _read(name, None if market is None else MARKET_DIRS[market])
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["exposure_raw"] = df["exposure"]
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    df["exposure"] = days / DAYS_PER_YEAR
    df["dq_exposure_fixed"] = (df["exposure_raw"] - df["exposure"]).abs() > 0.01
    df["year"] = df["period_start"].dt.year
    df["line"] = line
    return df


def all_policies(kind: str = "train") -> pd.DataFrame:
    return pd.concat([policies(ln, kind) for ln in LINES], ignore_index=True)
