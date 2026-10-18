"""Base feature frame shared by every experiment.

FROZEN: the experiment agent must not edit this file. Experiments derive extra
features from this frame inside experiment.py.

Rules enforced here:
- renewal_decision (train-only, describes what happened after the period) is dropped.
- Weather joins on the row's own year, never a later one.
- Claim-history features count only claims reported before the row starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iaa import data

# renewal_decision: train-only, describes what happened after the period.
# customer_service_contacts: in training rows it includes claim-related contacts (rows
#   with a claim average 3.5 per exposure-year vs 0.5 without); test rows match the
#   no-claim distribution, so it leaks the target. Found 2026-10-18, see data_quality.md.
# customer_since_year: drifts between train and test by construction; tenure replaces it.
FORBIDDEN = {
    "renewal_decision",
    "exposure_raw",
    "dq_exposure_fixed",
    "customer_service_contacts",
    "customer_since_year",
}
ID_COLS = ["row_id", "policy_id", "customer_id", "address_id"]
HIST_WINDOW_DAYS = 365
_EPOCH = pd.Timestamp("2000-01-01")


def _days(s: pd.Series) -> np.ndarray:
    return (s - _EPOCH).dt.days.to_numpy()


def _window_counts(
    keys: pd.Series, starts: pd.Series, ckeys: pd.Series, cdates: pd.Series, cvals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum of claims with the same key reported in [start - window, start)."""
    codes, _ = pd.factorize(pd.concat([keys, ckeys], ignore_index=True))
    rk, ck = codes[: len(keys)].astype(np.int64), codes[len(keys) :].astype(np.int64)
    scale = 100_000  # days since 2000 stay below this
    comb_c = ck * scale + _days(cdates)
    order = np.argsort(comb_c, kind="stable")
    comb_c = comb_c[order]
    cum = np.concatenate([[0.0], np.cumsum(cvals[order])])
    hi = np.searchsorted(comb_c, rk * scale + _days(starts), side="left")
    lo = np.searchsorted(comb_c, rk * scale + _days(starts) - HIST_WINDOW_DAYS, side="left")
    return (hi - lo).astype(float), cum[hi] - cum[lo]


def claim_history(rows: pd.DataFrame) -> pd.DataFrame:
    """Prior-12-month claim counts and cost for the policy and for the customer."""
    c = data.claims()
    c = c[c["status"] != "repudiated"]
    pol_map = data.all_policies("train")[["policy_id", "customer_id"]].drop_duplicates("policy_id")
    c = c.merge(pol_map, on="policy_id", how="left")
    out = pd.DataFrame(index=rows.index)
    for key in ("policy_id", "customer_id"):
        cc = c.dropna(subset=[key])
        n, s = _window_counts(
            rows[key], rows["period_start"], cc[key], cc["report_date"], cc["net"].to_numpy()
        )
        out[f"hist_n_{key[:-3]}"] = n
        out[f"hist_cost_{key[:-3]}"] = s
    # Rows starting before 2022 have no full look-back window: their history is blanked
    # (NaN), otherwise "no prior claims" means different things in different years.
    full = (rows["period_start"] >= pd.Timestamp("2022-01-01")).to_numpy()
    out.loc[~full, :] = np.nan
    return out


def build(kind: str = "train") -> pd.DataFrame:
    """All three lines, one frame. Line-specific columns are NaN on other lines."""
    pol = data.all_policies(kind)
    pol = pol.drop(columns=[c for c in FORBIDDEN if c in pol.columns])
    cust = data.customers().copy()
    cust["date_of_birth"] = pd.to_datetime(cust["date_of_birth"])
    df = pol.merge(cust, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(data.addresses(), on="address_id", how="left", suffixes=("", "_addr"))
    w = data.weather().rename(columns={"year": "year"})
    df = df.merge(w, on=["address_id", "year"], how="left", suffixes=("", "_wx"))
    df["customer_age"] = (df["period_start"] - df["date_of_birth"]).dt.days / 365.25
    df["customer_tenure"] = df["year"] - df["customer_since_year"]
    df["property_age"] = df["year"] - df["year_built"]
    df["start_month"] = df["period_start"].dt.month
    df = df.drop(columns=["date_of_birth", "surname"])
    df = df.drop(columns=[c for c in FORBIDDEN if c in df.columns])
    df = pd.concat([df, claim_history(df)], axis=1)
    holdings = data.all_policies(kind).groupby(["customer_id", "year"])["line"].nunique()
    # no n_ prefix: the harness treats n_ columns as claim counts and hides them
    df["lines_held"] = df.set_index(["customer_id", "year"]).index.map(holdings).to_numpy()
    return df


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Text and boolean feature columns become categoricals (shared levels across splits)."""
    for col in df.columns:
        if col in ID_COLS or col in ("split", "line"):
            continue
        dt = df[col].dtype
        if isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_datetime64_any_dtype(dt):
            continue
        if not pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
            df[col] = df[col].astype("string").astype("category")
    return df
