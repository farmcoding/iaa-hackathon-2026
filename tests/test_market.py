"""Part 2 market switch: the emerging schema maps onto the developed vocabulary, and the
mapping leaves the developed market untouched. Reads the raw tables of both markets;
skips when either is absent."""

from __future__ import annotations

import pandas as pd
import pytest

from iaa import data, experiment, features

HAVE = all((data.MARKET_DIRS[m] / "claims.parquet").exists() for m in data.MARKET_DIRS)
pytestmark = pytest.mark.skipif(not HAVE, reason="data for both markets not present")


def _cust(market: str) -> pd.DataFrame:
    return data._read("customers", data.MARKET_DIRS[market])


def test_developed_customers_unchanged_by_harmonize():
    raw = _cust("developed")
    out = features.harmonize_customers(raw)
    assert "self_employed" not in out.columns
    pd.testing.assert_series_equal(out["occupation"], raw["occupation"])


def test_emerging_occupation_folds_to_developed_titles():
    dev = set(_cust("developed")["occupation"].dropna())
    raw = _cust("emerging")
    out = features.harmonize_customers(raw)
    assert set(out["occupation"].dropna()) <= dev
    assert out["occupation"].isna().sum() == raw["occupation"].isna().sum()
    prefixed = raw["occupation"].str.strip().str.lower().str.startswith("self-employed")
    assert (out["self_employed"].fillna(False) == prefixed.fillna(False)).all()


def test_emerging_construction_codes_map_to_developed_labels():
    dev = data._read("policies_buildings", data.MARKET_DIRS["developed"])
    emg = data._read("test_buildings", data.MARKET_DIRS["emerging"])
    mapped = features.harmonize_policies(emg)
    labels = set(experiment._fix_construction(dev).dropna())
    assert set(experiment._fix_construction(mapped).dropna()) <= labels


def test_emerging_age_bands_become_ages():
    emg = data._read("test_motor", data.MARKET_DIRS["emerging"])
    out = features.harmonize_policies(emg)
    assert out["driver_age"].notna().sum() == emg["driver_age_band"].notna().sum()
    assert set(out["driver_age"].dropna()) <= set(features.AGE_BAND_MEAN.values())


def test_emerging_test_rows_match_sample_submission():
    rows = pd.concat(
        [data.policies(ln, "test", "emerging")["row_id"] for ln in data.LINES], ignore_index=True
    )
    sample = pd.read_parquet(data.MARKET_DIRS["emerging"] / "sample_submission.parquet")
    assert rows.is_unique and set(rows) == set(sample["row_id"].astype("string"))
