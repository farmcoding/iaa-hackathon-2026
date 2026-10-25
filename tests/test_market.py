"""Part 2 market switch: the emerging schema maps onto the developed vocabulary, and the
mapping leaves the developed market untouched. Reads the raw tables of both markets;
skips when either is absent."""

from __future__ import annotations

import numpy as np
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


def _toy():
    tr = pd.DataFrame(
        {
            "line": ["motor", "motor", "contents"],
            "exposure": [1.0, 1.0, 1.0],
            "y": [100.0, 300.0, 50.0],
            "row_id": ["a", "b", "c"],
            "x": [1.0, 3.0, 2.0],
        }
    )
    va = pd.DataFrame(
        {
            "line": ["motor", "contents"],
            "exposure": [1.0, 1.0],
            "row_id": ["d", "e"],
            "x": [2.0, 4.0],
        }
    )
    return tr, va


class _Stub:
    @staticmethod
    def fit_predict(train, valid):
        return valid["x"].to_numpy() * 7.0  # arbitrary level, relativities from x


def test_balance_on_train_sets_line_level_from_training_rows():
    from iaa import harness

    tr, va = _toy()
    cfg = {"harness": {"level": "balance_on_train"}}
    p = harness.predict(_Stub, tr, va, cfg)
    # motor: train preds 7 and 21 (sum 28) must scale to actual 400 -> factor 400/28
    assert p[0] == pytest.approx(2 * 7 * 400 / 28)
    assert p[1] == pytest.approx(4 * 7 * 50 / 14)


def test_level_neutral_rescales_each_line_to_its_actual():
    from iaa import harness

    va = pd.DataFrame({"line": ["m", "m", "c"], "y": [10.0, 30.0, 5.0], "pred": [1.0, 3.0, 9.0]})
    out = harness.level_neutral(va)
    assert out.groupby("line")["pred"].sum().to_dict() == pytest.approx({"m": 40.0, "c": 5.0})


def _synthetic(effect: float, n: int = 40_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 50.0, n)  # expected cost per row
    g = rng.choice(["F", "M"], n)
    noise = rng.choice(["a", "b", "c"], n)
    lam = 0.08 * base / 100 * np.where(g == "M", effect, 1.0)
    n_claims = rng.poisson(lam)
    y = n_claims * rng.gamma(2.0, 600.0, n)
    return pd.DataFrame({"base": base, "gender": g, "noise": noise, "n_total": n_claims, "y": y})


def test_lr_gate_passes_a_real_effect_and_rejects_noise():
    from iaa import experiment_emerging as e

    df = _synthetic(effect=1.3)
    assert e.lr_gate(df, "gender", n_tests=10)[0]
    assert not e.lr_gate(df, "noise", n_tests=10)[0]


def test_offset_glm_recovers_the_effect_direction_and_shrinks_with_alpha():
    from iaa import experiment_emerging as e

    df = _synthetic(effect=1.3, n=60_000)
    m = df["gender"] == "M"
    weak = e.offset_glm(df, df, ["gender"], alpha=1e-6)
    strong = e.offset_glm(df, df, ["gender"], alpha=10.0)
    ratio_weak = weak[m.to_numpy()].mean() / weak[~m.to_numpy()].mean()
    ratio_strong = strong[m.to_numpy()].mean() / strong[~m.to_numpy()].mean()
    assert 1.15 < ratio_weak < 1.45
    assert 1.0 <= ratio_strong < ratio_weak


def test_helpers_handle_a_single_level_factor():
    from iaa import experiment_emerging as e

    df = _synthetic(effect=1.0, n=2_000).assign(one="x")
    assert e.lr_gate(df, "one") == (False, 1.0)
    assert (e.offset_glm(df, df, ["one"]) == 1.0).all()
    assert e.offset_glm(df, df, ["one", "gender"]).shape == (len(df),)
