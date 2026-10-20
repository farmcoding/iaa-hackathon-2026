"""Data-contract guardrail tests (owner item 3.28, docs/pipeline_review.md section 4.3).

These read the cached competition dataset, so they skip when the data (or its cache)
is absent. Module-scoped fixtures build the dataset and load claims once so the whole
file stays well under the 60-second budget (the dataset is read from an existing
parquet cache rather than rebuilt; see harness.dataset()).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iaa import data, features, harness, score, ultimate

HAVE_DATA = (data.RAW / "claims.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="competition data not present")

TARGET_LIKE_COLS = ("cost_", "n_", "ult_", "comp_")


@pytest.fixture(scope="module")
def dataset():
    if not HAVE_DATA:
        pytest.skip("competition data not present")
    return harness.dataset()


@pytest.fixture(scope="module")
def claims():
    if not HAVE_DATA:
        pytest.skip("competition data not present")
    return data.claims()


@pytest.fixture(scope="module")
def cfg():
    return ultimate.load_config()


@needs_data
def test_forbidden_and_target_columns_absent_from_features(dataset):
    """No FORBIDDEN or target-prefixed column reaches the feature set the experiment
    agent can see for any line; customer_service_contacts named explicitly (it is the
    one FORBIDDEN leak with no obvious name match to a target prefix)."""
    from iaa import experiment

    assert "customer_service_contacts" in features.FORBIDDEN
    for line in data.LINES:
        sub = dataset[dataset["line"] == line].head(5)
        cols = set(experiment._feature_cols(sub))
        assert not (cols & features.FORBIDDEN)
        assert not any(c.startswith(TARGET_LIKE_COLS) for c in cols)
        assert "y" not in cols and "y_ult" not in cols


@needs_data
def test_exposure_strictly_positive_and_bounded(dataset):
    """Stricter than the existing test_core check (which admits 0): a policy period of
    zero days would be a data defect, not a valid row."""
    assert (dataset["exposure"] > 0).all()
    assert (dataset["exposure"] <= 1.0028).all()


@needs_data
def test_no_duplicate_row_ids(dataset):
    assert dataset["row_id"].is_unique


@needs_data
def test_test_split_carries_no_target_columns(dataset):
    """Test (2025) rows must never carry a developed target: merging one in by
    accident would let a 'feature' leak the answer."""
    te = dataset[dataset["split"] == "test"]
    # "y" (the on-levelled, trended target) is computed per fold in harness.split, not
    # a dataset() column; y_ult (developed, untrended) is.
    target_cols = [c for c in dataset.columns if c.startswith(TARGET_LIKE_COLS)] + ["y_ult"]
    for col in target_cols:
        assert te[col].isna().all(), f"{col} is not all-null on the test split"


@needs_data
def test_peril_groups_cover_every_peril(claims, cfg):
    """Every peril seen in the claims table for a line is assigned to exactly one
    group in configs/market_dev.yaml development.peril_groups; an unassigned peril
    would be silently dropped from the row's developed cost."""
    groups = cfg["development"]["peril_groups"]
    for line in data.LINES:
        seen = set(claims.loc[claims["line"] == line, "peril"].unique())
        grouped = {p for perils in groups[line].values() for p in perils}
        assert seen <= grouped, f"{line}: ungrouped perils {seen - grouped}"
        # and no group lists a peril that doesn't exist for the line (stale config)
        assert grouped <= seen, f"{line}: configured perils absent from claims {grouped - seen}"


@needs_data
def test_claim_dates_ordered_and_matches_fall_inside_their_row(claims):
    non_rep = claims[claims["status"] != "repudiated"]
    assert (non_rep["report_date"] >= non_rep["loss_date"]).all()

    from iaa import target as target_mod

    pol = data.all_policies("train")
    hits, _ = target_mod.match_claims(pol, claims)
    merged = hits.merge(pol[["row_id", "period_start", "period_end"]], on="row_id", how="left")
    assert (merged["loss_date"] >= merged["period_start"]).all()
    assert (merged["loss_date"] <= merged["period_end"]).all()


@needs_data
def test_weather_join_matches_the_row_own_year(dataset):
    """Weather features come from the row's own year, never a later one (features.py
    joins on [address_id, year])."""
    wx = data.weather().rename(columns={"year": "year"})
    wx_cols = [c for c in wx.columns if c not in ("address_id", "year")]
    sample = dataset.dropna(subset=["address_id"]).sample(n=min(500, len(dataset)), random_state=0)
    check = sample[["address_id", "year"] + wx_cols].merge(
        wx, on=["address_id", "year"], how="left", suffixes=("", "_raw")
    )
    for col in wx_cols:
        a, b = check[col].to_numpy(dtype=float), check[f"{col}_raw"].to_numpy(dtype=float)
        ok = np.isnan(a) & np.isnan(b) | np.isclose(a, b, equal_nan=False, rtol=1e-6)
        assert ok.all(), f"weather column {col} does not match the row's own year"


@needs_data
def test_history_features_only_count_claims_reported_before_the_row_starts():
    """Spot-check features.claim_history: a sampled row's hist_n_policy must equal a
    from-scratch count of its own policy's non-repudiated claims reported in
    [period_start - 365d, period_start)."""
    pol = data.all_policies("train")
    c = data.claims()
    c = c[c["status"] != "repudiated"]
    hist = features.claim_history(pol)
    full = pol[pol["period_start"] >= pd.Timestamp("2022-01-01")]
    sample = full.sample(n=min(300, len(full)), random_state=0)
    for row_id, policy_id, start in zip(
        sample["row_id"], sample["policy_id"], sample["period_start"]
    ):
        window = c[
            (c["policy_id"] == policy_id)
            & (c["report_date"] >= start - pd.Timedelta(days=365))
            & (c["report_date"] < start)
        ]
        expect_n = len(window)
        got_n = hist.loc[pol.index[pol["row_id"] == row_id][0], "hist_n_policy"]
        assert got_n == pytest.approx(expect_n), row_id


@needs_data
def test_row_ultimates_reconcile_to_the_group_chain_ladder(cfg):
    """Each row's ult_<group> (from developed_targets) must equal reported group cost
    divided by its own completion, times the freq-sev adjustment k, summed by line,
    group and year. Rebuilt from the lower-level pieces (delay_cdfs,
    completion_by_month, row_completion, freq_sev_adjustment) instead of calling
    developed_targets a second time, so a bug in how its loop assembles the sum (e.g.
    a dropped `* k`, or `y_ult` not accumulating every group) is caught rather than
    merely restated."""
    from iaa import target as target_mod

    rows, _, _ = target_mod.reported_targets()
    dev = ultimate.developed_targets(rows, cfg)
    cdfs = ultimate.delay_cdfs(data.claims(), cfg)
    comp = ultimate.completion_by_month(cdfs, cfg)
    adj = ultimate.freq_sev_adjustment(rows, cfg)
    for line, groups in cfg["development"]["peril_groups"].items():
        m = dev["line"] == line
        sub = dev.loc[m]
        for grp, perils in groups.items():
            cols = [f"cost_{p}" for p in perils if f"cost_{p}" in dev.columns]
            cost = sub[cols].sum(axis=1).to_numpy()
            c = ultimate.row_completion(sub, comp, line, grp)
            k = np.array([adj.get((line, grp, int(y)), 1.0) for y in sub["year"]])
            est = pd.Series(cost / c * k, index=sub.index)
            est_by_year = est.groupby(sub["year"]).sum()
            got_by_year = sub.groupby("year")[f"ult_{grp}"].sum()
            for year, expect in est_by_year.items():
                assert got_by_year[year] == pytest.approx(expect, rel=1e-6), (line, grp, year)


@needs_data
def test_train_test_schema_matches(dataset):
    """Train and test must offer the same feature columns and dtypes, and any test
    categorical level unseen in training must be a small minority (a large share would
    mean the encoder's new-level path, not its trained levels, is doing the work)."""
    tr = dataset[dataset["split"] == "train"]
    te = dataset[dataset["split"] == "test"]
    non_target = [
        c
        for c in dataset.columns
        if not c.startswith(TARGET_LIKE_COLS) and c not in ("y", "y_ult", "split")
    ]
    assert set(tr[non_target].columns) == set(te[non_target].columns)
    for col in non_target:
        assert tr[col].dtype == te[col].dtype, col
        if isinstance(tr[col].dtype, pd.CategoricalDtype):
            tr_levels = set(tr[col].dropna().unique())
            unseen = ~te[col].dropna().isin(tr_levels)
            share = unseen.mean() if len(te) else 0.0
            assert share < 0.01, f"{col}: {share:.1%} of test rows carry an unseen level"


def test_predictions_must_be_finite_non_negative_and_floored():
    good = harness.check_predictions(np.array([0.0, 5.0, 1e-6]), 3)
    assert (good >= score.FLOOR).all()
    for bad in (np.array([-1.0, 2.0]), np.array([np.nan, 2.0]), np.array([np.inf, 2.0])):
        with pytest.raises(ValueError):
            harness.check_predictions(bad, 2)
    with pytest.raises(ValueError):
        harness.check_predictions(np.array([1.0, 2.0]), 3)  # wrong length


def test_submission_matches_sample_submission():
    """If a final submission has been written, its row count and row_ids must match
    sample_submission exactly (harness.final already enforces this at write time; this
    guards against a submission reaching submissions/ by any other path)."""
    files = sorted(harness.SUBMISSIONS.glob("*.csv")) if harness.SUBMISSIONS.exists() else []
    if not files:
        pytest.skip("no submission written yet")
    sub = pd.read_csv(files[-1], dtype={"row_id": str})
    sample = pd.read_parquet(data.RAW / "sample_submission.parquet")
    assert len(sub) == len(sample)
    assert set(sub["row_id"]) == set(sample["row_id"].astype(str))


@needs_data
def test_dataset_fingerprint_is_stable_across_calls():
    assert harness._fingerprint() == harness._fingerprint()
    path = harness.CACHE / f"dataset_{harness._fingerprint()}.parquet"
    assert path.exists()
