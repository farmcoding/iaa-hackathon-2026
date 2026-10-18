"""Guardrail tests on the frozen core. These run before every experiment and in CI.

Tests that need the competition data skip when it is absent (CI has no data).
"""

import numpy as np
import pytest
from sklearn.metrics import d2_tweedie_score

from iaa import data, score

HAVE_DATA = (data.RAW / "claims.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="competition data not present")


def test_d2_matches_sklearn_constant_baseline():
    rng = np.random.default_rng(0)
    y = rng.gamma(0.2, 500, 5000) * (rng.random(5000) < 0.1)
    p = np.full(5000, y.mean()) * rng.uniform(0.5, 1.5, 5000)
    assert score.d2(y, p) == pytest.approx(d2_tweedie_score(y, p, power=1.5), rel=1e-9)


def test_zero_prediction_is_floored_not_infinite():
    assert np.isfinite(score.d2(np.array([0.0, 100.0]), np.array([0.0, 0.0])))


@needs_data
def test_exposure_recomputed_and_bounded():
    for line in data.LINES:
        for kind in ("train", "test"):
            df = data.policies(line, kind)
            assert df["exposure"].between(0, 1.0028).all(), (line, kind)
        assert data.policies(line, "train")["dq_exposure_fixed"].mean() < 0.01


@needs_data
def test_no_claim_matches_two_rows_and_orphans_are_few():
    from iaa import target

    _, _, stats = target.reported_targets()
    assert stats["matched_multiple"] == 0
    assert stats["orphans"] / stats["claims_non_repudiated"] < 0.005


@needs_data
def test_developed_2024_loss_cost_lands_on_trend():
    """The discriminating test for the development method: developed 2024 loss cost per
    line must sit near the 2021-2023 trend line (buildings may sit above on weather)."""
    from iaa import target, ultimate

    rows, _, _ = target.reported_targets()
    r = ultimate.developed_targets(rows)
    lc = r.groupby(["line", "year"]).apply(lambda g: g["y_ult"].sum() / g["exposure"].sum())
    for line, tol in (("motor", 0.05), ("contents", 0.05), ("buildings", 0.08)):
        s = lc.loc[line]
        slope = np.polyfit([2021, 2022, 2023], np.log(s.loc[[2021, 2022, 2023]]), 1)[0]
        implied = s.loc[2023] * np.exp(slope)
        assert abs(s.loc[2024] / implied - 1) < tol, (line, s.loc[2024], implied)


def test_renewal_decision_is_not_a_feature():
    from iaa import features

    assert "renewal_decision" in features.FORBIDDEN
