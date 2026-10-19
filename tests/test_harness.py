"""Guardrail tests on the harness's own rules (no competition data needed)."""

import csv

import pandas as pd
import pytest

from iaa import features, harness


def test_hide_target_removes_every_target_column():
    va = pd.DataFrame(
        {
            "row_id": ["1"],
            "lines_held": [2],
            "n_total": [1],
            "n_bodily_injury": [0],
            "cost_total": [10.0],
            "ult_all": [12.0],
            "comp_all": [0.8],
            "y": [12.0],
            "y_ult": [12.0],
        }
    )
    assert list(harness.hide_target(va).columns) == ["row_id", "lines_held"]


def test_no_feature_shares_a_target_prefix():
    # a feature named n_* or cost_* would be hidden from validation rows (pipeline issue P1)
    import inspect

    src = inspect.getsource(features)
    for prefix in harness.TARGET_COLS_PREFIX:
        assert f'df["{prefix}' not in src, f"features.py creates a column starting {prefix}"


def _ledger(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=harness.FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    path = tmp_path / "results.tsv"
    _ledger(
        path,
        [
            {"timestamp": "t1", "verdict": "keep", "d2_A": 0.12, "d2_B": 0.13, "min_fold": 0.12},
            {"timestamp": "t2", "verdict": "keep", "d2_A": 0.13, "d2_B": 0.14, "min_fold": 0.13},
            {
                "timestamp": "t3",
                "verdict": "keep_owner",
                "d2_A": 0.125,
                "d2_B": 0.135,
                "min_fold": 0.125,
            },
            {"timestamp": "t4", "verdict": "blend_pass", "d2_A": 0.2, "d2_B": 0.2, "min_fold": 0.2},
        ],
    )
    monkeypatch.setattr(harness, "RESULTS", path)
    return {
        "harness": {"primary_weights": {"A": 1, "B": 2}, "noise_floor": 0.002, "superseded": []}
    }


def test_best_kept_ignores_blends_and_discards(toy):
    assert harness.best_kept(toy)["timestamp"] == "t2"


def test_superseded_run_leaves_the_comparison_bar(toy):
    toy["harness"]["superseded"] = ["t2"]
    assert harness.best_kept(toy)["timestamp"] == "t3"


def test_keep_rule_needs_both_margin_and_stability(toy):
    best = harness.best_kept(toy)["_primary"]
    better = {"primary": best + 0.0021, "min_fold": 0.13}
    assert harness.verdict(better, toy) == "keep"
    assert harness.verdict({**better, "primary": best + 0.0019}, toy) == "discard"
    assert harness.verdict({**better, "min_fold": 0.1275}, toy) == "discard"


def test_experiment_seeds_follow_the_confirmation_offset(monkeypatch):
    # the confirmation run only means something if every seed moves with the offset
    import importlib

    from iaa import experiment

    monkeypatch.setenv("IAA_SEED_OFFSET", "100")
    try:
        mod = importlib.reload(experiment)
        assert mod.SEED_OFFSET == 100
        assert mod.SEEDS and min(mod.SEEDS) >= 100
    finally:
        monkeypatch.delenv("IAA_SEED_OFFSET")
        importlib.reload(experiment)


def test_peril_trend_retrends_only_its_group():
    df = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "line": ["buildings", "buildings", "motor"],
            "year": [2022, 2022, 2022],
            "y_ult": [100.0, 50.0, 80.0],
            "ult_water": [40.0, None, None],
        }
    )
    cfg = {
        "predict_year": 2025,
        "trend": {"buildings": 0.05, "motor": 0.03},
        "peril_trend": {"buildings": {"water": 0.10}},
    }
    tr, _ = harness.split(df, [2022], 2024, cfg)
    f = 1.05**2
    assert tr["y"].iloc[0] == pytest.approx(60 * f + 40 * 1.10**2)
    assert tr["y"].iloc[1] == pytest.approx(50 * f)
    assert tr["y"].iloc[2] == pytest.approx(80 * 1.03**2)
