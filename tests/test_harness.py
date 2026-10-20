"""Guardrail tests on the harness's own rules (no competition data needed)."""

import csv

import numpy as np
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
    toy["harness"]["superseded"] = ["t3"]
    assert harness.best_kept(toy)["timestamp"] == "t2"


def test_superseded_run_leaves_the_comparison_bar(toy):
    toy["harness"]["superseded"] = ["t3"]
    assert harness.best_kept(toy)["timestamp"] == "t2"


def test_owner_keep_rebases_the_bar(toy):
    # t3 is an owner keep scoring below t2: the bar is the model now in use
    assert harness.best_kept(toy)["timestamp"] == "t3"


def test_keep_rule_needs_both_margin_and_stability(toy):
    toy["harness"]["superseded"] = ["t3"]  # compare against t2 (min fold 0.13)
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


def _toy_two_year_frame(rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    """All three lines, one train year (2021) and one valid year (2022). Returns the
    frame plus the valid rows' actual y_ult (line-major, matching the frame's row
    order) for a stub to predict from (fit_predict only ever sees `hide_target(va)`,
    which drops y_ult)."""
    lines = ("motor", "buildings", "contents")
    chunks = []
    for line in lines:
        for year, prefix in ((2021, "t"), (2022, "v")):
            chunks.append(
                pd.DataFrame(
                    {
                        "split": "train",
                        "line": line,
                        "year": year,
                        "row_id": [f"{line}_{prefix}{i}" for i in range(rows)],
                        "y_ult": np.linspace(50.0, 150.0, rows),
                        "exposure": 1.0,
                    }
                )
            )
    df = pd.concat(chunks, ignore_index=True)
    valid_actual = df.loc[df["year"] == 2022, "y_ult"].to_numpy()
    return df, valid_actual


def test_score_folds_sets_ae_flag_outside_band():
    """End-to-end through the real `_score_folds` and a stub experiment: no
    competition data, no model fit needed."""
    import types

    cfg = {
        "predict_year": 2099,
        "trend": {"motor": 0.0},
        "harness": {"folds": {"B": {"train": [2021], "valid": 2022}}, "primary_weights": {"B": 1}},
    }
    df, actual = _toy_two_year_frame(40)

    # row order is preserved by `split`/`hide_target`, so `actual` lines up with `va`;
    # only motor is mispriced (2x), so only its A/E (0.5) sits outside the band
    def bad_motor(tr, va):
        factor = np.where(va["line"].to_numpy() == "motor", 2.0, 1.0)
        return actual * factor

    res, _ = harness._score_folds(types.SimpleNamespace(fit_predict=bad_motor), df, cfg)
    assert res["ae_motor_B"] == pytest.approx(0.5)
    assert res["ae_buildings_B"] == pytest.approx(1.0)
    assert res["ae_flag"] == 1

    stub_ok = types.SimpleNamespace(fit_predict=lambda tr, va: actual * 1.0)
    res_ok, _ = harness._score_folds(stub_ok, df, cfg)
    assert res_ok["ae_flag"] == 0


def test_ensure_header_migrates_old_ledger_in_place(tmp_path):
    path = tmp_path / "results.tsv"
    old_fields = [f for f in harness.FIELDS if f != "ae_flag"]
    _ledger(path, [])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerow({"timestamp": "t1", "verdict": "keep", "d2_A": "0.12"})
    harness._ensure_header(path, harness.FIELDS)
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    assert rows[0]["ae_flag"] is None  # old row, field added after it was written
    assert rows[0]["timestamp"] == "t1"
    # running it again is a no-op
    harness._ensure_header(path, harness.FIELDS)
    rows2 = list(csv.DictReader(path.open(), delimiter="\t"))
    assert rows2 == rows


def test_ensure_header_preserves_crlf_data_rows(tmp_path):
    """A CRLF ledger must not have its data rows silently rewritten as LF: that would
    show up as a whole-file git diff on the next run after this change merges."""
    path = tmp_path / "results.tsv"
    old_fields = [f for f in harness.FIELDS if f != "ae_flag"]
    header = "\t".join(old_fields)
    data_row = "t1\tkeep\t\t0.12"
    raw = (header + "\r\n" + data_row + "\r\n").encode()
    path.write_bytes(raw)
    harness._ensure_header(path, harness.FIELDS)
    new_raw = path.read_bytes()
    assert new_raw.endswith(data_row.encode() + b"\r\n")
    assert new_raw.count(b"\r\n") == 2
    assert b"\n\n" not in new_raw.replace(b"\r\n", b"")  # no stray bare LF introduced
    new_header = new_raw.split(b"\r\n", 1)[0].decode()
    assert new_header.split("\t") == harness.FIELDS


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
