# IAA AI-Assisted Actuarial Hackathon 2026, Part 1

Solo entry (team "The KMR") by a Fellow of the Canadian Institute of Actuaries. The task is
to predict the expected ultimate claim cost of each 2025 policy row for motor, buildings
and contents. Entries are scored by Tweedie D² with power 1.5.

The project runs an AI agent loop inside a frozen actuarial harness. The agent proposes
one change at a time to the model file. The harness scores every change the same way, on
two out-of-time folds, and a mechanical rule decides what is kept. Actuarial assumptions,
submissions and the final selection are human decisions, recorded in the repo.

Competition data is not in this repository (the rules forbid redistribution).

## Where things are

| Path | What it holds |
|---|---|
| `configs/market_dev.yaml` | Owner assumptions: development groups, bodily injury frequency-severity, trend, folds, keep rule |
| `src/iaa/` | Data loading, target matching, development to ultimate, features, scorer and harness (frozen); `experiment.py` is the only file the agent may edit |
| `program.md`, `scripts/loop.sh`, `scripts/run_batches.sh` | Agent instructions and the loop driver |
| `results.tsv`, `ai_log/` | Every experiment: hypothesis, scores, verdict, driver action |
| `scripts/tabm_fit.py`, `scripts/blend_submission.py` | TabM neural net (local GPU) and the fixed-weight blend with LightGBM |
| `reports/` | Data quality, assumptions log, dashboards after each batch, variable study |
| `docs/` | Governance, sign-off log, questions register, deck outline, video notes |
| `research/` | Literature notes and the experiment backlog |

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/). Download the competition data with the
Kaggle CLI into `data/raw/public/`.

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q

# score the current experiment on folds A (2021-22 -> 2023) and B (2021-23 -> 2024)
.venv/bin/python -I -m iaa.harness "description"
# fit 2021-2024 and write the 2025 submission to submissions/
.venv/bin/python -I -m iaa.harness --final "description"
```

The TabM blend needs a separate GPU environment (PyTorch with Apple Metal):

```bash
uv venv ~/.cache/iaa-gpu/.venv && uv pip install --python ~/.cache/iaa-gpu/.venv -e ".[gpu]"
scripts/tabm_run_folds.sh A B final      # about 25-40 minutes per fold on an M4
.venv/bin/python -I scripts/blend_submission.py --lgbm-run <run_id> \
    --final --lgbm-sub submissions/<harness final file>.csv
```

LightGBM runs are deterministic. TabM training on Metal is not bit-for-bit reproducible,
so the saved TabM predictions are the artefact of record; their sha256 hashes are in
`docs/SIGNOFF.md`.

The first harness run after the dataset cache is rebuilt can crash in the target encoder
(known issue in `reports/data_quality.md`); running it again works.
