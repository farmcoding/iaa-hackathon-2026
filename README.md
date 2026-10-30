# Pricing a new market with an AI-assisted actuarial workflow

Entry to the IAA AI-Assisted Hackathon 2026 (Parts 1 and 2, Kaggle) by team "The KMR", a
solo entry supported by AI agents. Released under the Apache License 2.0 (`LICENSE`).

The written report is [`docs/report_v2.pdf`](docs/report_v2.pdf) (5 pages). This README
explains the competition, the results, and how to follow the work through the repository.

## The competition

The International Actuarial Association ran the hackathon in two parts on Kaggle.

- **Part 1, developed market.** Policy, customer, address, weather and claims tables for
  2021-2024, three lines: private motor, home buildings and home contents.
- **Part 2, emerging market.** The same insurer expands into a new market with two years of
  history (2023-2024) and a different schema. The 2025 book is mostly new business from a
  broker transfer.

**The task in both parts.** For every 2025 policy row in `test_motor`, `test_buildings` and
`test_contents`, predict the expected ultimate claim cost: the final cost of all claims with
a loss date in the row's period, net of the policy excess and of recoveries, excluding
declined claims. That is the risk premium, not the premium charged. Part 2 also asks for the
office premium of 20 reference policies, built from the predicted risk premium with the
supplied `expenses` assumptions (report section 8).

**The metric.** Tweedie D² with power 1.5, pooled across the three lines, against a
constant-rate baseline. Higher is better. Kaggle scores a random third of the 2025 rows on a
public leaderboard during the competition; the other two-thirds (private) decide the
ranking.

**Judging.**

| Criterion | Weight |
|---|---|
| Pricing performance and adaptation to the new market | 40% |
| Effective use of AI | 25% |
| Trust, validation and guardrails | 20% |
| Communication and documentation | 15% |

**Deliverables.** The Part 1 and Part 2 prediction files, a video of at most 5 minutes, a
write-up of at most 5 pages, and the code under version control so that the development of
the solution can be understood and reproduced.

## Results

| Submission | Validation D² | Public leaderboard | Private leaderboard |
|---|---|---|---|
| Part 1 final (commit `710775e`) | 0.1425, weighted mean of folds A and B | 0.1262 | 0.1336 |
| Part 2 selected (`o4_mech_lvl_20261028.csv`, sha256 `dabd894e8644f416`) | 0.1535 / 0.1286, level-neutral, folds B / R | 0.1662 | after close |

Fold A trains on 2021-22 and scores 2023; fold B trains on 2021-23 and scores 2024 (Part 1).
In Part 2, fold B trains on 2023 and scores 2024 and fold R trains on 2024 and scores 2023.
Every submission and its sign-off is in [`docs/SIGNOFF.md`](docs/SIGNOFF.md).

## How the work is organised

An AI agent proposes model changes, one at a time, by editing a single experiment file. A
frozen harness scores every change the same way on out-of-time folds, and a rule declared in
advance decides what is kept. I own everything that defines the problem: data
treatment, the claims target, development, trend, the folds and the scorer. The agent cannot
edit any of these. I review kept changes before they are used, and every
leaderboard submission has a sign-off row.

## Reading the work as an actuary

In this order:

1. [`docs/report_v2.pdf`](docs/report_v2.pdf): the 5-page report. Method, decisions, use of
   AI, validation, assumptions, limitations and the office premium build-up.
2. [`docs/charter.md`](docs/charter.md): purpose, scope and constraints, written before Part 2.
3. [`reports/data_quality.md`](reports/data_quality.md): each data issue found, its evidence
   and its treatment (for example the leaking customer-service field and exposures in days).
4. [`configs/market_dev.yaml`](configs/market_dev.yaml) and
   [`configs/market_emerging.yaml`](configs/market_emerging.yaml): every actuarial
   assumption (development groups, bodily-injury frequency-severity, trends, folds, keep
   rule), each with its reason. Changes are logged in
   [`reports/assumptions_log.md`](reports/assumptions_log.md).
5. The claims target, in [`src/iaa/target.py`](src/iaa/target.py) (claims matched to policy
   rows, repudiated claims excluded, net of recoveries) and
   [`src/iaa/ultimate.py`](src/iaa/ultimate.py) (chain ladder on report delay by line and
   peril group, bodily injury as ultimate count times trended severity, trend to 2025).
6. Validation, in [`src/iaa/harness.py`](src/iaa/harness.py) (folds, guardrails, keep rule)
   and [`src/iaa/score.py`](src/iaa/score.py) (Tweedie D² matched to the leaderboard).
7. The models: [`src/iaa/experiment.py`](src/iaa/experiment.py) is the Part 1 final and
   [`src/iaa/experiment_emerging.py`](src/iaa/experiment_emerging.py) the Part 2 model (O4).
   The Part 2 trends and 2025 level are applied by `scripts/part2_trend_bench.py` and
   `scripts/part2_candidate.py`.
8. The experiment record: [`results.tsv`](results.tsv) (Part 1, 166 runs) and
   [`results_emerging.tsv`](results_emerging.tsv) (Part 2, 82 runs), with the agent's
   hypothesis, scores and verdict for each; [`ai_log/`](ai_log) holds the loop transcripts
   and the diff of every run.
9. The reasoning behind each test, including the ones dropped:
   [`research/experiment_backlog.md`](research/experiment_backlog.md) (Part 1) and
   [`research/experiment_backlog_emerging.md`](research/experiment_backlog_emerging.md)
   (Part 2). [`research/sources.md`](research/sources.md) lists the literature used.
10. Review and governance: [`docs/SIGNOFF.md`](docs/SIGNOFF.md) (submissions, selection
    rules and their amendments), [`docs/governance.md`](docs/governance.md) (mapping to the
    CIA Standards of Practice and the IAA AI Task Force papers),
    [`docs/pipeline_review.md`](docs/pipeline_review.md), [`docs/questions.md`](docs/questions.md),
    and the external model reviews in [`reports/reviews/`](reports/reviews).
11. Part 2 adaptation evidence: [`reports/part2_drift.md`](reports/part2_drift.md) (what
    changed in the new market) and
    [`reports/part2_office_premium.md`](reports/part2_office_premium.md) (the 20 reference
    policies).

## Where things are

| Path | What it holds |
|---|---|
| `src/iaa/` | Data loading, target, development to ultimate, features, scorer and harness (frozen); the two experiment files are the only ones the agent may edit |
| `configs/` | Owner assumptions per market |
| `program.md`, `program_emerging.md` | The agent's instructions for each part |
| `scripts/loop.sh`, `scripts/run_batches.sh` | The experiment loop driver |
| `scripts/part2_*.py` | Part 2 analysis, the selected-file rebuild, the office premium and the report figures |
| `scripts/tabm_fit.py`, `scripts/blend_submission.py` | TabM neural network (local GPU) and the fixed-weight blend (Part 1) |
| `reports/` | Data quality, assumptions, drift, the batch dashboards (`dashboard/`), feature and explainability studies, reviews, figures |
| `docs/` | Report, charter, governance, sign-off log, questions register, video script |
| `research/` | Literature notes and the experiment backlogs |
| `tests/` | 41 tests on the data contract, target, scorer, harness and market switch |
| `RECOMMENDATION.md` | The day-one setup and participation plan (18 October) |

Competition data is not in this repository: the rules forbid redistributing it. Prediction
files are not committed either.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/). Download the competition data with the
Kaggle CLI: the Part 1 tables into `data/raw/public/` and the Part 2 emerging-market tables
into `data/part2_raw/emerging/`.

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q
```

Part 1:

```bash
# score the current experiment on folds A (2021-22 -> 2023) and B (2021-23 -> 2024)
.venv/bin/python -I -m iaa.harness "description"
# fit 2021-2024 and write the 2025 submission to submissions/
.venv/bin/python -I -m iaa.harness --final "description"
```

Part 2, the selected file:

```bash
# folds B and R for the Part 2 model
IAA_MARKET=emerging .venv/bin/python -I -m iaa.harness --module iaa.experiment_emerging "description"
# O4 refitted with the 2023-24 trends (contents 0%, motor bodily injury 12%)
IAA_MARKET=emerging .venv/bin/python -I scripts/part2_trend_bench.py contents=0.0 motor_bi=0.12 write=1
# apply the shrunk motor level and check the hash of the submitted file
.venv/bin/python -I scripts/part2_candidate.py
# office premium for the 20 reference policies, and the report figures
.venv/bin/python -I scripts/part2_office_premium.py submissions/part2/o4_mech_lvl_20261028.csv
.venv/bin/python -I scripts/part2_report_figures.py
```

Each full-dataset run holds 7-11 GB of memory, so `harness.dataset()` takes a lock and runs
one at a time. LightGBM runs are deterministic. TabM training on Apple Metal is not
bit-for-bit reproducible, so its saved predictions are the artefact of record; their hashes
are in `docs/SIGNOFF.md`. The TabM blend needs a separate GPU environment:

```bash
uv venv ~/.cache/iaa-gpu/.venv && uv pip install --python ~/.cache/iaa-gpu/.venv -e ".[gpu]"
scripts/tabm_run_folds.sh A B final      # about 25-40 minutes per fold on an M4
```

## Use of AI

Claude Code agents profiled the data, researched methods, wrote most of the code and ran the
experiment loop. Models from other providers reviewed the design and the results through
`scripts/alt_call.py`, with only aggregates and code structure sent out. Every model claim
was checked against the data before it changed anything, and I approved every
submission. Report section 5 has the detail.

## Licence

Copyright 2026 Karl Meissner-Roloff. Licensed under the Apache License, Version 2.0; see
`LICENSE`. Third-party packages keep their own licences (`pyproject.toml`, `uv.lock`).
