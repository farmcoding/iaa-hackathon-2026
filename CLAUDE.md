# iaa

IAA AI-Assisted Hackathon 2026 entry: predicts the 2025 expected ultimate claim cost
(risk premium) per policy row for motor, buildings and contents, in a developed market
(Part 1) and an emerging market (Part 2). See README.md for the competition and results.

## Architecture

> Keep this section current — it is the living architecture doc the `spec-first`
> skill reads before non-trivial changes and updates after. Describe the real
> system, not aspirations.

- **Entry points:** `python -m iaa.harness [--module iaa.experiment_emerging] [--final]
  "description"` (`IAA_MARKET=emerging` for Part 2); `scripts/loop.sh` drives the agent
  loop; `scripts/part2_trend_bench.py` + `scripts/part2_candidate.py` rebuild the Part 2
  selected file.
- **Data flow:** `data.py` (raw parquet per market, cleaning) -> `target.py` (claims to
  policy rows, net, repudiated excluded) -> `ultimate.py` (chain ladder by line and peril
  group, BI frequency-severity, trend) -> `features.py` -> experiment file `fit_predict` ->
  `score.py` (Tweedie D2, p=1.5) -> `results*.tsv` ledger.
- **Interface contracts / invariants:** `fit_predict(train, valid)` returns finite,
  non-negative expected ultimate cost per valid row in order. Only the experiment files are
  agent-editable; data, target, ultimate, features, score and harness are frozen. Keep rule
  and folds live in `configs/market_*.yaml`. Every leaderboard submission has a row in
  `docs/SIGNOFF.md`.
- **Failure modes:** memory (7-11 GB per full-dataset run; `runguard` lock); the first
  run after a cache rebuild can crash in the target encoder (rerun); TabM on Metal is not
  bit-reproducible (saved predictions are the record).
- **External services & how credentials are handled:** Kaggle CLI (its own config);
  OpenRouter for outside-model reviews via `scripts/alt_call.py`, key from the environment
  or a gitignored `.env`; models are named by llm-profiles, never literal ids.

## Conventions

- Python 3.12, `uv`-managed venv. Deps pinned in a lockfile before first deploy.
- Secrets via `.env` (chmod 600) or Keychain — never literals in source.
- New endpoints authenticate by default; outbound calls set explicit `timeout=`.
- Quality gates: `pre-commit` (ruff + gitleaks) locally, `quality.yml` in CI
  (ruff + gitleaks secrets + pytest). Run `pre-commit install` on a fresh clone.
- Deploying to Fly.io? Follow the `flyio-deploy` skill — pre-flight checklist, `fly.toml` template, and lessons learned.
- One full-dataset process at a time (7-11 GB each on a 16 GB machine). `harness.dataset()`
  takes the `iaa.runguard` lock, so a second harness/explain/analysis run waits for the
  first. Don't work around it with `IAA_RUN_LOCK=off` or parallel launches; on 25 Oct three
  concurrent runs exhausted swap. `scripts/loop.sh` also stops below 3 GB free disk.

## Run / test

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q
```
