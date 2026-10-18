# program.md: instructions for the experiment agent

You are running one experiment in an autoresearch-style loop for an actuarial pricing
model. A human actuary (FCIA) owns this work, sets the direction in this file, and
approves every leaderboard submission. You propose and test; you never submit.

## The task

Predict expected ultimate net claim cost per 2025 policy row for three lines (motor,
buildings, contents). The score is Tweedie D² with power 1.5 against a constant mean,
pooled across lines, exactly as the Kaggle leaderboard computes it (verified: local
cross-validation 0.1056 vs public leaderboard 0.1060 for the baseline).

## What you may change

Only `src/iaa/experiment.py`. Its contract is in its docstring:
`fit_predict(train, valid) -> np.ndarray`. Everything else is frozen: data loading,
target construction, development to ultimate, trend, folds, scoring, the harness, the
tests and the config. The loop driver reverts the run and logs a violation if any
other file changes. Do not try to improve the score by changing the target, the folds
or the scorer; those are actuarial assumptions owned by the human.

## One iteration

1. Read `results.tsv` (all past runs and verdicts), `research/experiment_backlog.md`
   (ranked ideas with sources; items marked OWNER are human-only, skip them) and the
   current `src/iaa/experiment.py`.
2. Pick ONE hypothesis. Tier 0 of the backlog (re-tests) comes first, in its order.
   Otherwise prefer the highest-ranked backlog idea not yet tried, or a
   follow-up that a past result clearly suggests. Do not repeat a discarded idea
   unless you change something material, and say what.
3. Edit `src/iaa/experiment.py` to test it. Keep the change small enough that a
   reviewer can read the diff in two minutes.
4. Do not run anything. You have no shell; the driver runs the frozen harness on your
   change after you finish, appends the result to `results.tsv` and commits or
   reverts. You will see the outcome in `results.tsv` on the next iteration, so read
   the code carefully: a crash costs an iteration. Runs marked `keep_owner` are human
   decisions (e.g. monotone constraints): build on them, don't remove them. Check column names against
   `src/iaa/features.py` and the contract in the experiment docstring.
5. Finish with a final message whose first line is exactly:
   HYPOTHESIS: <what you are testing and why, one line, cite the backlog item or run>
   then:
   CHANGE: <what you edited>
   EXPECTED: <which fold or line you expect to move, and why>

Edit only `src/iaa/experiment.py`. Any other file you touch is reverted and the
iteration is logged as a violation.

## Rules

- Time budget: the harness marks a run `over_budget` above 1800 seconds (the current
  model takes about 230). Seed-bag
  (`SEEDS`) every model; single-seed noise is about 0.0026, bigger than most gains.
- Keep rule (applied by the harness): the primary score, a weighted mean of fold D²
  with fold B (score 2024) counted twice because it tracks the public leaderboard, must
  beat the best kept run by more than 0.002, and the worse fold must not fall by more
  than 0.002. Gains that only show in fold A have not reached the leaderboard.
  A run that passes is re-run with every seed shifted (`IAA_SEED_OFFSET`) and must pass
  again; otherwise it is logged `unconfirmed`. So every random choice you add (model
  seeds, sampling, CV splits, encoders) must derive from `SEED_OFFSET` or `SEEDS` in
  `experiment.py`. A model with a fixed literal seed cannot be confirmed.
- Discarded changes are saved as patches in `ai_log/diffs/` (from 2026-10-18 16:09);
  read one before re-testing an idea.
- The driver runs a 20-second smoke test on a subsample first. A smoke crash is
  recorded as `crash` with the error text in the ledger description; read it before
  retrying the idea.
- When every backlog item is tried, parked or owner-only and no follow-up is
  justified by the ledger, finish with exactly `HYPOTHESIS: NONE` and make no edit.
  The loop then stops.
- Leakage: never use `renewal_decision`, `customer_service_contacts`, any target
  column (`cost_*`, `n_*`, `ult_*`, `comp_*`, `y`, `y_ult`) or the row's year as a
  feature. Encodings fitted on the target must be fitted inside `fit_predict` on the
  training frame only, out of fold.
- Keep the level anchoring (or an equivalent balance step) unless the experiment is
  about the level itself; actual-to-expected per line should stay within 0.97 to 1.03.
- Portability: Part 2 brings a new market with different data. Prefer features and
  methods that would transfer (rates, ratios, generic encodings) over hard-coded
  levels, thresholds or category lists from this market.
- Prediction must be finite and non-negative for every valid row.
- Use only open-source libraries already installed: lightgbm, catboost, xgboost,
  interpret (EBM: `from interpret.glassbox import ExplainableBoostingRegressor`), glum,
  scikit-learn, numpy, pandas. All were licence-checked and approved. Do not install
  packages or use the network.
- CatBoost: pass `allow_writing_files=False` (otherwise it writes `catboost_info/` into
  the repo and the driver stops), and convert categorical NaN to a string such as
  `"missing"`. The smoke test allows 300 s on 50k rows per line across both folds and
  all seeds; keep iterations and threads modest so it fits.
