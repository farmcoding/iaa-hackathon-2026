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
   (ranked ideas with sources) and the current `src/iaa/experiment.py`.
2. Pick ONE hypothesis. Prefer the highest-ranked backlog idea not yet tried, or a
   follow-up that a past result clearly suggests. Do not repeat a discarded idea
   unless you change something material, and say what.
3. Edit `src/iaa/experiment.py` to test it. Keep the change small enough that a
   reviewer can read the diff in two minutes.
4. Do not run anything. You have no shell; the driver runs the frozen harness on your
   change after you finish, appends the result to `results.tsv` and commits or
   reverts. You will see the outcome in `results.tsv` on the next iteration, so read
   the code carefully: a crash costs an iteration. Check column names against
   `src/iaa/features.py` and the contract in the experiment docstring.
5. Finish with a final message whose first line is exactly:
   HYPOTHESIS: <what you are testing and why, one line, cite the backlog item or run>
   then:
   CHANGE: <what you edited>
   EXPECTED: <which fold or line you expect to move, and why>

Edit only `src/iaa/experiment.py`. Any other file you touch is reverted and the
iteration is logged as a violation.

## Rules

- Time budget: the harness marks a run `over_budget` above 900 seconds (the baseline
  takes about 150). Seed-bag
  (`SEEDS`) every model; single-seed noise is about 0.0026, bigger than most gains.
- Keep rule (applied by the harness): mean fold D² must beat the best kept run by more
  than 0.002 and the worse fold must not fall by more than 0.002.
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
- Use only open-source libraries already installed (lightgbm, glum, scikit-learn,
  numpy, pandas). Do not install packages or use the network.
