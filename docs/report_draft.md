# Pricing across two markets with an AI agent inside an actuarial harness

IAA AI-Assisted Hackathon 2026, team "The KMR" (solo entry, FCIA). Draft 1, 28 October 2026.
Numbers are from the repository: `docs/SIGNOFF.md`, `reports/assumptions_log.md`,
`results.tsv`, `results_emerging.tsv`.

## 1. Summary

The task is the expected ultimate claim cost of each 2025 policy row for private motor,
home buildings and home contents, scored by Tweedie D² (power 1.5). Part 1 used four years
of a developed market. Part 2 moved to an emerging market with two years of history and a
2025 book that is 77% new business from a broker transfer.

The entry is built around one idea: an AI agent proposes changes to one model file, and a
frozen harness that the agent cannot touch decides what is kept. Data, target, development,
trend, validation folds, scoring and tests are actuarial decisions. They are written in
config files and code that the agent may not edit, and every leaderboard submission was
signed off by the actuary.

| | Internal validation | Public | Private |
|---|---|---|---|
| Part 1 final (710775e) | 0.14252 (folds 0.14234 / 0.14261) | 0.12622 | 0.13363 |
| Part 2 current selection (O4, 0c4e403) | 0.12217 (deploy metric, folds B / R 0.14199 / 0.10236) | 0.16326 | after close |

## 2. Data and target (both parts)

**Data quality.** AI profiling found and the actuary confirmed eight findings
(`reports/data_quality.md`). Two would have broken the model. Customer-service contacts
include claim calls in training but not in the test files. It was the strongest predictor
(72% of gain) and pure leakage, so it is excluded. About 0.4% of exposures were recorded
in days and are recomputed from the dates. A further 410 claims match no policy period,
`renewal_decision` exists only in training, and `customer_since_year` drifts by
construction; all are treated and tested.

**Target.** The score is on ultimate cost, so the training target is developed. Reported
net cost is developed by a chain ladder on report delay, one triangle per line and peril
group. Motor bodily injury needed more in the developed market: same-delay claims got
cheaper in younger years while other perils inflated, so case reserves were light, and
bodily injury is projected as ultimate count times the trended severity of the most mature
year. It is capped at 1m per claim with the excess loaded from a fitted Pareto tail. The
check that runs automatically: developed 2024 has to land on the 2021-23 trend. Raw motor
was 12% short; developed, it is within 2%. Trends are line-level (motor 3%, contents 5%)
and per peril on buildings (weather and water 9%, fire and accidental damage 4%).

## 3. Part 1 model and validation

Per line: motor bodily injury is a Poisson frequency model with a flat severity. Other
motor perils and the property lines are blends of LightGBM, an Explainable Boosting
Machine and (property) CatBoost, with Tweedie loss on the rate, credibility target encoding
of high-cardinality fields and monotone constraints on excesses, no-claims years and
flood zone. The actuary excluded the incumbent's written premium (an output of another
tariff, carrying experience re-rating), email domain, sales and admin fields and
customer-level claim history on admissibility grounds, not score.

Validation is out of time: fold A trains 2021-22 and scores 2023, fold B trains 2021-23 and
scores 2024. A change is kept only if it beats the best run by more than seed noise on both
folds and reproduces on fresh seeds. The local score tracked the board from the first
baseline (0.1056 local, 0.1060 public). Nine submissions were signed off; each row in
`docs/SIGNOFF.md` records the commit, the file hash, the fold scores and the reason.

**What Part 1 taught Part 2.** After the close, eight late submissions (scored, not ranked)
tested only the line levels. Anchoring each line to its historical loss cost put motor
about 10% high. Balancing on the training rows instead, so that the 2025 mix moves the
level through the model's relativities, would have scored 0.13438 rather than 0.13363.
That method was adopted for Part 2 before any Part 2 model was scored.

## 4. Part 2: what changed, and what was carried over

**What changed** (`reports/part2_drift.md`). At the same maturity, emerging severity is
0.6-0.8 of developed, and bodily injury 0.45-0.62 (largest claim 156k against 11.9m). Motor
frequency is 1.2-1.8 times developed, glass and hail the highest. Storm is 1.3-1.7 times and
escape of water 0.6-0.7. Hail days run 14-17 against about 6. The schema lacks credit band,
mileage, telematics and date of birth (age bands only), adds gender, and has 12% missing on
several fields. The 2025 book is 77% new business against 6-9% in history, and half its
drivers are 17-34 against 38%.

**What transferred.** The Part 1 specification refitted on emerging data alone scored
0.12904. The same specification fitted on the developed market and applied to emerging
rows scored 0.14243. Developed relativities carry over better than two small years can
estimate them, so every candidate builds on a developed base. Development patterns are
borrowed from the developed market (both markets are valued 31 December 2024, and report
shares and open shares match at the same maturity). Bodily injury is developed by cost
chain ladder, since no emerging year is mature enough to anchor a severity. Trends are
carried over because two years cannot separate trend from weather.

**How the model was adapted.** Each step below was tested on the folds and then once on the
board:

| Step | Public |
|---|---|
| One rate per line (plumbing check) | 0.08396 |
| Developed base, historical level | 0.14243 |
| Same base, level from the 2025 mix (balance on train) | 0.15079 |
| P0: one developed refit per peril, summed | 0.15195 |
| L1: + motor gender (ridge offset GLM, likelihood-ratio gated) | 0.15933 |
| O4: + per-peril emerging levels (credibility k=100), contents 50:50 with a pooled two-market base, new-business claim-count factor (motor, contents) | **0.16326** |

Per-peril bases let each peril take its own emerging level, which moves the relativities
towards the risks exposed to the perils that are heavier here. Gender is admissible in this
market per the brief (owner decision); the developed base has none, and it mispriced men and
women by 12-17% on motor in both years.

**Validation for a broker book.** Two folds on two years: B trains 2023 and scores 2024, R
trains 2024 and scores 2023. The metric is D² at the production level, two-thirds on the
natural book and one-third on the book reweighted to the 2025 mix. A keep needs a gain over
0.002, neither fold down more than 0.003, and a paired bootstrap whose 10th percentile is
above zero.

## 5. Use of AI

**Where AI did the work.** Claude Code agents profiled the data, researched methods
(Richman and Wüthrich, past Kaggle tabular winners, UK bodily-injury law), wrote most of the
code and ran an experiment loop modelled on Karpathy's autoresearch. A headless agent edits
only the experiment file and states a hypothesis; the driver reverts anything else, runs the
frozen harness and writes the ledger. The loop ran 233 logged iterations: in Part 1, 166
experiments (10 loop keeps, 12 owner keeps); in Part 2, 82 (9 and 2). Models from other
providers reviewed the design (OpenAI GPT-6.1 Sol on 25 October; on 27 October four models
generated ideas and three reviewed the results adversarially, for $2.90). Heavy refits ran
on private Kaggle notebooks, including a GPU neural-network base.

**Where the human caught the AI.** Three cases changed the method.
- On 25 October the loop kept eleven small factors, stacked in pairs that each just missed
  the margin. The folds gained +0.0068 and the board lost 0.0029, because every factor was
  chosen on the claims it was scored on. The actuary replaced the keep rule with the deploy
  metric and bootstrap, required a pre-registered list, one mechanism per run, and a
  claim-count gate on every factor, and banned stacking discarded runs.
- An external model built a recommendation on 2023 being all new business. It is 6.4%.
- Reviewers caught that a claim-count model offset by a cost prediction estimates frequency
  relative to cost and learns inverse-severity effects; an earlier pooled base had weighted
  rows by duplicating them, which let the trees memorise claims. Both were fixed.

**What did not work, and why that is evidence.** On 27-28 October 20 pre-registered
mechanisms and five new base families were tested against O4: count-based correction
blocks, a hierarchical new-business layer, shape and temperature recalibration, weather-year
factors, previous insurer, region, missingness, self-employment, sample-weighted pooled
bases, in-band age averaging, square-root-exposure weights and a TabM neural base. None met
the rule (`research/experiment_backlog_emerging.md`, tier N). The pattern was consistent:
anything learned on emerging rows helped the forward fold a little at best and hurt the
reverse fold every time; anything that replaced the developed base hurt the forward fold.
O4 was kept.

## 6. Guardrails and reproducibility

The harness, target, development, trend, folds and scorer are frozen for the agent and
covered by 41 automated tests that run before every loop iteration. Pre-commit runs ruff and
gitleaks, and CI repeats them with the tests. Market assumptions live in one config file
per market with the reason beside each value. The final-selection rule for each part was
declared before any candidate under it was scored. Every submission has a sign-off row.
After three full-dataset runs exhausted memory on 25 October, the harness took a
one-process lock and the loop a disk margin. LightGBM runs are deterministic; the neural
predictions are saved with their hashes as the artefact of record. The work follows the CIA
Standards of Practice (1440 data, 1450 models, 1490 documentation, 1600 assumptions, 1700
reporting) and the IAA AI Task Force papers on governance, testing and documentation
(`docs/governance.md`).

## 7. Assumptions, limitations and uncertainty

| Item | Effect |
|---|---|
| Case reserves on open claims taken as adequate | +/-10% moves the 2025 motor level +/-3.4%, buildings +/-1.9% |
| Bodily-injury development tail | +/-5% moves motor +/-1.9% |
| New business priced about 1.27 times renewal on motor (developed relativity times an emerging factor) | Raises the 2025 motor level 17%; the developed market's own new-business loss-cost relativity is 1.29-1.51 at every exposure length, so the level is defensible, but the emerging evidence is thin (new-business A/E 0.92 and 1.75 on the two folds) |
| Buildings level | Dominated by fire: 49 claims in 2023, 21 in 2024; A/E 0.60 and 1.70 on the folds; a 10% buildings level error is worth 0.004-0.005 of D², motor 10% about 0.001 |
| Trends carried over from the developed market | Two emerging years cannot separate trend, weather and development |
| Weather year | 2025 is milder (11 hail days); developed elasticities put the effect under 1% on any line, so it is not modelled |

Validation limits: each fold trains on one emerging year while the final fit uses two, and
the reverse fold trains on the less developed year, so it penalises local fitting more than
production would. Neither fold contains a broker book. A grouped multi-year design is the
first change to make with more time.

## 8. Office premium for the reference policies

Office premium = (risk premium + fixed expense + cat reinsurance) / (1 - variable expense -
commission - premium tax - profit margin), with the emerging assumptions: fixed 85 CU per
policy, cat reinsurance 12% of the property risk premium, variable expense 15%, commission
20% broker and 0% direct, premium tax 8%, profit margin 8%. Broker business carries 51% of
premium in proportional loads. The fixed expense dominates contents, where risk premiums are
40-120 CU. Full table, every component: `reports/part2_office_premium.md`.

| Policy | Line | Channel | Risk premium | Office premium | Loss ratio |
|---|---|---|---|---|---|
| M00161473 | motor | broker | 1,203.63 | 2,629.86 | 45.8% |
| M00185311 | motor | direct | 798.69 | 1,280.71 | 62.4% |
| B00217584 | buildings | broker | 866.82 | 2,154.77 | 40.2% |
| B00215809 | buildings | direct | 449.67 | 853.09 | 52.7% |
| C00286287 | contents | broker | 118.40 | 444.11 | 26.7% |
| C00270331 | contents | broker | 38.60 | 261.71 | 14.8% |
