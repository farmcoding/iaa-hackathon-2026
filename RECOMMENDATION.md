# IAA AI-Assisted Actuarial Hackathon 2026: setup and participation plan

Competition: `iaa-ai-assisted-hackathon-2026-part-1` on Kaggle (Community, CAD $500 / $100 / $50).
Prepared 2026-10-18. Sources are the competition's own pages, pulled with the Kaggle CLI, and a profile of the downloaded data.

## 1. What is being judged

The Part 1 leaderboard decides very little on its own. Prizes are awarded after Part 2 closes, on this weighting:

| Criterion | Weight | How it is assessed |
|---|---|---|
| Pricing performance and adaptation | 40% | Private leaderboard of Part 1 and Part 2 (equal weight), plus the panel's view of modelling choices, adaptation to the new market, and treatment of data quality and uncertainty |
| Effective use of AI | 25% | Panel, from the final deliverables. AI used across EDA, features, modelling, diagnostics and testing, with evidence that it improved the work and that its output was verified |
| Trust, validation and guardrails | 20% | Panel. Automated tests, diagnostics, governance controls, reproducibility, human review and escalation |
| Communication and documentation | 15% | Panel. Video (only the first 5 minutes graded), report (only the first 5 pages graded), version-controlled repo |

So about 60% of the result is scored from material handed in at the end of Part 2, and an AI screening tool reads it first. The judges will read the git history and the experiment record as evidence of the workflow. Both need to exist from day one, because neither can be reconstructed credibly at the end.

The final deliverables, submitted through a form published on the Part 2 page: team details, the Part 1 and Part 2 prediction files, the video, the report, and the code.

## 2. The task and the metric

Predict, for every 2025 policy row in `test_motor`, `test_buildings` and `test_contents` (626,639 rows in one CSV, `row_id,prediction`), the expected ultimate claim cost for that row: claims with a loss date inside the row's period, net of excess and recoveries, excluding repudiated claims. That is a risk premium, not the charged premium.

Scoring is Tweedie D² with power 1.5, pooled across the three lines, relative to a constant-rate baseline. Higher is better. Current public leaderboard (2026-10-18): top 0.1289; host GLM sandbox 0.1056; host average-cost baseline 0.0717.

Two practical points:

- A prediction of exactly zero gives infinite Tweedie deviance for any row that has a claim. Floor every prediction at a small positive value.
- "Constant-rate baseline" is not defined precisely. The first leaderboard submission should be our own average-cost baseline, so the local scorer can be calibrated against the host's 0.0717 before any modelling decision depends on it. The calibration also settles an inconsistency: the host's `sample_submission` scores 0.0717 on the leaderboard, the same as its average-cost baseline, yet the downloaded `sample_submission.parquet` is all zeros, which a strict Tweedie scorer could not score. Either the host's file differed or the scorer floors predictions.

## 3. Rules that constrain the plan

- Submissions: the rules say 5 per day and the platform allows 10. A forum thread (#750378) asks which applies, and it is unanswered. Treat 5 as binding.
- Final submission: one per part. If none is selected, Kaggle uses the best public score, which is usually the most overfit one. Select it deliberately.
- Teams: up to 3 people, and the team must be the same in both parts. The merger deadline in the rules is still a `[DATE]` placeholder.
- Prize eligibility: members or students of an IAA Full or Associate Member Association. The CIA qualifies.
- Employment: rule 3.1.c says that if you enter within the scope of your employment, the rules bind your employer too and you warrant its consent. Decide whether this is a personal entry.
- Data: you may process the data with hosted AI tools (rule 2.5.a). You may not redistribute it. The data stays out of git; `.gitignore` already excludes `data/`, `scratch/` and `*.parquet`.
- Code: sharing code privately with other teams is prohibited. A winner must release the code under an OSI licence with no limit on commercial use, so keep dependencies OSI-licensed. A private GitHub repo during the competition is fine.
- External data and tools: allowed if publicly available at little or no cost. A Claude subscription counts as a widely used tool.

## 4. Calendar

Part 1 closes **Friday 2026-10-23 at 16:00 UTC, which is noon EDT**. Part 2 opens when Part 1 closes and runs on a separate competition page, probably for one week. The exact dates are also placeholders in the rules.

Your calendar has the Honduras trip on 23 to 26 October, and Shabbat begins Friday at sunset. In practice:

- The Part 1 model has to be frozen, submitted and selected by **Thursday 22 October in the evening**, before you travel.
- The first days of Part 2 fall during the trip. Either a teammate covers the opening, or Part 2 work starts on 27 October. The pipeline should be ready to accept the new market's data with little manual work, which section 8 covers.

That leaves about four working days for Part 1: Sunday 18 to Thursday 22 October.

## 5. What the data shows

From a profile of the downloaded files:

| Table | Rows |
|---|---|
| `policies_motor` / `_buildings` / `_contents` (2021 to 2024) | 979,968 / 707,726 / 721,617 |
| `test_*` (2025) | 254,986 / 184,155 / 187,498 |
| `claims` (valued 31 Dec 2024) | 163,150 |
| `customers`, `addresses` | 545,240, 575,983 |
| `weather` (address by year, 2021 to 2025) | 2,879,915 |

Findings that shape the approach:

1. 2024 is immature. 34% of 2024 net incurred is still on open claims, against 4% for 2021. Motor 2024 incurred is 90.7m against about 117m for 2022 and 2023 on similar exposure. Reports lag losses by up to 1,361 days (motor, 99th percentile 377 days). Raw incurred understates the ultimate the scorer uses.
2. There is a trend. Buildings net incurred rises from 54.1m (2021) to 67.4m (2024) on flat exposure, roughly 7% a year. Contents shows the same direction. Predictions have to be projected to 2025 levels.
3. Severity is heavy-tailed. Motor bodily injury has a median of 11.5k and a maximum of 11.9m. Buildings fire reaches 631k and flood 352k. Large losses need capping and a separate load.
4. 2025 weather is given. Weather is supplied for the prediction year itself, so storm, flood, hail and freeze perils can be driven by the actual 2025 conditions at each address.
5. Injected data-quality defects. About 0.4% of training rows have exposure recorded in days instead of years (values 365, 89, 292 and so on), and about 0.3% show 1.01. The test files are clean. The fix is to recompute exposure from `period_start` and `period_end`. Other defects: 2.5% missing values spread evenly across unrelated columns, which looks injected (for example `vehicle_age`, `smoke_alarms`, `security_level`); 274 claims with 2020 loss dates that match no policy row; one non-positive net claim. Each should be documented as a data-quality finding with its treatment, since the judges score exactly this.
6. `renewal_decision` is train-only and describes what happened after the period. It must not be used as a feature.
7. About 85% of 2025 policies were also in force before 2025, so each policy's own claims history (and the customer's) can be used as a credibility-style feature, provided it is built only from claims reported before the row starts.
8. Size. About 190 MB of Parquet. The whole job fits on the Mac mini (M4, 16 GB). LightGBM and a GLM library train in minutes. Kaggle notebooks and GPUs aren't needed.

## 6. Recommended modelling approach

### 6.1 Target construction (most of the achievable score is here)

1. Join claims to policy rows on `policy_id` with `period_start <= loss_date <= period_end`. Drop repudiated claims. Net amount is `incurred_to_date - recoveries`. A test then requires every non-repudiated claim to match exactly one policy row. No match is an orphan (the 274 claims dated 2020 are), and two matches means overlapping mid-term-change periods. Orphan counts by loss year go in the data-quality log. One quirk is already checked: 2024 rows end on 30 December because the generator treats a year as 365 days, but no claims are dated 2024-12-31, so nothing is lost there.
2. Develop to ultimate by line and peril group: incurred development factors for open claims (IBNER) and a reporting-lag distribution for claims not yet reported (IBNR), fitted on loss-quarter maturity. Apply them as row-level adjustments or as weights, and check that line totals reconcile to a chain-ladder ultimate.
3. Trend frequency and severity to 2025, by peril, with the fitted rates stated in the report.
4. Cap large losses (motor bodily injury first, possibly buildings fire and flood) at a threshold chosen from the data. Spread the excess back as a load by line, or model it with a separate large-loss frequency.
5. Balance: for each line, the sum of predictions on a validation year should equal its developed and trended ultimate. D² against a constant-rate baseline punishes level error as much as ranking error.

### 6.2 Models

- Baseline: Poisson frequency by Gamma severity GLM per line, on the main rating factors. It is explainable and is the reference every challenger has to beat.
- Main challenger: frequency and severity by peril, with gradient-boosted models (LightGBM, monotone constraints where actuarially justified, such as excess and NCB). Peril-level models let 2025 weather act on the perils it drives.
- Ensemble partner: a direct Tweedie GBM with power 1.5, the same loss the scorer uses.
- Blend with weights fitted on the out-of-time validation, then re-balance per line.
- Features: customer and address joins, the row-year's weather (the training row's own year, never a later one), multi-line holdings, lagged policy and customer claim history, and exposure as an offset.

### 6.3 Validation

Use a time-based split that mirrors the real task. Fit on 2021 to 2022 and score 2023, then fit on 2021 to 2023 and score 2024, both against developed targets. Use `sklearn.metrics.d2_tweedie_score(power=1.5)` pooled across lines, the way the leaderboard does, and also report it per line. After the scorer calibration (section 2), spend two or three leaderboard submissions early to confirm that local validation moves with the public score. After that, use the leaderboard only as an occasional check. The private leaderboard uses a different sample, so tuning to the public score overfits.

## 7. The experiment loop (Karpathy-style)

Karpathy's `autoresearch` pattern fits this competition, and it supplies the evidence for the AI and guardrails criteria (45% of the score):

- A human-written `program.md` sets the research direction and the rules the agent must follow.
- The agent may edit one experiment file (features, model configuration, blend). It cannot edit the data preparation, the target construction or the scorer, which are frozen and covered by tests.
- Each run has a fixed time budget and writes one row to `results.tsv`: commit hash, hypothesis, local D² overall and by line, runtime, keep or discard.
- An improvement is kept as a git commit. Anything else is reverted. The git log becomes the research record.
- Human gate: no leaderboard submission without a person reviewing the diff, the diagnostics and the guardrail checks, and stating the reason for the submission.
- Guardrails that run on every experiment: leakage tests (no claim after the row's start in history features, no `renewal_decision`, no future weather), prediction floor and range checks, line-level balance, stability across the two validation years, and a cap on how much any single feature can move.
- Every AI interaction that changes the model is logged (prompt summary, what was accepted, how it was verified). That log goes straight into the report and the video.

The agent runs headless through `claude -p` using a task profile from `llm-profiles`. The model ID never appears in the repo. The competition data is synthetic and the rules allow hosted AI processing.

## 8. Building for Part 2 now

Part 2 brings an "emerging market" dataset with different characteristics. The judges want to see how the team spots what changed, adapts, and communicates the remaining uncertainty. Build for that in Part 1:

- Put every market-specific choice (paths, column mappings, development factors, trend, caps, thresholds) in a per-market config, not in code.
- Make drift detection a pipeline step: per-feature distribution shift, adversarial validation between markets, and a missing-value and schema diff, with a generated report.
- Plan to use the Part 1 model as a prior (an offset), with credibility weighting toward the new market's own experience as its data supports it.
- Report uncertainty explicitly, for example bootstrap intervals on line-level ultimates and sensitivity to the development and trend assumptions.

## 9. Workspace setup

Already done:

- Kaggle CLI 2.2.4 installed with `uv tool install kaggle`. The API token is stored in `~/.kaggle/access_token` (mode 600), outside every repo. Default competition is set, so `kaggle competitions submit -f <file> -m "<message>"` works from anywhere.
- Data downloaded to `~/Projects/iaa-hackathon-2026/data/raw/public/`.
- `git init` with the data folders ignored. The weekly PR sweep (Sunday 19:00) will now see a repo with only this file and `.gitignore`, and no data.

Proposed layout, built with the `project-scaffold` skill once the tooling details are settled:

```
iaa-hackathon-2026/
  pyproject.toml            uv, Python 3.12, pinned deps (OSI licences only)
  configs/market_dev.yaml   per-market assumptions (section 8)
  src/iaa/
    data.py                 load + validate (pandera schemas, exposure fix)
    target.py               claim join, development, trend, capping   [frozen]
    score.py                Tweedie D² exactly as the leaderboard     [frozen]
    features.py
    models/                 glm.py, gbm_peril.py, tweedie.py, blend.py
    experiment.py           the one file the agent edits
  program.md                instructions for the experiment loop
  results.tsv               experiment ledger (committed)
  tests/                    target reconciliation, leakage, scorer, schema
  reports/                  EDA, data-quality log, drift report, final report
  ai_log/                   record of AI interactions and their verification
  data/                     gitignored
```

Kaggle habits worth adopting:

- Every submission's message records the commit hash and local scores. `kaggle competitions submissions` then serves as a second ledger.
- Read the competition's Discussion tab every day. Hosts answer rule questions there and post the Part 2 link.
- Pick the final submission by validation performance and robustness, not by public rank.

## 10. Actuarial skills

No actuarial modelling skill is installed under `~/.claude/skills` or the plugins. The closest material is in the vault, and it will be used through `vault-recall`:

- `concepts/insurance_pricing_gbm_tuning`: Optuna/TPE tuning with cross-validation plus a separate held-out set
- `concepts/actuarial_reserving_credibility`: credibility and mixed-model references for the development and blending steps
- `concepts/cas_research_resources`: open-source actuarial Python stack (`chainladder`, and others)
- `concepts/insurance_pricing_fairness`: proxy-discrimination checks, relevant to the governance section
- `concepts/ai_vs_ml_vocabulary_actuarial` and `concepts/graph_engineering_software_factory`: the governance framing (deterministic numbers, maker-checker, the actuary signs off)

After the competition, this project is a good source for a reusable P&C pricing skill.

## 11. Open questions

1. Travel and Shabbat: Is the Part 1 freeze on Thursday 22 October right? Who, if anyone, covers the opening of Part 2 during 23 to 26 October?
2. Team: Solo or with teammates (up to 3)? Teammates must join before the merger deadline, which hasn't been published.
3. Eligibility: Are you entering as a CIA member, and as a personal entry (rule 3.1.c)?
4. Submissions: Is 5 per day acceptable as the working limit? May I make the calibration submission (average-cost baseline) this weekend?
5. Actuarial skills: If you have actuarial skills elsewhere (claude.ai, another machine, a repo), where are they? Otherwise the vault material above is the base.
6. Tooling: Your follow-up will cover this. Specifically: GLM library (`glum` or `statsmodels`), GBM library, experiment tracking (plain `results.tsv` or MLflow), and whether the repo goes to a private `farmcoding` GitHub repo now.

Rotate the Kaggle API token once setup is finished, because it has appeared in chat. The new token goes in the same file, `~/.kaggle/access_token`.
