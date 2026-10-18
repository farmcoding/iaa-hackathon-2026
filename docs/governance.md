# Governance: the standards this project is run and documented against

This is a personal entry by a Fellow of the Canadian Institute of Actuaries. Competition
work isn't an actuarial engagement in the formal sense, but the project follows the CIA
standards as if it were one, because the judges assess validation, governance and
documentation (45% of the score), and because that is the professional standard the
owner works to.

Sources were gathered on 2026-10-18 by a research agent and are listed at the end. Items
marked **verify** still need a direct read of the primary text before they are quoted in
the report.

## Binding frame: CIA

| Provision | What it requires here | Project artefact |
|---|---|---|
| Rules of Professional Conduct, Rule 1 (integrity) and Rule 2 (qualification) | Do the work competently and only within qualification; be honest about what AI did | AI-use declaration in the report; `ai_log/` |
| SOP General Standards 1440, Data | Check data sufficiency and reliability; document defects and their treatment | `reports/data_quality.md`; `src/iaa/data.py`; `tests/test_core.py` |
| 1450, Models | Model appropriate for its use; understand its limitations; validate it | Frozen harness (`src/iaa/harness.py`), two out-of-time folds, CV vs leaderboard check |
| 1460 Quality assurance, 1470 Control | Checks proportionate to the work; control over the process | Pre-commit and CI gates; tests run before every loop iteration; driver reverts changes outside the experiment file |
| 1480, Reasonableness of results | Results checked against expectations | Developed 2024 must land on trend (automated test); actual-to-expected by line in every run |
| 1490, Documentation | Enough documentation for another actuary to understand and reproduce | Git history, `results.tsv`, `reports/assumptions_log.md`, README reproduction steps |
| 1600, Assumptions and methods | Assumptions justified and disclosed, with sensitivity | `configs/market_dev.yaml` with reasons in comments; `reports/assumptions_log.md` |
| 1700, Reporting (1710 external user report) | Report states scope, data, assumptions, methods, reliance, limitations and deviations | The 5-page report follows this order |
| Educational Note, Use of Models (2017) | Model risk management for the model lifecycle | Model inventory note in the report |
| Educational note on machine learning in P&C pricing (**verify** title and number; sources disagree between 217007 and 225061) | ML-specific pricing practice | Cited in methods once verified |
| CIA advice statement to the IAA, *Professional Considerations for Actuaries Interacting with an AI Model or System* (30 Mar 2026) | Professional conduct when using AI | AI-use declaration; human gate |

Section numbers follow the General Standards effective 1 Oct 2025 (revised 1 Apr 2026).
The older numbering (1500 for models, 1800 for reporting) is out of date.

## AI governance frame: IAA, CAS and others

| Source | Use in this project |
|---|---|
| IAA AI Task Force, *AI Governance Framework* (final, Dec 2025) | Structure of the governance section: roles, human oversight, risk controls. **verify** detailed provisions |
| IAA AI Task Force, *Testing of AI Models* (Dec 2025) | Frozen harness, data partitioning, robustness checks, continuous testing |
| IAA AI Task Force, *Documentation of AI Models and Systems* (Dec 2025) | Documentation set: ledger, logs, report |
| CAS, *AI Primer: Practical Guidance for Actuaries*, 2nd edition (Jul 2026), agentic AI section | Risks of agent loops and the adoption framework; basis for the loop's guardrails |
| OSFI Guideline E-23, Model Risk Management (effective 1 May 2027) | Not binding here; cited as the Canadian direction for model inventory, independent validation and monitoring |
| ASOP 23, 41, 56 (US) | Cross-reference only |

## How the agent loop is governed

1. The agent may edit one file. The data, target, development, trend, folds, scorer and
   tests are frozen; the driver reverts anything else and logs a violation.
2. Tests run before every iteration. A failing test stops the loop.
3. The keep rule is mechanical (beat the best run by more than seed noise on both folds),
   so the agent cannot argue a result in.
4. Every iteration is logged: hypothesis, change, result, and the driver's action.
5. No submission without human sign-off in `docs/SIGNOFF.md`: the reviewer reads the
   diff, the scorecard and the actual-to-expected by line, and records the reason.
6. Actuarial assumptions (development method, trend, caps) change only by human
   decision, recorded in `reports/assumptions_log.md`.

## Mapping to the judging criteria

| Criterion | Weight | Evidence |
|---|---|---|
| Pricing performance and adaptation | 40% | Leaderboard; data-quality treatment; development and trend; Part 2 drift report |
| Effective use of AI | 25% | Loop ledger and logs; research synthesis; AI-found issues with human verification (leak, BI reserves) |
| Trust, validation and guardrails | 20% | This document; tests; frozen harness; sign-off log; reproducibility |
| Communication and documentation | 15% | 5-page report in SOP 1700 order; 5-minute video; repo history |

## Sources

- CIA Standards of Practice, General Standards: https://www.cia-ica.ca/publications/226004e/
- CIA Educational Note, Use of Models (2017): https://www.cia-ica.ca/publications/217007e/
- CIA advice statement to the IAA (2026): https://www.cia-ica.ca/?p=815202
- CIA Rules of Professional Conduct: https://www.cia-ica.ca/being-a-member/rules-of-professional-conduct-and-why-they-are-important/
- IAA AITF papers: https://actuaries.org/news-post/iaa-releases-three-new-papers-to-support-responsible-ai-in-actuarial-practice/
- CAS AI Primer, 2nd edition: https://www.casact.org/article/second-edition-cas-ai-primer-now-available
- OSFI E-23: https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/guideline-e-23-model-risk-management-2027
