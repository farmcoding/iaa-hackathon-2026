# Owner runs, 20 October afternoon

Each run was scored on both folds and compared with the champion of the time on the same actuals (fold A scores 2023, fold B scores 2024; primary = (A + 2B) / 3). Seed noise: primary sd 0.0002, fold B sd 0.0001. None of these wrote a ledger row except the three owner keeps.

## Adopted (owner keeps)

| Change | Primary | Fold A | Fold B | Why |
|---|---|---|---|---|
| Starting point (per-peril buildings trends) | 0.14260 | 0.14256 | 0.14262 | |
| No customer-level claim history (3.45) and buildings `construction_type` labels harmonised | 0.14266 | 0.14262 | 0.14268 | Simpler and fairer (no pooling across a customer's lines); one category was recoded T to W between extracts and labels had case variants |
| Motor credit band ordered and monotone; distance to river dropped from motor (constraints option 2b) | **0.14252** | 0.14234 | 0.14261 | Defensible relativities; fold B cost -0.00007, inside seed noise |

## Tested, not adopted

| Run | Primary vs champion | Fold B vs champion | Finding |
|---|---|---|---|
| 3.45 same-line claim history only (buildings) | +0.00005 | +0.00012 | Neutral; dropping customer history entirely was simpler and slightly better |
| 3.1 GLM baseline (Tweedie, main effects, per line) | 0.12678 | 0.12377 | The actuarial benchmark: 0.016 below the champion; motor close on fold A (0.0867 vs 0.0885), well behind on fold B and on property |
| 3.34 claim-history cost capped at the line's 95th percentile | -0.00004 | -0.00003 | Neutral; a defensibility change for Part 2 (training history is more developed than 2025's) |
| 3.27 fold B trained on targets as known at 31 Dec 2023 | | -0.00059 | Fold B is optimistic by about 0.0006 because its training targets use the 2024 valuation; motor A/E 0.984 vs 1.011 |
| Capped BI target (3.5 / 3.37 in the target) | -0.00008 | -0.00008 | The model already neutralised the 11.9m claim; BI level about 2% higher |
| Constraints with CatBoost/EBM wiring only (control) | -0.00026 | -0.00026 | Passing constraints to CatBoost and the property EBM costs on its own |
| All explainability constraints | -0.0007 to -0.0012 | -0.0001 to -0.0007 | Motor fold A -0.0035; wildfire constraint costs contents |
| Constraints option 2 (credit band on all lines, river off motor) | -0.00027 | -0.00043 | Property credit constraint costs about 0.0004 on fold B for a field adding 0.0004-0.0006 |
| 3.31 TabM, 3 seeds, last-year early stopping (fold B) | | blend +0.00000 at 10% | TabM alone 0.1350; no blend weight helps |

## Diagnostics and reports

- Evidence pack (3.6, 3.24, 3.7, 3.47): `2026-10-20_evidence/report.md`.
- Deeper reserving (3.2, 3.4, 3.10, 3.40): `2026-10-20_reserving_deeper/report.md`. No change to the target from any item.
- Governance (3.35, 3.28, 3.23, 3.30): `docs/charter.md`, `docs/feature_registry.md`, `tests/test_data_contract.py`, ledger `ae_flag`, `use_premium` flag. Merged; the champion reproduced exactly on the merged code.
