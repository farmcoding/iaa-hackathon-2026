# Meta-review: why the field scores higher, and why the loop stalled

Prepared 2026-10-19 evening, read-only against `~/Projects/iaa-hackathon-2026` (ledger through the 18:42 row, champion run `20261019T172626`, primary 0.14048, public 0.12476). Every number below comes from the saved fold predictions in `data/cache/preds/`, the cached dataset, the ledger, the loop logs and the Kaggle pages. Scripts and raw output are in `~/.cache/iaa-owner/1019/meta/` (`decomp.py`, `paired.py`, `segments.py`, `drift.py`, `rowstruct2.py`, `reported.py`, `ledger.py`).

## 1. Why the rest of the field predicts better

### What the leaderboard is

The public board is scored on about 33% of the 2025 test rows (roughly 207,000 of 626,639); the private board is the other 67%. The truth is the host's ultimate cost per row, net of excess (the host confirmed this week that the excess applies to every peril) and net of recoveries, excluding repudiated claims. Scoring is Tweedie D² at power 1.5 against a constant baseline, pooled across lines. There are no public notebooks and no discussion thread on method; the only threads are the submission limit (now 10 a day) and the excess question. Nothing on Kaggle says what the top teams do.

Standings tonight: leader 0.12996 (14 entries), 13 teams above 0.1250 and 14 above us, us 0.12476 at rank 15 with 8 entries. The host's GLM sandbox scores 0.10563.

### The local-to-public offset is not the problem

Fold B scores 0.1415 and the public board 0.1248. That 0.017 is not evidence of overfitting; it is three unrelated things:

- Target scale. Fold B scores against developed 2024 targets (reported cost divided by completion, which is 0.74 for motor, 0.92 for buildings). Scored on the reported scale instead, the same predictions give 0.1336 (`reported.py`). About half the offset is the construction of the local truth, not the model.
- Sampling. D² on a 207,000-row sample of fold B has a standard deviation of 0.009 (heavy-tailed claims). The absolute public number is a draw.
- Year. 2025 is a different year with its own large claims.

What is diagnostic is gain tracking, and `docs/SIGNOFF.md` shows fold B gains carrying to the public board at 0.5 to 1.3 times across five submissions (B +0.0053 / public +0.0060; +0.0090 / +0.0072; +0.0036 / +0.0039; +0.0008 / +0.0003; +0.0016 / +0.0014). Fold B is a sound currency. To reach 0.130 public, fold B needs about +0.005.

### Level is not where the gap sits

With oracle line levels (each line rescaled to its actual fold total) the champion gains +0.0001 on fold A and +0.0000 on fold B. A 10% level error on motor costs 0.0011, on buildings 0.0007, on contents 0.0004; 5% errors cost under 0.0004. The anchoring, trend and development assumptions could all be off by several per cent and the score would barely move. The pooled metric with a constant baseline gives 0.082 of fold B's 0.1415 to line levels alone (a per-line constant scores 0.0822), but every team gets most of that for free; the contest is in the remaining 0.06, the within-line ranking.

### Where the ranking gap is: motor

Fold B, champion:

| Line | Share of baseline deviance | Within-line D² | A/E |
|---|---|---|---|
| Motor | 0.46 | 0.0849 | 1.007 |
| Buildings | 0.33 | 0.1137 | 1.038 |
| Contents | 0.21 | 0.0945 | 0.997 |

Motor is the largest line and the weakest, and `d2_motor_B` has been 0.0847 to 0.0849 in every ledger row since the seeded re-baseline on 10-18 14:02. Every loop keep after the bodily-injury split (09:07) and the credibility encoder (09:52) went to property: EBM blend, CatBoost near misses, field drops. Motor's LightGBM settings (15 leaves, min_data_in_leaf 2000, 1,200 rounds at 0.03) were chosen on 10-18 02:07 when the pooled motor target still carried the 11.9m claim; the BI model is now a Poisson count model and the BI target is capped at 1m, so the reason for that regularisation no longer applies to motor, and it was never revisited there (2.16 loosened property only; 2.6 only scaled min_data down for the smaller lines). The EBM on motor was tried once (+0.0005) with `interactions: 0`. If motor's within-line D² reached contents' 0.095, the pooled score would rise about 0.005, which is the whole gap to the leader.

### How much of the field's lead is noise

All teams are scored on the same 207,000 rows, so between-team spread is a paired difference. On fold B, the paired standard deviation of the D² difference on 33%-size subsamples is 0.0003 between two near-identical models (champion vs the CatBoost variant, full-fold difference 0.0006), 0.0016 between the champion and the 10-18 14:02 champion (difference 0.0060), and 0.0032 against the day-one baseline (difference 0.020). Pipelines built by different teams differ more than our own successive versions, so a paired sd of 0.002 to 0.003 between teams is the right order. The thirteen teams between 0.1250 and 0.1300 sit within about two sd of one another, and teams with 20 to 27 entries have had more draws at the public sample. Part of the 0.005 is sample and selection; the private 67% will reorder the top ten. Not all of it is noise: three to four thousandths of fold-B ranking power is a fair estimate of what is real, and the evidence above puts it in motor.

### Things checked and ruled out

- Missing customer fields (10.0% of motor rows in train and test, the customer_id is absent from `customers`). Their loss cost is 598 per exposure-year against 593 for the rest; the champion's A/E on them is 0.84 in fold A and 0.97 in fold B, so there is no persistent misfit. LightGBM takes the NaN natively.
- Written premium. Removing it cost 0.0004 on motor and nothing on property in the ledger (09:32 to 10:28). It is not the field's edge.
- Row structure. Rows ending early and mid-term-change rows exist in the test file at the same exposure shares as in 2022-24 (8.7% and 2 to 4%), and they run at 0.86 to 0.99 of full-year loss cost. No train-test mismatch there. (Note for anyone repeating this: train 2024 rows end on 30 December, not 31 December.)
- Segment calibration. Of the segments where the champion's fold-B A/E is off by more than 5%, most flip sign in fold A (driver age over 80: 0.90 / 1.13; buildings new business 0.88 / 1.14; contents broker 1.01 / 1.16). Two persist: motor new business is under-predicted in both folds (A/E 1.10 / 1.14, 9% of exposure; new business runs 1.3 to 1.5 times renewals in every training year) and rows with exposure under 0.5 are under-predicted on motor and contents (1.06 / 1.32 and 1.24 / 1.15) and over-predicted on buildings (0.78 / 0.89). Fixing every such segment to its own level is worth about 0.001 pooled, a real but small amount, because D² at power 1.5 is insensitive to calibration.
- 2025 weather in the level. Buildings' fold-B A/E of 1.038 costs 0.0002.

## 2. What went wrong in the loop

Ledger: 101 rows over 41 hours. 68 discards, 15 crashes, 7 loop keeps, 8 owner keeps, 2 over budget, 1 blend. All seven loop keeps are from 10-18 (the last at 22:24); day two has 32 scored loop runs and zero keeps. Day one's loop took the primary from 0.1056 to 0.1389; day two's owner changes took it to 0.1405.

### The keep rule threw away the gains the loop found

Twenty-one scored discards beat the bar of their day by more than 0.0005. With the measured seed noise at 0.0002 on the primary, the 0.002 floor in force until this afternoon was ten standard deviations, on top of a fresh-seed confirmation run that already guards against lucky draws. Concretely:

- The property field-drop direction was found by the loop at 15:34 on 10-18 (+0.0014) and again at 23:29 (+0.0009). Both discarded. The owner did it by hand eighteen hours later and got +0.0018.
- CatBoost as a third property member: +0.0014 at 23:54, discarded. Early-stopped rounds: +0.0012 at 03:54 and 04:11, discarded.
- Between 06:08 and 07:13 the agent did the right thing on its own and stacked the near misses: 0.1404, 0.1407, 0.1406, 0.1406 against a bar of 0.1389. All four discarded. The owner keeps that followed re-based the bar above them, so that work is gone unless someone re-applies the patches.
- Tonight, under the new 0.001 floor, CatBoost on top of the champion scored 0.14115 against 0.14048 (+0.0007; the champion loses on 96% of paired public-size subsamples) and was discarded again.

The owner's judgement changes did move the score more than the loop on day two, but the loop had found a similar amount and was not allowed to keep it. A floor of two to three seed sd (0.0004 to 0.0006) with the confirmation run kept is enough protection; about one null change in six was clearing the old 0.002 floor on the unseeded model, and the seeded model with confirmation is a different regime.

### Throughput halved and was spent on repeats and crashes

- Mean scored run went from 469 s on day one to 927 s on day two after the EBM keep; 48 scored runs on day one, 38 on day two. Every motor-only idea paid for an EBM refit on two property lines it did not touch.
- 15 crashes: 8 were `NameError` on a re-applied diff whose import was dropped (`xgb` three times, `CatBoostRegressor` twice, `KFold` twice, `IsotonicRegression`), 3 were the glum track (two silent segfaults and the joblib PATH error), 2 were CatBoost (the NaN category and a silent retry), 1 a `GroupKFold` signature, 1 a `KeyError`. The lint step added this afternoon fixes the first class; the glum ban fixes the second.
- Repeats: item 2.13 appears in 8 ledger rows, 2.14 in 6, 2.17 in 5, 2.15 and 2.7 in 4, 2.16, 2.20, 2.3, 2.34 and 2.45 in 3. Some repeats were legitimate re-tests under a changed bar; most were crash retries or the same idea against a stale bar.
- Two 1,889 s EBM runs were marked over budget against an 1,800 s limit for a 5% overrun, costing an hour before a single-seed EBM was kept.
- Five ledger rows read "(no HYPOTHESIS line)"; one of them (12:35, +0.0016) is now a backlog item whose idea had to be recovered from the patch.
- The loop never emitted `HYPOTHESIS: NONE`; it kept finding "next untried" items, including governance measurements it expected to discard (2.26, 2.27, 2.31 were run as "expected discard, for the report").

### The agent never saw a diagnostic

`program.md` step 1 tells the agent to read `results.tsv`, the backlog and `experiment.py`, and nothing else. Across 91 hypothesis lines the agent cites the backlog 99 times, `results.tsv` 41 times, `feature_study.json` 5 times, `model_card.json` never, "decile" 3 times, and in 11 lines the justification is "untried" or "next in Tier 2 order". The A/E numbers it quotes (the 0.88 to 1.18 decile spread, buildings 1.049) are the ones written into backlog items by the owner's reviews, not something it computed. The one row-level signal it could see, the per-line fold-B D² in the ledger, it used well once: the 07:42 run noticed `d2_motor_B` had been flat for eleven experiments and tried EBM on motor.

So the loop was a backlog executor. The backlog was written from research on 10-18 morning and from three owner reviews on 10-19; the data-led items (2.16 to 2.34, Tier 0b) were all generated by humans or review agents outside the loop. The loop generated its own hypothesis in about six iterations out of 91: the encoder-leak fix (10:15), the BI frequency misspecification fix (10:45), the four stacking runs, and the EBM-on-motor run.

### Verdict

The loop design is sound where it protects the owner (frozen target and scorer, driver runs the harness, patches saved, confirmation run). It failed on three settings and one omission: a keep floor set from a noise figure that was obsolete within hours, a time budget and model structure that made each motor experiment pay for property, an agent that re-implements from prose instead of applying its own patches, and no diagnostics in the agent's view. Day two produced 32 scored loop runs and about 10 compute hours for nothing kept, while the same runs contained roughly +0.002 of real gain.

## 3. Grounded in this data, or imported?

By scored loop runs (86): about 40 came from research, papers or generic practice (training power K1/K3, regularisation K9, isotonic R5, CANN R3, seed bag R6, GLM blend K11, null importance, Bühlmann-Straub encoder, the CatBoost/XGBoost/EBM bake-offs, path_smooth, early stopping, completion weights, recent-year anchoring, peril mechanics J5, the motor peril split, outer bags). About 33 came from an observation about this data or this model (the BI split, weather float, premium normalisation, lines_held, loosening property after the TabM comparison, the permutation-importance field drops, log exposure, duplicate columns, weather anomaly, tempering, the premium benchmarks, premium over sum insured, the history split, the interaction families, the near-miss stacks, the two bug fixes). Roughly 55/45.

By what was kept, the picture is different. The five loop keeps that moved the score were training power (generic), regularisation (generic), the BI split (this data), the credibility encoder (generic, prompted by this data's cardinality) and the EBM blend (library bake-off). The owner's +0.0018 came from pruning predictors measured against the model card, which is the most data-led step of the two days, and it was never available to the loop. Of the data-led loop items, the ones that produced near misses (field drops, history split) came from reviews run outside the loop; the loop itself computed nothing about residuals.

The experiments were grounded in this data at one remove. Humans and review agents looked at diagnostics and wrote backlog rows; the loop agent executed them in rank order with a one-line rationale copied from the row. It did not look at where its own model was wrong.

## 4. Changes, in priority order

1. Keep floor 0.0005 on the primary, worse-fold tolerance 0.002, confirmation run kept. Then re-apply the saved near-miss patches on today's champion in one combined run (CatBoost member, early-stopped rounds, same-line history), since each scored +0.0007 to +0.0016 alone, which clears the proposed floor. Expected: +0.001 to +0.002 fold B, which is the overnight loop's lost work.
2. Point the loop at motor. Cache the property predictions per fold once (both members, all seeds) and let a motor-only experiment reuse them; run time falls from 900 s to about 300 s. First motor items, all cheap and all justified by the ledger: re-tune the BI Poisson and the other-perils Tweedie now that the heavy-tail reason for 15 leaves and min_data 2000 is gone (31 to 63 leaves, min_data 200 to 500, early-stopped rounds); EBM on motor with `interactions` 10 to 20 instead of 0; a frequency times severity split for the other perils; CatBoost as a motor member. Motor is 46% of the deviance and has not moved in 30 hours.
3. Feed the agent diagnostics every iteration. Add a driver step after each scored run that writes `ai_log/diag_<run_id>.md` for the champion and the candidate: per-line decile A/E, A/E by the 15 largest features' bins, flagged as persistent only when both folds agree in direction, double lift against the champion, and the per-line D² change. The scripts in `~/.cache/iaa-owner/1019/meta/` (`segments.py`, `drift.py`) are most of it. Put that file in step 1 of `program.md` and require the HYPOTHESIS line to name the diagnostic it answers. The two persistent misfits to name now: motor new business (A/E 1.10 / 1.14) and short rows on motor and contents.
4. Subtraction first. The two largest gains in 48 hours came from removing things (pruning +0.0018, regularisation +0.0073). Add a standing rule to `program.md`: before adding a model, feature or stage, test removing the weakest one in the model card; and every kept addition must beat its removal on the next iteration.
5. Apply patches, do not re-implement. The driver should hand the agent `git apply` of the saved diff for any Tier 0 item, and the agent edits from there. Eight crashes and about three hours went to missing imports.
6. Budget as a soft cap. Log over-budget runs with their score and let the owner decide; two 0.139 runs were discarded for 89 seconds.
7. Ledger hygiene. Make the driver reject an agent reply without a `HYPOTHESIS:` line and ask once for it, the same way it does for lint. Mark governance measurements (expected discards) with a `measure` verdict so they do not count as failed experiments or consume Tier 0 slots.
8. Stop the backlog-ordering rule. Replace "highest-ranked untried item" with "the item whose expected gain, in the diagnostic file's units, is largest, or a follow-up to the last two results". Let the agent say `HYPOTHESIS: NONE` when nothing in the diagnostics points anywhere, and treat that as a signal to the owner rather than a failure.
9. Final-selection rule before Thursday (backlog 3.32). With a paired public sd of 0.002 to 0.003 and a private board three times the size, pick the final file by fold B and the confirmation run, not by public rank. Do not spend submissions chasing the sample.
