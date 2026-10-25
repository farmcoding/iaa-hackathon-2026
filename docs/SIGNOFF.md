# Submission sign-off log (human gate)

No submission is made without an entry here. The reviewer reads the diff since the last
submission, the fold scorecard, actual-to-expected by line and the guardrail tests, and
records the reason for submitting. Rules allow 5 submissions a day; one final
submission is selected by the human.

| Date (UTC) | Commit | File | CV primary (A / B) | Public LB | Reason | Approved by |
|---|---|---|---|---|---|---|
| 2026-10-18 05:18 | ec0919f+wip | calib_pooled_rate.csv | n/a | 0.03051 | Calibrate the local scorer: decides constant vs rate baseline | Karl (in chat) |
| 2026-10-18 05:45 | afcb387 | afcb387_20261018_0141.csv | 0.1056 (0.0900 / 0.1212) | 0.10597 | First real baseline; checks that CV tracks the leaderboard | Karl (confirmed 2026-10-18: part of the initial build plan) |
| 2026-10-18 12:23 | 1b2bbf0 (experiment) | 4ff7730_20261018_0823.csv | 0.1201 (0.1138 / 0.1265) | 0.11196 | Best loop model after batch 1 (training power 1.1, stronger regularisation). Public gain +0.006 vs CV gain +0.015: CV overstates gains, track the ratio | Karl (in chat) |
| 2026-10-18 14:56 | ce894e7 (experiment) | ce894e7_20261018_1056.csv | 0.1339 (0.1292 / 0.1362) | 0.11913 | Best loop model after relaunched batch 1: credibility encoding plus motor BI frequency split. Fold B gain +0.0097 vs public +0.0072 | Karl (in chat) |
| 2026-10-18 19:37 | cadd31e (seeded LightGBM, harness --final from a clean commit; TabM `scripts/tabm_fit.py`; `scripts/blend_submission.py`) | blend_cadd31e_20261018_1523_w0.5_20261018_1523.csv | 0.1392 (0.1395 / 0.1391); seeded LightGBM alone 0.1330 (0.1280 / 0.1355) | 0.12299 | TabM neural net blended 50/50 with the best LightGBM, re-anchored per line to LightGBM totals (line levels unchanged: buildings 500.1, contents 127.0, motor 638.6 per exposure year). Gain on both folds, largest on property. Weight fixed by owner, not tuned per fold. Artefact hashes below | Karl (in chat, 2026-10-18 15:3x EDT: "I approve you to submit the best result so far") |
| 2026-10-19 12:2x | 05cd391 (experiment; final fit run from an exact copy of HEAD `src/iaa/experiment.py` while the loop held the working tree, hence the `+dirty` file name) | c2516b3+dirty_20261019_0605.csv | 0.1389 (0.1370 / 0.1399); fresh-seed confirmation 0.1385 | 0.12332 | LightGBM blended 50/50 with an EBM (Explainable Boosting Machine) on buildings and contents; motor unchanged. Single model, no TabM: TabM adds +0.0001 to +0.0015 on top of it depending on seed (inside noise). Line levels unchanged: buildings 500.1, contents 127.0, motor 638.6 per exposure year. sha256 file `afd2ad42c8bd373974b17263dd2aae0d793343511a1c0d36335993815c9453f1`, experiment.py `c815be9c631c1c7fb71fb4c4d563a12fb3f613461339f938a16d00408866be43` | Karl (in chat, 2026-10-19: "1. Yes") |
| 2026-10-19 22:03 | efadffd (experiment; final fit from an exact copy while the loop held the tree) | 882e421+dirty_20261019_1803.csv | 0.1405 (0.1384 / 0.1415) | 0.12476 | No premium on any line; email domain, licence points (property) and five sales/admin fields removed; predictors adding <=0.0001 pruned (about half); BI capped at 1m with Pareto load; buildings water trend 9%. Line levels per exposure year: buildings 506.9, contents 127.0, motor 628.7 (was 500.1 / 127.0 / 638.6). sha256 file `fcf421d5f456d9d3fde9478564f6d29f0e2246f6b3ed16bc0d54707d6bcc9623`, experiment.py `a410ccb9d00ab4c6961ab0ea4fbea7150156bd7d06e774c4ef0756065926eeb7` | Karl (in chat, 2026-10-19: "Submit best model") |
| 2026-10-20 11:03 | bf8aeb7 (experiment; final fit from an exact copy while the loop held the tree) | 82906d6+dirty_20261020_0656.csv | 0.1423 (0.1419 / 0.1424) | 0.12602 | Adds to the 19 Oct model: EBM blended into motor other perils, CatBoost as a third property member, early-stopped rounds on property (all loop keeps, confirmed on fresh seeds). Line levels per exposure year unchanged: buildings 506.9, contents 127.0, motor 628.7. sha256 file `6b4d17a8d2265241fa772a76b6a2faf85c0ef3127bf375c9a1e5e09012471ead`, experiment.py `66bf80cac239bda1f171a4da1babb404d4ef50205378134ea4be0c9377169cc7` | Karl (in chat, 2026-10-20: "1 yes") |
| 2026-10-21 12:02 | 0b1d113 (experiment; final fit from clean HEAD 710775e) | 710775e_20261021_0733.csv | 0.14252 (0.14234 / 0.14261) | 0.12622 | Adds to the 20 Oct model: BI floor removed from the model level, buildings per-peril trends (weather and water 9%, fire and accidental damage 4%), customer-level claim history dropped from buildings, construction_type coding fix (T/t merged into W/w), monotone credit band on motor (option 2b). All owner keeps on the frozen harness. Line levels per exposure year: buildings 515.1, contents 127.0, motor 626.9 (was 506.9 / 127.0 / 628.7). Contents predictions identical to the 20 Oct file. sha256 file `3f9205c0cedceb2b6089738d844441fbf669ace2f5f511a78b84be8026fdd57c`, experiment.py `d5d72c8cfc206c34e1990a42430d51bedc3e438a688559579549b5754e46f407` | Karl (in chat, 2026-10-21: "yes submit it and sign-off as approved") |

**Final selection rule (pre-declared 2026-10-20, Karl):** the one final submission selected on Kaggle is the signed-off candidate with the best fold B D²; ties go to the higher primary, then the higher worse fold. Public leaderboard rank is not a criterion.

**Amendment for two target tracks (pre-declared 2026-10-20, Karl, before any track-2 result existed):** fold B D² is not comparable across tracks, because each track scores against its own developed actuals. The track is chosen first, in this order: (1) the development method the as-if reserving replay (backlog 3.9-lite) favours on late-reported claims; (2) confirmed by the score against reported cost at the valuation date (backlog 3.6), which is the same yardstick for both tracks; (3) one leaderboard submission of track 2's best model, as the only ground truth both tracks share. Within the chosen track, the rule above applies unchanged.

### Artefact hashes (sha256)

TabM training on Metal is not bit-for-bit reproducible, so these saved files are the artefacts of record. Dataset fingerprint `6250034b1783`.

**Blend submission, 2026-10-18**

- TabM A: `tabm_A_6250034b1783.parquet` `0b0367cc71007a7524af8b390c293c170c4a3a00e83f3c2e156582cc6f574a0c`
- TabM B: `tabm_B_6250034b1783.parquet` `e58b042f81ae652dadc39903e7cec721177589c1245f83a29b33f6350434c7f9`
- TabM final: `tabm_final_6250034b1783.parquet` `5c01276bc0ec62a7a24ff8414f0a8bf75c90394ae920dd98bf4ce7b69f1c15e1`
- LightGBM final: `cadd31e_20261018_1523.csv` `f24ec9b85f7c153ad3070220703c253c2c63d2020941f8dd764e2f8c1abba218`
- blend submission: `blend_cadd31e_20261018_1523_w0.5_20261018_1523.csv` `d2387582e9129842ce03d90f7276f4bf38782a8576790b25c4e7e7d76c24775f`

Public gain +0.0039 (0.11913 to 0.12299) for a fold B gain of +0.0029 over the seeded LightGBM: validation keeps tracking the leaderboard.

**Re-tag, 2026-10-18 16:10 EDT.** The dataset fingerprint changed from `6250034b1783` to `62f521c8370f` when the fingerprint was narrowed to data-building code and the cache crash (P3) was fixed. The rebuilt dataset was compared with the old cache column by column (values and dtypes): identical, 3,035,950 rows x 138 columns. The TabM files were copied to the new tag; their bytes and the sha256 values above are unchanged, and the blend re-scores at 0.13922.

## Part 2: emerging market

Same gate. Ten submissions a day; one final submission is selected by the human. Baseline rows check the file plumbing and set the floor; they are not candidates for the final.

| Date (UTC) | Commit | File (sha256, first 16) | CV fold B (train 2023, score 2024) | Public LB | Reason | Approved by |
|---|---|---|---|---|---|---|
| 2026-10-25 04:55 | 68e71bf + market switch (uncommitted), `scripts/part2_baseline.py` 76e9df44 | baseline_rate.csv (a37d37d28c69133d) | n/a | 0.08396 | Baseline, not a candidate: one trended loss cost per exposure per line; checks the 209,063-row file and the scorer | Karl (asked for a dummy baseline submission, 2026-10-25) |
| 2026-10-25 05:10 | 68e71bf + market switch (uncommitted), experiment.py d5d72c8c (Part 1 final, unchanged) | native_v0_20261025_0105.csv (d5f91231ce710938) | 0.1149 (buildings A/E 0.60) | 0.12904 | Baseline, not a candidate: the Part 1 specification refit on emerging 2023-24 with no changes; first bar for the adaptation | Karl (same request) |
| 2026-10-25 05:55 | 68e71bf + market switch (uncommitted), `scripts/part2_transfer.py` 96bd7957 | transfer_p1.csv (dc6f0b5026a0b53d) | level-neutral 0.151 on emerging 2024 (native 0.123) | 0.14243 | Baseline, not a candidate: the Part 1 final specification fitted on developed 2021-24, emerging line levels as the other baselines; tests whether developed relativities transfer | Karl (same request) |
| 2026-10-25 05:58 | aa065d1 (pre-commit state), diagnostic | transfer_p1_mixlevel.csv | n/a | 0.15079 | Diagnostic, not a candidate: transfer_p1 with each line scaled by the 2025 mix index from the developed relativities (motor 1.16, buildings 1.087, contents 1.018). Tested whether anchoring to the historical loss cost understates a riskier book: +0.0084. The method was then frozen in the harness (balance on train), not fitted to the board | Karl (asked to find what was missing, 2026-10-25) |

### Part 1 late submissions (diagnostic, after the close; they do not change the Part 1 result)

Part 1 accepts late submissions after its close and scores them without ranking them. Eight were used once, on 25 October, to learn what the Part 1 model missed. None is a candidate; the Part 1 final stays 710775e (private 0.13363).

| File | Change to the Part 1 final | Public | Private |
|---|---|---|---|
| p1_all_x105 | all lines x1.05 | n/a | 0.13253 |
| p1_motor_x0.9 / x1.1 | motor level | n/a | 0.13437 / 0.13166 |
| p1_buildings_x0.9 / x1.1 | buildings level | n/a | 0.13329 / 0.13305 |
| p1_contents_x0.9 / x1.1 | contents level | n/a | 0.13335 / 0.13343 |
| p1_mixlevel | each line scaled by the model's own 2025 mix index (developed base fit: motor 0.906, buildings 0.991, contents 1.005) | 0.12639 | 0.13438 |

The fitted best levels from the probes (motor 0.90, buildings 0.98, contents 1.00) match the mix index the model gives without the board. Anchoring each line to the historical loss cost put Part 1 motor about 10% high; balancing on the training rows would have scored 0.13438 instead of 0.13363. The probes stopped there.
