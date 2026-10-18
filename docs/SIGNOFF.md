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
| pending | a7aa9b7 (seeded LightGBM + TabM, `scripts/blend_submission.py`) | blend_2eb383c+dirty_20261018_1407_w0.5_20261018_1516.csv | 0.1392 (0.1395 / 0.1391); seeded LightGBM alone 0.1330 (0.1280 / 0.1355) | | TabM neural net blended 50/50 with the best LightGBM, re-anchored per line to LightGBM totals (line levels unchanged: buildings 500.1, contents 127.0, motor 638.6 per exposure year). Gain on both folds, largest on property. Weight fixed by owner, not tuned per fold | **awaiting Karl** |
