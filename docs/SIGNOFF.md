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
