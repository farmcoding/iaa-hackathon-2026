# Submission sign-off log (human gate)

No submission is made without an entry here. The reviewer reads the diff since the last
submission, the fold scorecard, actual-to-expected by line and the guardrail tests, and
records the reason for submitting. Rules allow 5 submissions a day; one final
submission is selected by the human.

| Date (UTC) | Commit | File | CV primary (A / B) | Public LB | Reason | Approved by |
|---|---|---|---|---|---|---|
| 2026-10-18 05:18 | ec0919f+wip | calib_pooled_rate.csv | n/a | 0.03051 | Calibrate the local scorer: decides constant vs rate baseline | Karl (in chat) |
| 2026-10-18 05:45 | afcb387 | afcb387_20261018_0141.csv | 0.1056 (0.0900 / 0.1212) | 0.10597 | First real baseline; checks that CV tracks the leaderboard | Claude, under the plan's CV-tracking check; Karl to confirm retroactively |
