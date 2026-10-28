# Video speaker notes (5 minutes)

Draft 1, 2026-10-18. About 720 words at roughly 145 words a minute. Numbers in
[brackets] get refreshed on Wednesday 21 October with the deck, and again after Part 2.
Only the first 5 minutes are graded, so the timings are hard limits.

---

## Slide 1 - The ask and the answer (0:00 to 0:30)

I'm Karl, a P&C actuary and a Fellow of the CIA, entering on my own.

The task was to price 2025 risk premium for motor, buildings and contents - first in a developed market, then in a new one with two years of data. My Part 1 model scored 0.134 on the private board, and the Part 2 model scores 0.166 on the public board. I'll spend most of the time on how I got there - because the AI did a lot of the work, and an actuary still signed off every number.

## Slide 2 - The data had traps (0:30 to 1:15)

Before any modelling I had the AI profile every table, and I checked what it found.

Four things mattered. About 0.4% of training rows had exposure in days, not years - we recompute it from the dates. One field, customer service contacts, included claim calls in training but not in the test file. It looked like the best predictor in the data and it was pure leakage, so it's out. Motor bodily injury in 2024 was only half reported. And 410 claims matched no policy period.

Each one is in the data quality log with what we did about it.

## Slide 3 - Getting the target right (1:15 to 2:00)

The score is on ultimate cost, so the target has to be ultimate too.

We develop reported claims with a chain ladder on report delay, by line and peril group. Bodily injury needed more. Case reserves on open claims were too light - same-delay claims got cheaper in younger years while everything else inflated. So bodily injury is projected as ultimate count times trended severity from the most mature year.

The test I set: developed 2024 has to land on the 2021 to 2023 trend. Raw motor was 12% short. After development it's within 2%. That test runs automatically.

## Slide 4 - The model and how we know it works (2:00 to 2:45)

The production model is a blend of LightGBM, an explainable boosting machine and CatBoost per line, with bodily injury as a frequency model. The model gives the relativities. The overall level comes from trended history, which is how we'd price in practice.

Validation is out of time - train on earlier years, score the next one, twice. Local score and leaderboard agreed almost exactly: 0.1056 locally, 0.1060 public. That's why I trusted my own validation and didn't chase the leaderboard.

## Slide 5 - How AI was used (2:45 to 3:45)

AI was in every step. It profiled the data, researched methods - Richman and Wüthrich's work, past Kaggle winners, UK bodily injury law - and wrote most of the code.

The core is an experiment loop in the style of Karpathy's autoresearch. An agent changes one file, runs it against a frozen harness, and the result goes in a ledger. It can't touch the data, the target, the folds or the scorer. A change only stays if it beats the best run by more than seed noise on both folds. About 250 experiments ran across both parts and 33 were kept - the chart shows the Part 1 score climbing from 0.106 to 0.143.

Nothing the AI produced went in unchecked. It found the leakage, and I confirmed it against the test file before anything changed. Every kept change has a commit, a test and a reason.

## Slide 6 - Guardrails and professional standards (3:45 to 4:20)

I ran this the way I'd run client work under the CIA Standards of Practice - data under 1440, models under 1450, documentation under 1490, reporting under 1700. The IAA AI Task Force papers on governance, testing and documentation gave the structure, and the CAS AI primer covers the agent risks.

In practice that means automated tests on the core, a human sign-off before every submission, a pinned and reproducible environment, and a full log of what the AI did and how it was checked.

## Slide 7 - Part 2, the new market (4:20 to 4:50)

The new market's claims are cheaper and more frequent, and the 2025 book is 77% new business. I kept the developed relativities, refitted them by peril, and let each peril find its own emerging level. Gender was new here and it mattered.

My folds couldn't see a book that's mostly new business, so the AI found an exact way to read the 2025 level off the leaderboard. I shrank it, checked it on the folds, and kept what our own trends explained.

## Slide 8 - Limits and close (4:50 to 5:00)

The biggest assumption is the bodily injury severity. It's flagged with a sensitivity range in the report. The code and full history are in the repo.

Thanks.
