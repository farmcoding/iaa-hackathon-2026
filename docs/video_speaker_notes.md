# Video speaker notes (5 minutes)

Final script, 30 October 2026. About 700 words at roughly 145 words a minute, so it runs
just under 5 minutes; only the first 5 minutes are graded. One section per slide of the
video deck (claude.ai artifact "IAA Hackathon 2026 video deck"). Part 2 gets about 80
seconds because adaptation carries 40% of the marks.

---

## Slide 1 - The task and the result (0:00 to 0:25)

I'm Karl, a P&C actuary and a Fellow of the CIA, entering on my own. The task was to price
the 2025 risk premium for motor, buildings and contents. First in a developed market, then
in a new one with two years of data. Part 1 scored 0.134 on the private board. Part 2 scores
0.166 on the public board. Most of this is about how I got there. The AI did a lot of the
work, and an actuary signed off every number.

## Slide 2 - Two data traps (0:25 to 1:00)

Before any modelling the AI profiled every table, and I checked what it found. Two things
would have broken the model. Customer-service contacts included claim calls in training but
not in the test file. It looked like the best predictor in the data, and it was leakage, so
it's out. And about 0.4% of exposures were in days, not years. We recompute them from the
dates. Every finding is in the data quality log.

## Slide 3 - A target developed to ultimate (1:00 to 1:35)

The score is on ultimate cost, so the target is ultimate too. Reported claims are developed
with a chain ladder on report delay, by line and peril group. Bodily injury case reserves
were too light, so it's projected as ultimate count times trended severity. The check:
developed 2024 has to land on the 2021 to 2023 trend. Raw motor was 12% short. Developed,
it's within 2%.

## Slide 4 - Part 1 model and validation (1:35 to 2:05)

Each line is a blend of LightGBM, CatBoost and an explainable boosting machine, with bodily
injury as its own frequency model. Validation is out of time: train on earlier years, score
the next one, on two folds. A change stays only if it wins on both. The final scored 0.126
on the public board and 0.134 on the private board.

## Slide 5 - The AI experiment loop (2:05 to 3:00)

AI was in every step. It profiled the data, researched methods and wrote most of the code.
The core is the experiment loop. An agent changes one file and states a hypothesis. A frozen
harness scores it, and the result goes in a ledger. The agent can't touch the data, the
target, the folds or the scorer. 248 experiments ran across both parts and 33 were kept. The
chart shows Part 1 climbing from 0.106 to 0.143. Models from other providers challenged the
work, and I checked their claims against the data. One built a recommendation on 2023 being
all new business. The data says 6%, so it was rejected.

## Slide 6 - Guardrails and professional standards (3:00 to 3:25)

I ran this the way I'd run client work under the CIA Standards of Practice, with the IAA AI
Task Force papers for structure. In practice that means 41 automated tests, a keep rule
declared before anything was scored, a human sign-off before every submission, and a final
file that rebuilds from the repo with a hash check.

## Slide 7 - Part 2: what carried over (3:25 to 3:55)

The new market's claims are 20 to 40% cheaper, and motor claims are more frequent. The 2025
book is 77% new business. Refitting the Part 1 model on emerging data alone scored 0.129.
Fitting it on the developed market and applying it to emerging rows scored 0.142. So the
developed relativities carried over better than two small years could estimate them, and
every Part 2 model starts there.

## Slide 8 - Part 2: what was kept and dropped (3:55 to 4:45)

From there the agent worked through a pre-registered list, one mechanism per run. Three
steps were kept. Each peril gets its own emerging level, motor adds gender, and O4 adds
emerging levels by peril and a new-business factor. About 20 other mechanisms were dropped
because they hurt a fold. Human review mattered most once. The agent stacked eleven small
factors that each just missed the margin. The folds went up and the board went down, so I
reverted them and tightened the rule. Last, the 2025 level. The folds can't see a book
that's mostly new business, so I read the level off the board, shrank it, checked it on the
folds, and used our own trends where they explained it. That took us to 0.166.

## Slide 9 - Limits and close (4:45 to 5:00)

The biggest uncertainties are the 2025 level, which no fold can test, and buildings, which
rests on few fire claims. Both are in the report with sensitivities. The code and full
history are in the repo. Thanks.
