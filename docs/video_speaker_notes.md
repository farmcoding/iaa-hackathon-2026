# Video speaker notes (5 minutes)

Final script, 30 October 2026, in Karl's voice and humanized: first person throughout, warmer
intro. About 750 words; the narrated video runs under 5 minutes, and only the first 5 minutes
are graded. One section per slide of the video deck (claude.ai artifact "IAA Hackathon 2026
video deck"). Part 2 gets about 80 seconds because adaptation carries 40% of the marks.

---

## Slide 1 - The task and the result (0:00 to 0:35)

Hi, I'm Karl. I'm a property and casualty actuary, and this is my solo entry. Thank you for
the opportunity to present my results. The task was to price the 2025 risk premium for motor,
buildings and contents. First in a developed market, and then in a new market with only two
years of data. The way I approached it was to use AI as much as I could, through the whole
process, but still land on a result I'd be comfortable signing off as an actuary. Part 1
scored 0.134 on the private board. Part 2 is at 0.166 on the public board.

## Slide 2 - Two data traps (0:35 to 1:05)

Before any modelling, the AI profiled every table, and I went through everything it found.
Two issues would have broken the model. Customer-service contacts included claim calls in
training, but not in the test file. It looked like the best predictor in the data, but it was
leakage, so I took it out. About 0.4 per cent of exposures were recorded in days instead of
years, so I recomputed them from the policy dates. Every finding is in the data quality log.

## Slide 3 - A target developed to ultimate (1:05 to 1:35)

The score is on ultimate cost, so I built the target to ultimate as well. Reported claims are
developed with a chain ladder on report delay, by line and peril group. Bodily injury case
reserves were too light, so I projected bodily injury as ultimate count times trended
severity. My check was whether developed 2024 lands on the 2021 to 2023 trend. Raw motor was
12 per cent short. Once developed, it's within 2 per cent.

## Slide 4 - Part 1 model and validation (1:35 to 2:05)

For Part 1, each line is a blend of LightGBM, CatBoost and an explainable boosting machine,
with bodily injury as its own frequency model. I validated out of time. Train on the earlier
years, score the next one, on two folds, and a change only stays if it wins on both. The
final model scored 0.126 on the public board and 0.134 on the private board.

## Slide 5 - The AI experiment loop (2:05 to 3:00)

AI was part of every step. It profiled the data, researched methods and wrote most of the
code. Model changes went through an experiment loop. An agent changes one file and states its
hypothesis, a frozen harness scores it, and the result goes into a ledger. The agent can't
touch the data, the target, the folds or the scorer. Across both parts, 248 experiments ran
and 33 were kept. On the chart you can see Part 1 climbing from 0.106 to 0.143. I also had
models from other providers challenge the work, and I checked every claim they made against
the data. One built a recommendation on 2023 being all new business. The data says 6 per
cent, so I rejected it.

## Slide 6 - Guardrails and professional standards (3:00 to 3:25)

I ran this the way I'd run client work, under the CIA Standards of Practice, and I used the
AI Task Force papers from the IAA for structure. In practice that means 41 automated tests, a
keep rule I set before anything was scored, my own sign-off before every submission, and a
final file that rebuilds from the repo with a hash check.

## Slide 7 - Part 2: what carried over (3:25 to 3:55)

Part 2 was a different problem. The new market's claims are 20 to 40 per cent cheaper, motor
claims are more frequent, and the 2025 book is 77 per cent new business. When I refit the
Part 1 model on the emerging data alone, it scored 0.129. Fitting it on the developed market
and applying it to the emerging rows scored 0.142. So the developed relativities carried over
better than two small years could estimate them, and I started every Part 2 model from that
base.

## Slide 8 - Part 2: what was kept and dropped (3:55 to 4:45)

From there, the agent worked through a pre-registered list, one mechanism per run. Three
steps were kept, and the slide lists them. About 20 other mechanisms were dropped
because they hurt a fold. My review mattered most when the agent stacked eleven small factors
that each just missed the margin. The folds went up and the board went down, so I reverted
them and tightened the rule. The last piece was the 2025 level. The folds can't see a book
that's mostly new business, so I read the level off the board, shrank it, checked it on the
folds, and used the trends from the emerging data where they explained it. That's what took
Part 2 to 0.166.

## Slide 9 - Limits and close (4:45 to 5:00)

The two biggest uncertainties are the 2025 level, which no fold can test, and buildings,
which rests on very few fire claims. Both are in the report, with sensitivities. The code and
the full history are in the repo. Thank you for watching.
