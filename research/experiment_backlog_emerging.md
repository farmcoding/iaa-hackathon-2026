# Experiment backlog: Part 2 (emerging market)

Pre-registered 25 Oct midday (owner), after two GPT-6.1 Sol reviews, a research report
(market transfer best practice) and an advisor check. Bar: P0, the sum of the per-peril
developed bases (owner keep under the deploy metric, primary 0.11554; board 0.15195).
Rules in program_emerging.md: one item and one setting per run, at most the settings listed,
no stacking, every factor gated with `lr_gate` inside the fit. The agent works down tier L
in order; OWNER items are human-only.

Gate evidence (owner check, claim counts against P0, Bonferroni over 30 tests): passes in
both years: motor gender, cover_type, occupation; contents tenure. One year only: motor
driver_age_band (2024), new_business and region (2023). Placebos (vehicle_colour,
email_domain, payment_method, channel) fail in both years.

## Tier L (agent, in order)

- **L1 Motor gender.** `offset_glm` on P0 (offset column `p0`) with cols ["gender"], gated
  (n_tests=1). Settings: alpha 0.001, 0.01, 0.1. Other lines unchanged.
- **L2 Motor gender x age band.** Mains gender and driver_age_band plus the interaction
  (gender, driver_age_band), gated as one block (n_tests=1). Alpha: the best of L1 (or 0.01
  if L1 was discarded at every setting), and 10x it.
- **L3 Motor joint GLM.** Candidate list (n_tests=8): gender, driver_age_band,
  new_business, region, cover_type, occupation, vehicle_value_group, ncb_years. Each gated
  in the fit; passing ones enter one ridge GLM on P0. Alpha 0.001, 0.01, 0.1.
- **L4 Property joint GLMs.** Buildings list (n_tests=6): gender, policyholder_age_band,
  new_business, region, construction_type, energy_rating. Contents list (n_tests=6):
  gender, policyholder_age_band, new_business, region, tenure, works_from_home. Gated; one
  ridge GLM per line on P0. Alpha 0.01, 0.1, 1.
- **L5 New business x age band (motor).** Interaction (new_business, driver_age_band) with
  mains, gated as one block; alpha 0.01 and 0.1. The 2025 book is 76% new business and
  younger: does new business behave differently by age here?
- **L6 Per-peril emerging levels.** `peril_levels(tr, line, k)` on every line with P0;
  k 100 and 300.
- **L7 Hail elasticity on hail perils.** The per-peril motor bases were fitted without
  hail_days (Part 1 pruning), so this adds an effect the base lacks: multiply
  base_glass_hail (motor) and base_storm (buildings) by (hail_days / training-row mean
  hail_days) ** e, e in {0.3, 0.6}.
- **L8 Total vs per-peril base.** Geometric blend sqrt(P0 x base); one setting.
- **L9 Base families** (only once the owner has added `base_compat*` or `base_pooled*`
  columns to the config): fixed blends of P0 with each family, weights 0.25 / 0.5 / 0.75
  on the family; and the family alone.

## OWNER

- O1 Schema-compatible developed bases (running): total and per-peril, compared with P0
  under the frozen metric; re-base if they win by the margin.
- O2 Pooled two-market bases (running): emerging weight 1 / 5 / 20, Part 1 pruning off.
- O3 Level method: balance-on-train kept (actuarial balance property, board-checked);
  Tweedie-MLE scalar scored 0.11538 vs 0.11554, inside the margin (25 Oct).
- O4 Combining kept tier-L mechanisms: once, by the owner, then confirmed on the board.
- O5 Target sensitivity: bodily-injury tail, recoveries on open own-damage claims (~1.5% of
  2024 motor), completion as offset.

## Parked (evidence against, 25 Oct)

Free boosted corrections on the base; one-way credibility factor stacks; emerging-only
refit; region/age credibility factors on cost.
