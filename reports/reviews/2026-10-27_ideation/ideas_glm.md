# Advisory review — Part 2, 48 hours to close

## 1. Constraints, ranked by cost

**1. Modelling constraint: the base's *shape* is frozen and only *levels* are adapted (largest cost, ~0.004–0.010 board).**
The evidence says the developed shape is wrong in the emerging market: motor decile A/E is non-monotone (decile 1 at 0.44, deciles 4–5 at 1.35), and 8+ segments sit at A/E 1.1–1.7. The entire adaptation stack (per-peril levels, credibility factors, gender GLM, NB factor) is *multiplicative on cost* — it can move levels but cannot repair rank-ordering or fix compression of relativities. Worse, the tree bases clamp on emerging geo/weather values that sit outside the developed range (hail 14–17 vs ~6; elevation, wildfire, river distance) — trees cannot extrapolate, so the base is structurally blind exactly where the emerging market lives. Relax safely by adding (a) one parametric shape recalibration inside `fit_predict` (idea 1), (b) count-based rather than cost-based corrections (idea 2), and (c) 1–2 new base-family fits on Kaggle (ideas 4, 9) that don't touch the Mac.

**2. Validation: two correlated folds, one true OOT read, no year-2025-like weather (~0.002–0.005 cost).**
Fold R scores 2023 with a 2024-trained model — it shares the same two weather years and mostly-renewal mix with fold B, so the folds are not independent reads. Two failure modes are already visible: the stacked-factors run gained +0.0068 on folds and *lost* 0.0029 on the board (fold overfit accepted), while the compat-base blend scored 0.11792 vs P0's 0.11554 primary and was apparently discarded (a likely true effect rejected). And the 2025 weather shift (hail 11.4 vs 14–17) is *invisible* to both folds — the reweighted validation book reweights mix, not weather. Relax safely: (i) same-day paired LB probes (candidate vs current best, difference-of-differences kills the 0.002 LB noise); (ii) cross-fitting *within* years for the correction layer so each fold has an out-of-sample read; (iii) for causal-prior mechanisms (physical perils, geo), accept at a 0.001 fold margin if both folds are positive and the LB probe confirms. You have ~15–20 submissions left — reserve 6–8 for paired A/B probes of the top 3 ideas, not for finals.

**3. Governance loop: one mechanism per run, ≤3 settings, no stacking (~0.002–0.004 cost).**
Serial gating at a 0.002 margin guarantees that a bundle of true-but-small effects (previous_insurer, missed payments, elevation, region, licence points, occupation — each individually sub-margin, jointly the whole A/E list) can never be adopted, because they can only be tested one at a time or as an unregularised joint GLM (which "diluted gender"). The compliant relaxation is not to drop gating but to change the *unit*: pre-register one **hierarchical/shrunk layer** as a single mechanism — many factors, per-factor ridge penalties, partial pooling across lines (idea 2). One run, one gate, but the joint effect is testable. Keep the "no stacking of discarded runs" rule.

**4. Compute: not actually binding (~0 cost once Kaggle is used).**
The 7–11 GB developed frame fits comfortably in Kaggle's ~30 GB CPU RAM, and the competition data is already attached — no upload needed. Run the geo/weather GAM fits, the buildings-relativity transfer, and the pooled-v2 refit in Kaggle CPU (and TabPFN on GPU) while the Mac runs the frozen agent loop. Two parallel tracks ≈ doubles base-family throughput over the remaining 40 h. Only rule: never more than one heavy frame process per machine; the 25 Oct crash was three on the Mac.

## 2. White space — what has not been tried

- **Transparent, unused:** GAM/splines as an *extrapolation layer* for geo/weather (the one place trees fail); proper **Bayesian hierarchical / mixed-effects** segment models (the current credibility k's are ad hoc and hand-set); **frequency–severity decomposition** — every correction is fitted on skewed cost; claim *counts* are far less noisy and the A/E diagnostics are already count-based; **monotone/non-monotone recalibration of the base** (isotonic was tried in Part 1 on developed, never in-emerging).
- **Transfer/domain adaptation, essentially unused:** importance weighting of the *fit* toward the 2025 covariate distribution (the classifier odds are already computed for validation but never used to weight estimation); transfer of *physical relativities* estimated on 3M developed rows (energy rating, trees, flood zone) into the emerging correction; explicit handling of out-of-support covariates.
- **Black-box, unused:** neural tabular (TabPFN v2 is genuinely strong at this sample size and can run on the free T4 as a *third opinion*); multi-task across lines beyond the pooled tree base; distillation; real AutoML/bandit search (the LLM loop is close, but the one-mechanism rule prevents it from exploring jointly).
- **Tried and failed (don't revisit):** free boosted residual corrections, unregularised stacked factors, joint unshrunk GLMs, scalar hail elasticity, isotonic on developed, TabM/RealMLP.

## 3. Ideas, ranked

The rough currency: past runs suggest a primary gain of +0.002 maps to roughly +0.002–0.007 on the board (L1: +0.0036 primary → +0.0074 board; O4: +0.0031 → +0.0039). Fifth place needs +0.0034 board. The realistic path is ideas 1–3 stacked, each individually gated.

Full details in the JSON below; the logic in one line each:

1. **Shape recalibration** — the decile A/E pattern is a conditional-mean error; a cross-fitted Tweedie spline on log(base) per line directly buys deviance and transfers because it's fitted on emerging experience that 2025 resembles more than developed does.
2. **Hierarchical count layer** — the known-miss segments each fail the 0.002 gate alone but are collectively large; Poisson counts + ridge shrinkage + cross-line pooling is the governance-compliant way to harvest them.
3. **New-business split** — 76% of the score book is new business; 2023 emerging is 100% new business, so you have a clean signal to give new business its own shape and level.
4. **Geo/weather parametric offsets** — fixes tree clamping on out-of-range covariates and gives the 2025 milder-weather level a mechanism the folds cannot test but the LB probe can.
5. **Buildings physical relativity transfer** — physics transfers across markets even when frequencies don't; 160k developed claims beat 1,000 emerging ones.
6. **compat blend retest under O4** — the earlier numbers (+0.0024 primary over P0) suggest this was a true effect lost to sequential gating.
7. **previous_insurer credibility encoding** — the largest single-segment spread (0.47–1.69) and it's the defining feature of a broker-sourced new-business book.
8. **Pooled base v2, properly weighted** — the duplication disaster was a leaf-size artefact, not a refutation; weight column + shallow depth + weight-aware leaf constraints.
9. **Importance weighting of the correction fit** — the covariate-shift machinery already exists in the harness.
10. **prior_claims / hist audit** — cheap check; if the developed exclusion silently dropped emerging-usable fields, this is free lift.
11. **Young-driver licensing interaction** — 50% of 2025 exposure is 17–34; the age→mean mapping flattens exactly the region that matters.
12. **TabPFN third opinion** — diversity play, free GPU, low probability of a hit but cheap.
13. **Weather-year index for 2025** — level effect invisible to the folds; needs mechanism prior + LB probe.

## 4. The weakest spot

**The pipeline repairs levels on top of a frozen developed shape, but the evidence says the *shape* is the problem — and the validation design cannot see it.** Decile 1 at A/E 0.44 means the base overprices its lowest-risk emerging rows by 2.3×; deciles 4–5 at 1.35 means the middle of the book is 35% underpriced; no multiplicative level factor, credibility k, or gated one-way term can fix that. Simultaneously, the tree bases clamp on every geo/weather covariate outside the developed range (hail, wildfire, elevation, river distance), so the emerging market's defining exposures enter the prediction as constants. And the 2025 book — 76% new business, 50% young, broker-sourced, a milder weather year — is precisely the population where neither the developed shape nor the two training years ever looked. The folds share both weather years and a renewal-heavy mix, so they will neither reward shape repair nor punish weather-level errors; the LB is the only instrument that can. The single largest expected gain is therefore: fix the conditional mean of the base per line (idea 1), harvest the segment bundle via a shrunk count layer (idea 2), split new business out (idea 3), and use paired LB probes as the tie-breaker the folds cannot provide.

```json
[
  {
    "name": "Mono-free shape recalibration of the base, per line",
    "mechanism": "Cross-fitted Tweedie GAM g(log base) per line (4-6 knots, heavy penalty, fitted on train rows only) multiplied into predictions; harness balance then fixes the level. Directly targets the motor decile A/E pattern (decile 1 at 0.44, deciles 4-5 at 1.35): the conditional mean E[Y|base] is misshapen, which is pure deviance waste a level factor cannot recover.",
    "expected_gain": "+0.0015 to +0.004 primary; largest single idea",
    "harness_fit": "Inside fit_predict(train, valid): fit spline on train rows' actuals vs their fold-safe bases, apply to valid. Settings: knots, penalty, per-line on/off.",
    "compute": "Seconds to minutes; numpy/scipy only",
    "where": "Mac, agent loop",
    "hours": 0.5-1 per run, 3-5 runs for the settings grid",
    "overfit_risk": "Medium: 6 knots on ~35k rows can chase fold noise. Test: cross-fit within each fold so every scored row's knot coefficients exclude it; require both folds positive and level-neutral fold B not down; paired LB probe of the best setting vs O4.",
    "test": "Keep rule as-is; additionally check A/E by decile becomes monotone-adjacent on both folds"
  },
  {
    "name": "Hierarchical count-based correction layer (replaces stacked factors)",
    "mechanism": "Poisson GLM on row claim counts, offset log(P0), with the full known-miss bundle as factors (gender, previous_insurer, missed_payments_12m, licence_points, region, occupation group, new_business), per-factor ridge penalties, payment/occupation/region factors POOLED across the three lines (partial pooling). Counts are far less skewed than cost, so more factors survive; shrinkage fixes the 'joint GLM diluted gender' problem.",
    "expected_gain": "+0.002 to +0.004 primary; board +0.003 to +0.008 if it holds",
    "harness_fit": "Inside fit_predict: one pre-registered mechanism (one shrunk layer, many factors). Settings: ridge scale, pooling on/off, count offset source.",
    "compute": "Statsmodels/GLMNet-style IRLS on 65k rows, minutes",
    "where": "Mac, agent loop",
    "hours": 0.5 per run; 4-6 runs",
    "overfit_risk": "Medium-high (the unshrunk version gained +0.0068 folds / -0.0029 board). Test: penalty chosen by inner CV inside train only; require bootstrap 10th pct > 0 on BOTH folds; paired LB probe mandatory before adoption.",
    "test": "Keep rule plus: factor-level sign check vs the 25 Oct A/E list (a factor flipping sign vs its prior is a red flag)"
  },
  {
    "name": "New-business segment split of adaptation and level",
    "mechanism": "Fit shape recalibration and key correction factors separately on new vs renewal training rows, and let the effective level differ by segment (fit_predict applies a segment factor; the harness balance then adjusts the whole line). 2023 emerging is 100% new business, so the new-business signal is clean. Motor NB A/E 1.20/1.07 and contents 1.18/1.43 say the base underprices new business even after the k=200 count factor.",
    "expected_gain": "+0.001 to +0.003 primary; transfers because 76% of the 2025 book is new business",
    "harness_fit": "Inside fit_predict: segment interaction flags; settings: which corrections split, shrinkage of the split toward pooled.",
    "compute": "Minutes",
    "where": "Mac, agent loop",
    "hours": 0.5 per run; 2-3 runs",
    "overfit_risk": "Medium: renewal rows are thin (~24% of 2025, more of train). Test: shrink split factors 50% toward pooled; require both folds positive; verify on the reweighted (2025-like) validation component specifically, since that is where NB dominates.",
    "test": "Keep rule; report gain separately on natural vs reweighted book"
  },
  {
    "name": "Geo-climate parametric offsets (extrapolation layer)",
    "mechanism": "Trees clamp on emerging values outside the developed range (hail days 14-17 vs ~6, elevation 0-3 m, distance_to_river, wildfire index). Fit small monotone log-link GAMs per weather/geo peril (hail, flood, storm) on the POOLED developed+emerging frame, with the base as offset, and apply the fitted coefficients as per-row multiplicative offsets. GAMs extrapolate where trees cannot; this targets elevation A/E 1.11-1.19 vs 0.80-0.83 and buildings flood zone 3 at 1.29-1.61.",
    "expected_gain": "+0.001 to +0.003 primary",
    "harness_fit": "New base-family-style fit (owner script) producing a coefficient table; experiment file applies offsets by row lookup. fit_predict change is trivial.",
    "compute": "Load 3M-row frame once; GAM fits minutes. Kaggle CPU 4-core is fine.",
    "where": "Kaggle CPU notebook (data already attached)",
    "hours": 1.5-3 total including frame load",
    "overfit_risk": "Low-medium: functional form is pre-specified and monotone. Test: check the fitted geo/ weather response is sensible across the developed range before applying; fold gains gated as usual; LB paired probe because 2025 weather (hail 11.4) sits outside both training years' range.",
    "test": "Keep rule + mechanism prior: offsets should reduce elevation/flood-zone A/E segments specifically"
  },
  {
    "name": "Buildings physical relativity transfer from developed",
    "mechanism": "Estimate relativities for energy_rating, trees_within_5m, flood_zone, construction/soil on the DEVELOPED buildings book (160k claims, stable), verify sign consistency with the emerging A/E list (energy E 1.51-1.84 vs A 0.58-0.73; trees 1.32-1.64; non-standard construction 1.39-1.46), then apply as multiplicative corrections blended w toward the emerging own-estimate with credibility. Physics transfers across markets even when frequencies don't; the emerging per-segment A/E is too noisy (500 claims/yr) to estimate alone.",
    "expected_gain": "+0.001 to +0.002 primary (buildings is a minority of pooled exposure)",
    "harness_fit": "New base-family output (relativity table from owner script); applied and credibility-gated inside fit_predict.",
    "compute": "One developed-frame GLM session",
    "where": "Kaggle CPU (parallel with Mac loop)",
    "hours": 1.5-2.5",
    "overfit_risk": "Low: parameters estimated on 3M rows, shrunk on application. Test: sign check vs emerging A/E is the gate; if signs disagree for a factor, drop that factor (interaction with different construction codes is the failure mode).",
    "test": "Keep rule; per-factor sign gate before the fold gate"
  },
  {
    "name": "base_compat blend retest under the O4 stack",
    "mechanism": "The 25 Oct trial gave compat-alone 0.11622 and 50:50 blend 0.11792 vs P0's 0.11554 - a +0.0024 primary effect that appears to have been lost to sequential gating. Retest the blend INSIDE the current O4 pipeline at tuned weights per line. Mechanism: mapping bands to in-band means and feeding missing credit/mileage/telematics into the developed model sends emerging rows down arbitrary tree paths; the compat base avoids those paths.",
    "expected_gain": "+0.001 to +0.002 primary",
    "harness_fit": "Inside fit_predict: pred = (1-w)*P0 + w*compat, w grid {0.2, 0.35, 0.5} per line. Base already fitted.",
    "compute": "None new; seconds",
    "where": "Mac",
    "hours": 0.5",
    "overfit_risk": "Low: the base is fold-safe by construction and the earlier evidence is positive. Test: standard keep rule; also check it does not erase the gender GLM's gain (interaction of blend and adaptation layer).",
    "test": "Keep rule; compare also on level-neutral fold B"
  },
  {
    "name": "previous_insurer credibility encoding for new business",
    "mechanism": "Largest single-segment A/E spread in motor (0.47-1.69), and the defining covariate of a broker-sourced book: where a customer came from signals their risk. Credibility target encoding of previous_insurer on train rows (counts, k grid 50-500, LRT-gated like gender), applied to motor and contents; unseen 2025 insurers get shrunk to the market level.",
    "expected_gain": "+0.001 to +0.002 primary",
    "harness_fit": "Inside fit_predict, single gated factor with interaction on new_business.",
    "compute": "Minutes",
    "where": "Mac, agent loop",
    "hours": 0.5-1",
    "overfit_risk": "Medium-high: 2025's insurer mix differs from 2023-24 new business, so fold gains may overstate transfer. Test: cap encoding influence (shrink hard, k>=100), check performance specifically on reweighted 2025-like validation, and paired LB probe before keeping.",
    "test": "Keep rule + require gain on the reweighted book component"
  },
  {
    "name": "Pooled base v2 with proper weighting (no duplication)",
    "mechanism": "The pooled base failed because row duplication let trees isolate and memorise emerging claims. Redo it as a single shallow LightGBM (max_depth 4-5, min_data_in_leaf 5000, weight column: emerging weight 3-8) on developed+emerging with the market flag, then blend at low weight into motor alongside P0. A shallow, weight-aware tree cannot memorise 7k claims/yr and instead learns shared structure plus market deltas.",
    "expected_gain": "+0.001 to +0.003 primary, wide uncertainty",
    "harness_fit": "New base family (owner script); blend weight is a fit_predict setting.",
    "compute": "Single LightGBM on 3M rows, 4 cores",
    "where": "Kaggle CPU",
    "hours": 2-4 per config; run 2 configs",
    "overfit_risk": "Medium: pooling alone scored 0.0998-0.1089, so blend weight must be small (0.15-0.3). Test: standard keep rule on the blend; reject if either fold drops > 0.001 even when primary rises.",
    "test": "Keep rule; verify emerging claim rows are not individually fitted (check train fit gap between duplicated and weighted versions)"
  },
  {
    "name": "2025-mix importance weighting of the correction fit",
    "mechanism": "The classifier odds used for validation reweighting are never used to weight ESTIMATION. Weight training rows by capped odds (cap ~10, ESS 15-23%) when fitting the gender GLM and the count layer, so the corrections are estimated on the population they will score (young, new, broker).",
    "expected_gain": "+0.0005 to +0.002 primary",
    "harness_fit": "Inside fit_predict: sample weights in the existing GLM/count fits; one setting (cap).",
    "compute": "None beyond existing runs",
    "where": "Mac",
    "hours": 0.2",
    "overfit_risk": "Low-medium: ESS drops to ~15-23%, so standard errors rise; weighting mainly matters for factors with age/NB interactions. Test: keep rule; require the gain to appear on the reweighted validation component (where it should, by construction) AND not hurt the natural book.",
    "test": "Keep rule split by validation component"
  },
  {
    "name": "prior_claims_5y / hist fields audit",
    "mechanism": "The Part 1 admissibility re-test excluded 'customer-level claim history', but the emerging schema explicitly provides prior_claims_5y, hist_n_policy, hist_cost_policy. If the base dropped these wholesale, a credibility factor on self-reported prior claims (count target, motor) is free lift - prior claims is a top-3 motor frequency predictor, and the 24% renewal rows have real history.",
    "expected_gain": "0 to +0.002 primary (binary: either free lift or nothing)",
    "harness_fit": "Inside fit_predict: single gated count factor; or feature re-enablement note for the base.",
    "compute": "Audit: 10 min reading the base feature list. Factor: minutes.",
    "where": "Mac",
    "hours": 0.5",
    "overfit_risk": "Low if gated; the 25 Oct re-test suggests limited value, so expectations should be modest. Test: standard keep rule; drop immediately if < margin.",
    "test": "Keep rule"
  },
  {
    "name": "Young-driver licensing interaction (motor)",
    "mechanism": "2025 has 50% of exposure at ages 17-34 vs 39% in training, and band->mean mapping flattens the age profile exactly there. Fit an interaction correction (age_band x years_licensed<3 x ncb_years) on motor counts with offset log(P0), shrunk. Emerging licensing/NCB regimes plausibly differ from developed, so the developed age curve misprices the young tail.",
    "expected_gain": "+0.0005 to +0.002 primary",
    "harness_fit": "Inside fit_predict: pre-registered interaction, ridge-shrunk; one mechanism.",
    "compute": "Minutes",
    "where": "Mac, agent loop",
    "hours": 0.5",
    "overfit_risk": "Medium: the folds are renewal-heavy and older, so fold gains understate 2025 transfer. Test: keep rule PLUS require gain on the reweighted (young-skewed) validation component and on the 2023 fold (which is younger/newer).",
    "test": "Keep rule + reweighted-component gate"
  },
  {
    "name": "TabPFN subsample-ensemble third opinion",
    "mechanism": "TabPFN v2 is near state-of-the-art on small tabular data; 65k emerging rows fit via 24-32 stratified subsample fits of <=10k rows, ensembled, trained on emerging rows with P0 as a feature. Blend at 10-20% into motor as a diversity component - it may capture emerging-specific interactions the developed-shaped base cannot express.",
    "expected_gain": "0 to +0.002 primary (diversity lottery ticket)",
    "harness_fit": "New base family output (per-row predictions file); blend weight in fit_predict.",
    "compute": "GPU T4; 24-32 fits of 10k rows",
    "where": "Kaggle GPU",
    "hours": 3-5 total",
    "overfit_risk": "Medium: folds will flatter it (it trains on emerging). Test: blend weight capped low, both folds must improve, paired LB probe mandatory before adoption.",
    "test": "Keep rule + LB paired probe"
  },
  {
    "name": "Weather-year index for 2025 (hail/wind/storm perils)",
    "mechanism": "Training years had hail 14-17 days; 2025 has 11.4. Balance-on-train sets peril levels to the hailier training weather, so 2025 hail/wind-exposed perils are overpriced at the level. Fit a log-link weather response per affected peril on the POOLED frame (where developed gives spatial weather variation), then apply the 2025 realised weather values. The earlier scalar hail elasticity failed; this uses a pooled, mechanism-based response instead.",
    "expected_gain": "+0.0005 to +0.0015 board (level effect invisible to folds)",
    "harness_fit": "Coefficient table from a pooled owner script; applied as peril-level multipliers inside fit_predict using the row's 2025 weather.",
    "compute": "Same pooled-frame session as idea 4 (share the load); minutes of fitting.",
    "where": "Kaggle CPU, same session as geo offsets",
    "hours": +0.5 beyond idea 4",
    "overfit_risk": "Cannot be validated on folds (both folds share training weather). Test: mechanism gate only (response must be monotone and consistent between developed regions and emerging), then a dedicated paired LB probe: this index alone vs without, same day.",
    "test": "Pre-register direction (2025 milder => lower hail/wind peril predictions), verify sign, decide by LB paired probe"
  }
]
```
