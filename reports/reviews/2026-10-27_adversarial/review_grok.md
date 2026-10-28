The wrong use of the last 40 hours is another round of emerging-residual learning selected on mean(deploy B, deploy R). Tonight already priced that family: it is worth about zero and the reverse year correctly kills it. Shift weights, a frequency-severity refit, and base-family blending are the same bet in a new costume.

## 1. What fold R is, and is not

Fold R is a valid veto on local residual models. It is not a valid ranking metric for the 2025 book, and it is not what is blocking a real gain. The queue is empty of transferable signal; R is just refusing to lie about that.

The absolute gap is not model quality. Deploy B is 0.14199, deploy R is 0.10236, level-neutral B is 0.15356. That 0.040 gap between years, and the 0.012 gap between deploy B and oracle-level B, is the buildings year shock: trended buildings level 585 in 2023 vs 352 in 2024, on ~500 claims. Balance-on-train cannot see a cat year. Mean(B, R) therefore has a year-sampling noise of order 0.02, ten times the 0.002 keep threshold. You are ranking models on one buildings year.

The *incremental* R damage is a different object, and it is real overfitting. Motor count block: k=50 moves R by −0.0155 and B by −0.0001; k=200 is R −0.0078, B +0.0011; k=800 is R −0.0026, B +0.0010, primary −0.0008. Damage shrinks monotonically with shrinkage, and B never shows a gain outside noise. That gradient is memorisation of 2024 claim-count residuals, not a 2025 effect that R is unfairly punishing. Temperature (R −0.0059), the NB×age regime (both folds down at every k), and the loose spline (R −0.0017 at alpha 0.1, alpha 10 identical to O4) say the same thing. Property k=800 at primary +0.00010 is a zero, not a green light.

Why a 2025-specific factor would not show up here, and why that does not rescue these runs: neither fold contains a 77% new-business book, so R cannot confirm a broker-mix effect. It can only reject factors that fail to travel from 2024 to 2023. Anything that cannot travel one year back will not travel to a third year whose mix is further away, unless the factor is a developed-market prior applied as a scoring-function change and not estimated on emerging residuals. The motor count block, the property block, temperature, and the spline are emerging residuals. R's veto stands. Do not ship k=800 because B ticked up. That is the 11-factor stack (+0.0068 folds, −0.0029 board) at one-tenth the size.

Replacement gate, pre-register before any further run. Do not select on deploy primary.

- Residual factor: same sign in 2023-only and 2024-only estimates, magnitude within 50%, at least 200 motor claims or 80 property claims in the treated segment each year, credibility k≥400, at most one new factor per submission. Level-neutral D² is reported, not used as a 0.002 hurdle.
- Scoring-function change (mapping, age imputation): level-neutral D² on each year not down more than 0.001, and a pre-registered cap on the 2025 level move.
- The 1/3 reweight is not a 2025 proxy. Adversarial AUC 0.99 is weather and tenure. Reweighting on that classifier manufactures a weather-shifted history book, which is why a stack can win the reweighted folds and still lose 0.0029 on the board. Drop it from the keep rule.
- Public board: two submissions left, not ten. Board sd ≈ 0.002. The expected upward bias from keeping the best of five board reads is about 0.003, which is the entire gap to 5th and will reverse on the private split.

## 2. Level

The +26% motor level (607 vs 472 / 493) is the largest *calibration* uncertainty and a second-order D² term. It is not the dominant open D² opportunity unless the unverified part of it exceeds ~15%.

Rough decomposition, to be replaced by the table in run 1, not used as a submission:

- Young mix 38%→50%. At a developed young/old relativity of ~1.8, that is about +7% on motor. Real, and already in the base if the age curve is right.
- Organic NB factor. A/E 1.20 then 1.07, k=200, on ~250–350 NB motor claims a year (6.4–9.1% of 3,500, frequency a bit above 1). Pooled shrunk factor ≈ 1.08–1.12. Applied to 77% vs ~8% of history, that is about +6–8% on the 2025 book relative to history, with overlap against the age effect. The two years disagree by 12 points before credibility. Tonight's NB×age run already showed the residual is not an interaction you can estimate. Keep the flat k=200 factor. Do not touch it.
- The rest of the +26% is previous_insurer, vehicle mix, years_licensed, and band-mean age error. This is the unverified part. A broker transfer is not organic new business: AUC 0.74 for 2025 renewal vs NB is make_model, previous_insurer, years_licensed. A flat NB load estimated on the insurer's own 2023–24 new business is the wrong instrument for that book.

D² value of a pure motor scalar error, calibrated off the fold-B level gap. Deploy B 0.14199 vs level-neutral B 0.15356 is 0.0116 of null deviance, mostly a ~50% buildings level miss. For p=1.5 the excess mean deviance of a multiplicative error ε is 4[(1+ε)^{−0.5}+(1+ε)^{0.5}−2] times the μ^{0.5} weight; a 10% error produces about 1/18 of the excess of a 50% error on the same weight. Motor's μ^{0.5} weight is a few times buildings' (claim counts 3,500 vs 500, exposure ratio smaller than that because motor frequency is higher). That puts a **10% motor scalar error at about 0.002–0.003 D², a 5% error at about 0.0005–0.0008, a 15% error at about 0.005–0.006**. Uncertainty about 2×, because the buildings shock is not a clean experiment. Consistent with "a train-set Tweedie scalar was inside noise": the scalar you can fit on history is not the scalar that matters, and a 5% mistake is below board noise.

So a 5–10% argument about the NB scalar is not the path to +0.0034. An 8% standard error on motor level, which is my prior on the unverified slice, is an expected D² loss near 0.0015. Worth one pre-registered correction if the decomposition shows it concentrated in a feature you can actually price; not worth a search.

How to price the broker book without the board. You cannot learn a broker relativity from 300 organic NB claims. You can:

- Price age from the developed curve, and use emerging age-band A/E only as a check. Young drivers are ~38% of history exposure and, at higher frequency, on the order of 1,500–2,000 motor claims a year. That is the only emerging slice thick enough to audit a 2025 mix driver. NB×age is not.
- Price previous_insurer only where both years have ≥30 claims and the same sign, shrunk at k=400 toward 1. Unseen or thin insurer codes, which may be most of a transferred book, get 1.0. No exceptions.
- Do not add a second NB load on top of age and years_licensed. The current factor is already residual to the base; a richer one failed tonight.
- State the level as a range (developed relativities only; plus organic NB on low-years-licensed rows; plus gated previous_insurer) and submit the centre once. Iterating the scalar on the public board will move the score by ~0.002 of noise and cost the private split.

## 3. Queue, killed and kept

| Item | Expected 2025 D² | Decision |
|---|---|---|
| Shift weights inside O4 local fits | −0.005 to +0.001 | Kill. ESS of the existing reweight is 15–23%, and the classifier is weather and tenure. You would be re-fitting the models R just rejected, on a smaller and wrong slice. |
| Frequency-severity developed refit | 0 to +0.002 | Kill for these 40 hours. You are already per-peril. One Mac process, 10–40 min plus a day of specification, blocks the only machine that can audit the base. Transfer to a broker book is the relativity, which a freq/sev split does not fix. |
| Blend P0 / compat / pooled under O4 | 0 to +0.001 | Kill as a search. Compat 50/50 was +0.0024 on the pre-O4 primary; O4 has since taken the schema corrections a blend would proxy. One pre-registered 70/30 is optional and last. |
| Pooled refit at sample weight 5 and 20 | negative at 20 | Weight 20 is duplication with a new name; discard without scoring. Weight 5: keep only if the train-minus-level-neutral gap does not widen by more than 0.01 vs weight 1. Do not blend it tonight either way. |
| Age-band teacher average | +0.0005 to +0.002, tail +0.003 if conditional age is legal | Keep, capped, after the A/E table. |
| Motor/property count blocks, temperature, spline, NB×age | ≤0, measured | Dead. Do not retune k. |

Missing, and higher value than anything still queued: a one-way mix decomposition of the 607, and an age-band A/E with claim counts. Not a model. A table. It decides whether the young half of the 2025 book is mis-ranked.

Five pre-registered runs, in order. No setting is free to move after seeing the score.

**1. Decomposition table. No submission. ~1 hour.**
On the 2024 emerging book, exposure-reweight one margin at a time to the 2025 margin, holding other covariates at 2024 rows. Features, exactly: driver_age_band, new_business, previous_insurer, years_licensed, vehicle_value_group, make_model, fuel_type, cover_type, region, occupation. Plus one joint reweight of (driver_age_band, new_business). Record the multiplicative change in mean O4 prediction. Separately print, for each age band and for new_business, 2023 and 2024 claim counts and cost A/E and count A/E against O4. Pre-registered decision: if both years' 17–34 cost A/E exceed 1.10 with ≥200 claims each, a young-band load of half the average excess, cap 1.15, is allowed in run 2. If both are below 0.90, a discount of half the gap, floor 0.85. Otherwise no residual age load. If previous_insurer's one-way reweight moves the 2025 mean by less than 3%, skip run 3.

**2. Age scoring correction. One submission only if the cap binds inside range.**
Fork on a check you do before fitting anything: if the developed base's partial dependence on years_licensed exceeds 5% across its observed range, years_licensed is already in the score and you must not impute age from it. In that case replace band-mean age with a 5-point average of the peril bases over the uniform within-band age (65+ weighted 0.4/0.3/0.2/0.1 at 65/70/75/80), NB factor and gender GLM held at current values, not refit. Cap: 17–24 mean prediction ≤ 1.15× current 17–24 mean; if the tree average exceeds that, shrink halfway to the cap. If years_licensed is not a material base feature, use age = clip(17+years_licensed, band) blended 50/50 with band-mean for the 17–24 band only, other bands as above, same cap. Success: level-neutral motor D² on each year ≥ −0.001, 2025 motor level change in [0, +8%]. Outside that band, do not submit.

**3. Previous-insurer, motor only, only if run 1 showed ≥3% one-way mix move.**
Multiplicative cost factor, Z = claims/(claims+400), shrink to 1. A level is applied only if 2023-only and 2024-only raw A/E are on the same side of 1 and each year has ≥30 claims. Every other code, including codes that dominate 2025 but are thin in history, is exactly 1. Do not refit gender, NB, or peril levels. Submit only if the non-1 factor covers ≥10% of 2025 motor exposure and level-neutral D² on each year is ≥ −0.001 and the 2025 motor level move is within ±5%. Otherwise record a no-op. Expected gain 0 if the book sits on thin codes, up to ~0.003 if it sits on two or three stable ones.

**4. One property factor and one motor factor, separately, not as a block.**
Motor missed_payments: cost A/E vs O4, k=400, applied only if both years are on the same side of 1 with ≥40 claims in the treated level. This is the only motor segment signal with a tight range (1.18–1.23) and enough claims to be past noise. Buildings energy_rating, one-way, same rule, k=200, minimum 40 claims per treated level per year; construction_type non-standard as a second level under the same rule, not a joint GLM. If either fails the sign gate, it is out. Do not stack the survivors onto run 3 in the same submission unless each individually passed and the combined 2025 level move is still within ±5%. Expected gain 0.0005–0.0015 each. Elevation is not in this run: it has to be recomputed within region first, and if the within-region spread falls under 5% it was a region proxy.

**5. Mapping audit, not a refit, unless a specific bug is written down first.**
One hour on the crosswalk for construction_type, occupation, and make_model. A developed refit is authorised only if you can name a code affecting >5% of 2025 exposure that currently maps to a developed level with relativity 1, or to the wrong level, and the emerging both-year A/E for that code exceeds 1.25 with the claim-count minimums above. Otherwise no refit. This is the only item with a tail to +0.004, and only if the bug is real. Looking at feature importances does not count as finding it.

Leave contents alone. Leave weather alone (you already measured −0.4% / +0.5%). Leave the BI severity ratio as a 30-minute confirmation inside run 1: if the BI peril level is already credibility-weighted at the 0.45–0.62 emerging/developed severity gap, stop. Case-reserve ±10% = 3.4% of motor level ≈ 0.0003 D². Not a lever.

## 4. Is +0.0034 honestly reachable?

Possibly, not probably, and not by the queue as written. Probability about 30% of an honest public gain that still holds on private, concentrated in runs 2 and 3 if run 1 shows a real age-curve miss or a concentrated, well-estimated insurer mix. Expected gain of the honest programme is closer to +0.001–0.002. The gap to 1st, +0.0098, is not available honestly in 40 hours; the Part 1 deficit to the leader was 0.0056 on a book you had 160k claims for, and you did not close it.

What would make top-5 fail even if you find the effect: choosing the submission by public score. Two pre-registered submissions, then stop. A clean written record of the decomposition, the gates, and the decision not to chase a 0.002 board tick is also the part of this a governance judge can verify; five fished submissions are not.
