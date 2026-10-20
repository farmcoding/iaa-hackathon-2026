## Assessment

**Motor looks close to its practical ceiling with these targets and predictors—not demonstrably at its statistical ceiling.** The 2024 deciles show little systematic compression or over-dispersion, and genuinely executed changes have produced small gains. I would not expect another improvement comparable to the original BI frequency split.

The best remaining opportunities are **coverage-specific structure in other perils** and **removing irrelevant predictors from BI without weakening other perils**. Most proposed experiments will probably fail the pooled keep rule.

### Establish the benchmark first

The evidence does not describe one consistent champion:

- Header: primary **0.14048**, motor B **0.08492**.
- Latest ledger keep: primary **0.14227**, motor B **0.08546**.
- Supplied code additionally contains **BI-only 31-leaf/500-minimum loosening**, apparently not yet a kept experiment.

First score the supplied code, with property frozen, and verify that motor predictions actually change. The gains below are **incremental to that reproducible baseline**, not to 0.14048.

## Ranked experiments

Gain ranges below are judgmental estimates of **positive motor D² improvement**, not confidence intervals; every experiment has a material probability of zero or negative gain. Run independently before combining.

### 1. Explicit, restricted coverage interactions in motor-other EBM

**Mechanism:** Coverage dominates the residual interaction table. More importantly, the observed vehicle-age relationship changes by cover: comprehensive costs eventually decline, while TPO costs rise substantially. An additive EBM cannot reproduce this.

**Exact change:** For **motor-other EBM only**, replace `interactions=0` with:
```python
[
    ("cover_type", "driver_age"),
    ("cover_type", "vehicle_age"),
    ("cover_type", "urban_density"),
]
```
Convert names to column indices if required; set `max_interaction_bins=8`. Keep the existing 50/50 LightGBM–EBM blend, main-effect settings and monotone constraints. Do not include years licensed alongside age in this first test.

**Expected:** B **+0.0004–0.0012**; A flat to **+0.0008**.  
**Risk:** Automatic interactions already failed. These deliberately selected pairs may still duplicate what LightGBM captures.

### 2. Prune vehicle/severity predictors from BI only

**Mechanism:** Several vehicle predictors matter for other costs but have effectively zero or negative BI contribution. They also generate BI direction/ICE flags. Previous vehicle-group removal damaged both submodels; that is not evidence against BI-only removal.

**Exact change:** Give `bag()` separate feature matrices and monotone vectors. From **BI only**, remove:
```python
cover_type, vehicle_age, vehicle_value_group, fuel_type,
overnight_parking, make_model, body_type, seats, engine_size_cc,
voluntary_excess, missed_payments_12m
```
Leave all other BI settings, its target and its anchor unchanged. Leave motor-other untouched.

**Expected:** B **+0.0003–0.0010**; A flat to **+0.0007**.  
**Risk:** Small real BI effects may be lost. Removing features also changes feature-subsampling behaviour, so fresh-seed confirmation matters.

### 3. Add a pooled frequency–severity challenger for other perils

**Mechanism:** Coverage and vehicle characteristics mix frequency with severity. Splitting into four separate cost models lost credibility; a **pooled** frequency–severity model tests that mechanism without four separate samples.

**Exact change:**
```python
N = sum(tr[f"n_{p}"] for p in
        ("own_damage", "third_party_prop", "theft", "glass_hail"))
```
Fit:

- Frequency: Poisson, target `N/exposure`, weight `exposure`; 15 leaves, minimum 2000, 1200 rounds.
- Severity: Gamma on rows `N > 0`, target `other.clip(lower=1e-6)/N`, weight `N`; 7 leaves, minimum 1000, 600 rounds.
- Use the existing feature matrices, seeds and applicable monotone constraints.
- Anchor `frequency * severity` to `other_lc`; blend **25% challenger / 75% current anchored other rate**.

**Expected:** B **+0.0002–0.0009**; A flat to **+0.0007**.  
**Risk:** Reported counts and developed costs may be mismatched, particularly in younger years. This is not the already-failed four-peril split.

### 4. Reparameterise age and driving experience

**Mechanism:** Age and years licensed are highly collinear. Their two-way table has little support away from the diagonal, making interaction and ICE diagnostics unreliable. The potentially informative quantity is age at obtaining a licence.

**Exact change:** On motor, replace `years_licensed` with:
```python
age_at_licensing = driver_age - years_licensed
```
Retain `driver_age`. Preserve missingness; set impossible values outside `[17, driver_age]` to missing. Apply to both submodels; no new monotone constraint.

**Expected:** B **+0.0001–0.0007**; A flat to **+0.0005**.  
**Risk:** The difference may mostly be recording noise. Trees may currently benefit from two correlated measurements of age.

### 5. Native-categorical CatBoost Poisson challenger for BI

**Mechanism:** BI has only one effective learner. Its EBM challenger failed, but that does not test native categorical estimation. Current high-cardinality encodings are learned from combined cost, not BI frequency.

**Exact change:** Fit CatBoost on the **raw admissible BI predictors**, without target encoding:
```python
loss_function="Poisson"
iterations=600
depth=4
learning_rate=0.03
l2_leaf_reg=20
```
Use three `SEEDS`, categorical missing-value conversion, and numeric monotone constraints from `MONOTONE`. For this experiment retain the incumbent BI target/weight convention to isolate learner diversity. Blend **25% CatBoost / 75% LightGBM frequency**, then apply the existing BI anchor.

**Expected:** B **+0.0001–0.0006**; A flat to **+0.0005**.  
**Risk:** BI is sparse, and prior blend gains have largely been exhausted.

### 6. One telematics–vehicle interaction in motor-other EBM

**Mechanism:** `vehicle_value_group × telematics_score` is the strongest motor-other FAST pair and has a defensible mechanism: driving behaviour can have different cost consequences for different vehicles.

**Exact change:** Starting from the baseline—not experiment 1—set motor-other EBM:
```python
interactions=[("vehicle_value_group", "telematics_score")]
max_interaction_bins=8
```
Keep the 50/50 blend. Do not add the weather pairs or automatic interaction selection.

**Expected:** B **+0.0001–0.0005**; A flat to **+0.0004**.  
**Risk:** FAST strength is not incremental predictive value. Missing telematics scores could be driving the result.

### 7. Re-test Tweedie power on motor-other alone

**Mechanism:** Power 1.1 was selected when motor still had a combined cost target dominated by BI. That is not the current other-perils problem. The current EBM already uses 1.5.

**Exact change:** Pass a motor-other-specific LightGBM parameter dictionary with:
```python
tweedie_variance_power=1.3
```
Change nothing else; retain the EBM and 50/50 blend.

**Expected:** B **0–+0.0005**; A likely similarly small.  
**Risk:** The original power sweep favoured 1.1. This is a narrow revisit justified by the changed target, not a general hyperparameter search.

### 8. Shrunken, coverage-specific residual spline correction

**Mechanism:** Smooth GLMs showed some motor promise, but adding another full main-effect model to the current blend barely helped. A correction restricted to the identified coverage residuals is a different test.

**Exact change:** Within each training fold:

1. Fit the current motor-other model on years before the latest training year; predict that latest year.
2. Let those predicted rates be `r`. Fit `TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000)` to target `other_rate/r`, weight `exposure*sqrt(r)`.
3. Predictors: cover indicators and cover-gated age/vehicle-age cubic splines—`n_knots=5`, constant extrapolation; scale the design. Exclude all monotone-constrained factors.
4. On outer validation, multiply the full-training baseline other rate by:
```python
np.exp(0.5 * np.clip(np.log(multiplier), np.log(0.8), np.log(1.25)))
```
Then re-anchor.

**Expected:** B **0–+0.0004**; A less reliable.  
**Risk:** Only one training year identifies the correction. Previous recalibration failures justify the strong restriction and shrinkage.

### 9. Saturate NCB after nine years

**Mechanism:** NCB is overwhelmingly the strongest predictor, but both BI frequency and other cost largely flatten after nine years. Forced continued decreases can fit noise in the tail.

**Exact change:** Before encoding, for motor training and validation:
```python
ncb_years = ncb_years.clip(upper=9)
```
Keep the original name and its `-1` constraint; change both submodels.

**Expected:** B **0–+0.0004**; A similar.  
**Risk:** Existing trees can already learn a plateau. Other-peril costs still decline somewhat beyond nine years.

### 10. Remove weak environmental nuisance predictors from other only

**Mechanism:** Other-peril permutation contributions are negative or negligible for several retained environmental variables, with contradictory directions and suspicious weather–weather interactions.

**Exact change:** Remove from **motor-other only**:
```python
dry_summer_index, distance_to_river_km,
wildfire_index, mean_temperature_c
```
Retain `freezing_days`, whose other-peril contribution is appreciably larger. Do not alter BI.

**Expected:** B **0–+0.0003**; A similar or slightly more favourable.  
**Risk:** Small genuine geographic/weather effects may disappear. Positive pre-pruning importance is weak evidence, but not no evidence.

## What looks wrong or suspicious

1. **Stale baselines and unkept code changes.** Resolve the three different benchmark states before interpreting any incremental gain.

2. **Frozen-model no-ops invalidate parts of the ledger.** Byte-identical motor results are neither successful nor failed tests. Conversely, the genuinely executed four-peril split at 04:48 was clearly worse; do not keep relabelling it “untried.”

3. **BI count/exposure specification is mathematically questionable.** Raw count weighted by exposure is not a frequency-rate fit. The usual specification is `count/exposure`, weight exposure. However, that correction already worsened validation: investigate exposure/count semantics rather than presenting it as a new likely gain.

4. **Capped A/E is not calibration against the scored target.** Comparing capped actuals with uncapped predictions mechanically lowers A/E, especially for BI-heavy segments. Fold A’s widespread low ratios are not automatically ranking defects. Also, **A/E below one means overprediction**, contrary to several ledger descriptions.

5. **Interaction diagnostics need qualification.** H-statistic **1.9987** needs an implementation/normalisation check. Age–licence interactions and off-manifold ICE curves are especially unreliable under their strong dependence. FAST strengths alone do not establish residual interactions.

6. **“Wiggles” in unordered categories are not direction violations.** Region, colour and body type have no natural monotone order. Those flags should not drive constraints.

7. **Shared cost target encoding partly undermines the BI split.** It can carry severity noise into BI predictors. That is a valid concern, but separate-target encoding was already tested unsuccessfully; it is not a fresh discovery or guaranteed fix.

8. **The property early-stopping probe is not cleanly time-cross-fitted.** Its feature encodings are constructed using the whole outer training window before the inner temporal split. This can contaminate round selection, though not directly leak outer validation targets.

9. **The Pareto fit uses policy-row ultimates, not individual claim severities.** Check its consistency with the configured BI count–severity development override. This is an audit issue, not permission to change frozen assumptions.

**Bottom line:** I would start with experiments **1 and 2**. Using 46% as a rough deviance weight, the pooled keep margin requires approximately **+0.0011 motor D² in both folds**, or **+0.0016 in B alone** if A is flat. Few remaining ideas have convincing evidence of that much headroom.