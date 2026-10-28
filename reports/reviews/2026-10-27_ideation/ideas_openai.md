# Executive recommendation

The best remaining opportunity is **better transport of conditional risk**, not another unrestricted learner or another collection of marginal cost factors.

Your current adaptation can change peril levels, gender and new-business relativities, but largely inherits the developed market’s **within-peril pure-premium ranking**. That ranking combines frequency, severity, deductibles, reporting and settlement effects. Several of those mechanisms demonstrably differ between markets.

I would use the next 40 hours to:

1. **Build one frequency–severity transfer family**, retaining strong developed-market priors.
2. Run a small queue of **count-led, anchored emerging-market adaptations**.
3. Test **low-dimensional recalibration of the inherited risk score**.
4. Audit whether validation is rewarding the right portfolio—and whether exposure and weighting are implemented as intended.
5. Keep O4 as the protected fallback. Use the public board as a diagnostic, not a parameter-selection device.

A further **+0.0034 public D² is plausible**, but neither the aggregates nor the historical public/fold relationship supports a reliable probability of reaching it. Primary gains do not translate one-for-one into leaderboard gains.

---

# 1. Constraints: what is probably costing predictive power?

The ranges below are **rough opportunity sizes in primary D²**, not measured effects, confidence intervals or additive contributions.

| Rank | Constraint | Likely opportunity cost | Recommendation |
|---|---|---:|---|
| **1** | Adaptation is mostly confined to fixed developed pure-premium scores and a few multipliers | **0.001–0.006** | Relax. Add frequency/severity bases and one tightly regularized source-model refresh. Preserve O4’s successful components. |
| **2** | Validation is dominated by historical composition, while deployment is mostly broker new business | **0–0.004**, through model selection | Add an explicit target-portfolio diagnostic and common-support analysis. Do not change the headline metric after seeing candidate results. |
| **3** | Borrowed development, settlement/recovery treatment and trends are treated largely as fixed | **0–0.003 if materially wrong; otherwise little** | Audit immediately. Change targets only for actuarially defensible reasons, not because a revised target improves its own validation score. |
| **4** | The hard +0.002 threshold applies to small, mechanistically useful changes | **0–0.002** | Relax the **screening** threshold, not automatically the deployment threshold. Permit a small predeclared combination of independently supported components, followed by the full keep rule. |
| **5** | Pooled learning has not received a clean, strongly anchored domain-adaptation test | **0–0.003**, overlapping rank 1 | Test proper weighting or source-parameter shrinkage—not duplicated rows and not an unconstrained market-flag model. |
| **6** | Heavy experiments are effectively constrained to the Mac | **0–0.002**, overlapping model opportunities | Use one private Kaggle CPU session for heavy work. Keep the Mac for emerging-only experiments. GPU is not the priority. |
| **7** | One mechanism per run, three settings, deterministic reproduction, human submission review | Usually **protects** predictive power | Keep. Relax only the inability to construct a small, predeclared combination after component diagnosis. Do not resurrect the eleven-factor stack. |
| **8** | Feature exclusions and submission limits | Probably small honest cost, based on your tests | Keep the exclusions. Ten submissions/day is ample for this programme. More submissions would mainly increase selection risk. |

The small emerging sample—especially property—is a larger limitation than several operational constraints, but it is not something you can honestly “relax.” The response is partial pooling, not more flexible target-only models.

## Two important corrections to the current interpretation

### Low reweighting ESS is partly unavoidable—not necessarily a bad propensity model

Using the stated new-business shares alone, exact target weighting would give approximately

\[
w_{\rm NB}=0.76/0.14=5.43,\qquad
w_{\rm renewal}=0.24/0.86=0.279.
\]

The implied ESS fraction is approximately

\[
\frac{1}{0.14(5.43)^2+0.86(0.279)^2}=0.239.
\]

Thus **about 24% ESS is already implied by the new-business shift**, before age and other differences. Your 15–23% is not automatically evidence of pathological weighting.

However:

- A classifier using weather-year identifiers can create extreme weights without improving transportability.
- Forcing ESS above 50% would necessarily abandon much of the true new-business shift.
- Clipped training weights can be sensible regularization, but must not be described as fully matching the 2025 book.

The current two-thirds-natural, one-third-target objective also substantially underrepresents deployment new business. As a rough loss-mass illustration, ignoring D² normalization and other shifts, its new-business share is closer to **35% than 76%**.

### Row duplication is not inherently a different statistical weighting scheme

Under an otherwise identical additive loss, duplication and sample weighting can be equivalent. The problems arise from implementation:

- duplicated copies crossing folds;
- row-count-based leaf constraints treating copies as independent observations;
- bagging and sampling behaviour;
- regularization changing relative to total training weight;
- excessive target-domain influence.

Replace duplication with explicit weights, but also inspect those mechanisms. **Changing only the API from duplication to `sample_weight` will not necessarily fix memorization.**

## Safe validation adjustments

I would freeze these before the next experiment batch:

1. **Keep the existing primary and production keep rule** for comparability.
2. Add a separate, fixed target-book diagnostic:
   - line;
   - new business versus renewal;
   - broad age/licensing groups;
   - a few supported broker-related strata.
3. Report **deviance differences**, not averages of separately calculated segment D² values.
4. Treat fold B as the temporal read and fold R as a robustness read—not equally strong evidence of forecasting performance.
5. Bootstrap policy/customer chains where identifiers permit. For weather and large losses, also show leave-region and large-claim influence diagnostics. Two years do not support a convincing climate-generalization confidence interval.
6. Recognize that a 10th-percentile bootstrap gate **does not correct for trying many candidates**. Freeze a short candidate queue and retain the complete run ledger.

---

# 2. White space: what is genuinely missing?

Several disciplines have already been used; calling them “new” would overstate the opportunity.

| Discipline | What is already present | Meaningful remaining white space | Fit now |
|---|---|---|---|
| **GLM/GAM** | Offset GLM, EBM, prior spline blends | Anchored count GAMs; source-informed smooths; protected coefficients | **High** |
| **Credibility** | Peril levels, count factors, target encoding | Joint partial pooling across market, peril, new business and year | **High** |
| **Bayesian hierarchical models** | Credibility-like components | Explicit varying effects with shared priors and posterior uncertainty | **High**, using penalized/empirical-Bayes approximations |
| **Frequency–severity** | Count-based adjustment factors | Full conditional decomposition, with separate transfer of frequency and severity | **Highest-priority new family** |
| **Boosting** | LightGBM, CatBoost, EBM, free residual boosting | Source-anchored leaf refresh; properly regularized cross-market learning | **Medium–high** |
| **Neural tabular models** | Tried in Part 1 | Emerging-specific or pretrained tabular models | **Low in 40 hours** |
| **Transfer learning** | Mapped source predictions and market-flag pooling | Parameter shrinkage toward source models; conditional teacher marginalization | **High** |
| **Domain adaptation** | Adversarial validation and evaluation reweighting | Support-aware training weights; missingness robustness | **Medium–high** |
| **Multi-task models** | Separate peril models and line-level shrinkage | Shared physical-hazard effects across property lines/perils | **Medium–high** |
| **Stacking** | Several blends already | Constrained arithmetic OOF stacking of genuinely different mechanisms | **Medium**, late in the queue |
| **Distillation** | Not evident | Teacher averaging over unavailable/coarsened variables | **Medium**, especially age bands |
| **Delay/survival models** | Borrowed reporting patterns | Truncation-aware market-specific reporting diagnostics | **High audit value; uncertain score lift** |
| **AutoML/bandit search** | A controlled automated search already exists | Better allocation between model families | **Low incremental value** |

I would not spend the first night on neural networks, generic AutoML, spatial interpolation, or another broad feature search.

---

# 3. Ranked ideas

## How to read the estimates

These are **planning judgments relative to O4**, not empirical forecasts. The central estimates are intentionally modest; many individual trials will fail. The ranges include plausible negative results and are not statistical intervals.

All ideas must ultimately pass **K**, meaning your complete existing keep rule:

- primary gain \(>0.002\);
- neither fold down \(>0.003\);
- level-neutral B not down \(>0.002\);
- paired-bootstrap 10th percentile \(>0\);
- deterministic reproduction.

An idea can remain useful as a diagnostic or component without qualifying for deployment alone. Do not add its estimated gain to another idea’s estimate.

Compute estimates are approximate **run time after implementation**, not guaranteed delivery time. First-use Kaggle setup and coding are additional.

---

## 1. Frequency–severity transfer bridge

**Expected primary gain:** approximately **+0.0020**; plausible trial range **−0.003 to +0.007**.

### Mechanism

Construct, by peril,

\[
\widehat\mu_i=\sum_p e_i\,\widehat f_{ip}\,\widehat s_{ip},
\]

where frequency and severity have **separate developed-market models and separate emerging-market shrinkage**.

For reported counts, prefer an observation model such as

\[
N^{\rm reported}_{ip}\sim
\operatorname{Poisson}\!\left(e^{\rm reported}_{ip}f_{ip}\right),
\]

with reporting-adjusted exposure, rather than simply treating developed fractional counts as equally precise observations.

Severity should estimate the conditional **mean net ultimate cost**, including a coherent treatment of zero-cost non-declined claims. Do not cap the mean because the emerging sample lacks an 11.9m claim.

### Why it could transfer

The observed shift is explicitly different for frequency and severity. A pure-premium multiplier cannot generally reproduce that change across ages, vehicle values, excesses and perils. Counts also offer substantially cleaner supervision than sparse costs.

### Harness and compute

**New base family**, with fold-safe frequency and severity predictions attached by the owner. Retain O4’s adaptation architecture initially, replacing only the underlying family.

**Kaggle CPU:** roughly **2–5 hours**, sequential fits. Severity training is much smaller than frequency training.

### Risk and test

Separate modelling can add error and need not beat direct Tweedie prediction. Require **K on total cost**, plus count and severity diagnostics by new business, age and peril. Check that improvement is not merely a different motor level or one large claim.

---

## 2. Protected count-residual adaptation

**Expected primary gain:** **+0.0015**; range **−0.002 to +0.0045**.

### Mechanism

Keep O4’s existing gender and other successful components fixed. Learn one small additional **count-risk block**, not another cost-factor stack.

A useful motor block is:

- `previous_insurer`;
- `missed_payments_12m`;
- `licence_points`.

Use partial pooling, rare-level shrinkage and a small number of penalty settings.

If frequency bases from idea 1 are unavailable, fit a nuisance count model using exposure, the existing risk score and established controls. Compare an extended count model with that nuisance model, and use their prediction ratio as the candidate frequency correction. **Do not treat a cost base as if it were an expected count.**

### Why it could transfer

These variables distinguish the broker/new-business population and already show persistent residual structure. Count learning is less exposed to severity noise.

### Harness and compute

**Inside `fit_predict`**, using training-only count labels; attach those aggregates if not already available.

**Mac:** **0.1–0.5 hours** for the candidate batch and diagnostics.

### Risk and test

A frequency improvement can be offset by lower conditional severity. Require **K**, positive count evidence in both folds, and no clear opposing severity effect. Freeze the existing gender component rather than jointly re-estimating and shrinking it away.

---

## 3. Hierarchical broker/new-business adaptation

**Expected primary gain:** **+0.0013**; range **−0.002 to +0.0045**.

### Mechanism

Replace—not multiply—the current single new-business factor with a small hierarchical model:

- line-level new-business effect;
- motor deviation by broad age/licensing group;
- strongly pooled deviation by previous insurer, only where supported;
- year-specific effects shrunk toward a common effect.

Use counts first. Examine a heavily shrunk new-business severity ratio as a falsification check before assuming count A/E translates directly into cost A/E.

### Why it could transfer

New business is the dominant deployment population. A historical average new-business multiplier may not describe a younger broker cohort with different previous insurers and licensing histories.

### Harness and compute

**Inside `fit_predict`**. A penalized GLM or empirical-Bayes implementation is sufficient.

**Mac:** **0.2–0.7 hours**.

### Risk and test

“New business” is not synonymous with “broker acquisition.” Unsupported target combinations must revert toward the broad effect, not receive extrapolated interactions.

Require **K**, plus same-direction new-business deviance improvement in both folds. Inspect contents separately: its count signal is promising, but its sample is small.

---

## 4. Anchored risk-score recalibration

**Expected primary gain:** **+0.0012**; range **−0.002 to +0.004**.

### Mechanism

Fit a low-dimensional correction to the inherited **rate** score:

\[
\mu_i^{\rm new}
=
\mu_i^{\rm O4}
\exp\!\left[h\!\left(\log(P0_i/e_i)\right)\right].
\]

Start with motor, using a slope and at most one or two smooth bends, strongly shrunk toward zero. Keep O4’s explicit factors intact.

This is not another decile-factor table and not unconstrained isotonic regression.

### Why it could transfer

The source market may impose the wrong spread or curvature of risk after severity changes and schema coarsening. A smooth correction can repair that without estimating dozens of new segment effects.

The decile A/E pattern does **not** prove that all predictions should be compressed: low-decile overprediction and middle-decile underprediction can require curvature rather than one slope.

### Harness and compute

**Inside `fit_predict`**.

**Mac:** **0.1–0.4 hours**.

### Risk and test

The observed residual deciles have already been inspected, so flexibility must be tightly limited. Require **K**, stability within new business and renewal, and improvement beyond simple line-level calibration.

---

## 5. Credibility-refresh source leaves

**Expected primary gain:** **+0.0010**; range **−0.003 to +0.0045**.

### Mechanism

Retain source-trained tree structures, but refresh a restricted set of leaf predictions using emerging training data, with a strong penalty toward the original values.

This allows conditional market adaptation while preventing the target sample from inventing new splits. Blend the refreshed component back toward the source component.

Use a supported refit implementation or a carefully tested penalized leaf update; do not assume a library’s “refit” operation has the intended objective and shrinkage.

### Why it could transfer

Developed data can identify useful partitions, while emerging data can reveal that particular source risk cells have different costs. This is materially different from the failed free residual boosting.

### Harness and compute

**Owner-level new base family**. Source trees may be reusable without a complete refit; otherwise refit sequentially. Begin with the LightGBM component rather than rebuilding every ensemble member.

**Kaggle CPU:** **1–3 hours**.

### Risk and test

Many leaf adjustments can still overfit. Restrict the refreshed component and use only a few shrinkage settings. Require **K**, leaf-support diagnostics, and no dependence on a handful of emerging claims.

---

## 6. Support-aware target weighting during adaptation

**Expected primary gain:** **+0.0008**; range **−0.002 to +0.0035**.

### Mechanism

Use target-composition weights in fitting the emerging adaptation—not just in evaluation.

Build weights from a short list of supported transport variables:

- new business;
- age/licensing;
- previous insurer;
- broad vehicle/property characteristics.

Exclude weather-year identifiers from the propensity model. Treat tenure carefully: genuine inception tenure can matter, but calendar/cohort identifiers can produce unsupported separation.

Compare a small number of clipped/shrunk weighting schemes.

### Why it could transfer

With limited model capacity, the fitted approximation should allocate capacity to the population it must price.

### Harness and compute

**Inside `fit_predict`**, with weights estimated from permitted covariates and no test outcomes.

**Mac:** **0.2–0.6 hours**.

### Risk and test

Weighting cannot repair unobserved broker selection or create support where none exists. Clipping introduces deliberate target mismatch.

Require **K using the unchanged evaluation weights**, and report target coverage, ESS, and sensitivity to clipping. Never let a candidate choose its own evaluation population.

---

## 7. Multi-task property hazard adaptation

**Expected primary gain:** **+0.0007**; range **−0.002 to +0.0035**.

### Mechanism

Share a few physically interpretable count effects across related property tasks, while allowing shrunk line/peril deviations.

Examples:

- flood hazard across buildings and contents;
- water-related risk using a small, predeclared physical feature block;
- a separate buildings construction vulnerability block.

Use developed effects as priors where possible. Avoid simultaneously fitting energy rating, construction, property age and several correlated proxies without strong regularization.

### Why it could transfer

Physical hazard relationships should transport more reliably than isolated occupation or marketing correlations. Sharing information is especially valuable with roughly 500 claims per property line per year.

### Harness and compute

**Inside `fit_predict`** with peril-count labels and low-dimensional source-informed effects. Frequency bases from idea 1 improve the offsets but are not mandatory if proper nuisance count models are used.

**Mac:** **0.3–1 hour**.

### Risk and test

Do not interpret the large buildings A/E figures as precise causal effects. Require **K**, peril-specific evidence and influence checks for large claims and individual regions. Prefer one coherent hazard block over a collection of marginal multipliers.

---

## 8. Conditional age-band teacher integration

**Expected primary gain:** **+0.0006**; range **−0.0015 to +0.003**.

### Mechanism

Replace “age band → one mean age” with an average of source-teacher predictions over plausible ages within the band:

\[
E\!\left[f_D(\text{age},x)\mid
\text{age band},\text{years licensed},x_{\rm compatible}\right].
\]

Use a small quadrature or sampling scheme. Compare a developed conditional age distribution with a simple within-band sensitivity distribution.

### Why it could transfer

For nonlinear models,

\[
f(E[\text{age}])\ne E[f(\text{age})].
\]

The problem is largest around young-driver thresholds, exactly where the 2025 exposure share rises. A compatibility refit removes information; this approach instead integrates over the missing information.

### Harness and compute

**Owner-generated new base variant**. If source models are saved, this mainly requires additional scoring and a source age-distribution fit, not full model retraining.

**Mac, one source process, or Kaggle CPU:** **0.5–2 hours**.

### Risk and test

Within-band age distributions may differ between markets. Do not impose an unverified legal licensing-age assumption.

Require **K**, stability under the two age-distribution assumptions, and improvement concentrated in supported age/licensing cells rather than an unexplained global shift.

---

## 9. Metric-correct exposure implementation

**Expected primary gain:** **+0.0004 overall**; usually zero if already correct, potentially material if an implementation mismatch exists.

### Mechanism

Audit the precise relationship between:

- row totals;
- annualized rates;
- exposure offsets;
- sample weights;
- prediction-time offsets.

For Tweedie power \(p\),

\[
d_p(er,e\widehat r)=e^{2-p}d_p(r,\widehat r).
\]

At \(p=1.5\), unweighted row-total deviance is equivalent to rate deviance weighted by **\(\sqrt e\)**, not by \(e\).

Exposure-weighted rate fitting is actuarially conventional and may still estimate a good conditional mean; it is simply not the identical finite-sample objective to this competition metric.

Also verify that any LightGBM initial-score offset is correctly restored at prediction time.

### Why it could transfer

New-business portfolios can have different part-year and start-month distributions. An exposure-related modelling mismatch can therefore become a transport problem.

### Harness and compute

**Inside `fit_predict`** for adaptation models; **owner base wrapper/refit** if the mismatch is in source training.

**Mac:** **0.1–0.3 hours** for audit; **1–3 hours** if a base refit is justified.

### Risk and test

Do not “fix” an implementation merely because it uses conventional exposure weights. First establish the actual objective and compare a matched variant.

Require **K**, plus short-exposure/full-year diagnostics. If no mismatch exists, stop—do not turn this into an exposure-factor fishing exercise.

---

## 10. Missingness-pattern robust source base

**Expected primary gain:** **+0.0003**; range **−0.002 to +0.0025**.

### Mechanism

Train a compatible source model with controlled masking of otherwise available fields, approximating the emerging market’s **joint missingness patterns**.

This differs from the failed compatibility model: it teaches the model how to predict when normally available fields are absent, rather than only removing fields that are always unavailable.

Use one masked version per source row or carefully weighted augmentation; do not accidentally manufacture independent observations.

### Why it could transfer

The rise from 2.5% to roughly 12% missingness can change which tree branches are used. Robust source learning may outperform default missing-value routing learned from a much cleaner market.

### Harness and compute

**New base family**, initially for motor only.

**Kaggle CPU:** **1.5–4 hours**.

### Risk and test

Artificial masking does not reproduce missing-not-at-random risk selection. Run this only if missingness-pattern residuals persist in both folds.

Require **K**, separate observed/missing subgroup diagnostics, and at most a few predeclared masking strengths.

---

## 11. Local severity-surface adaptation

**Expected primary gain:** **+0.0003**; range **−0.002 to +0.003**.

### Mechanism

With frequency held fixed, adapt a very small severity surface around the source severity prediction:

- vehicle value or sum insured;
- excess relative to insured value;
- a shrunk peril-specific slope deviation.

Use claim-level mean-severity modelling, not a positive-policy-cost model that confounds claim count and size.

### Why it could transfer

A market severity scale of 0.6–0.8 does not generally imply the same net-cost multiplier at every excess and insured value:

\[
E[(aX-d)_+] \ne aE[(X-d)_+].
\]

Thus historical peril-level calibration can leave systematic errors when the broker portfolio changes the excess/value mix.

### Harness and compute

**Inside `fit_predict` after frequency/severity artifacts are available**, with training-only claim summaries. Otherwise this is an extension to idea 1’s new base family.

**Mac:** **0.3–1 hour**, excluding the frequency/severity base work.

### Risk and test

Do not reconstruct gross losses by blindly adding deductibles to net amounts; below-excess claims, recoveries and coverage terms may make that invalid.

Require **K**, stable severity evidence across folds and maturity groups, and a tiny number of slope parameters.

---

## 12. Constrained arithmetic OOF ensemble

**Expected primary gain:** **+0.0002**; range **−0.001 to +0.0025**.

### Mechanism

Blend O4 arithmetically with one genuinely different family:

\[
\mu=(1-w)\mu_{\rm O4}+w\mu_{\rm alternative},
\qquad 0\le w\le w_{\max}.
\]

Use cross-fitted predictions, a strong preference for \(w=0\), and at most three candidate weights. Start with one global weight; only consider separate line weights if the evidence is substantial.

### Why it could transfer

A weaker standalone model can still correct errors made by O4. Frequency–severity and source-refresh models are more promising complements than several variants of the same marginal-factor model.

### Harness and compute

**Inside `fit_predict`**, after fold-safe alternative bases exist.

**Mac:** **0.1–0.3 hours**.

### Risk and test

Do not select a large library of losing models and optimize weights against the same two folds. Choose the eligible alternative in advance or from a tightly limited family.

Require **K** on the complete blend. Different public scores alone are not evidence of useful error complementarity.

---

# 4. Target and reserving audit: high priority, but not a score-mining exercise

I would perform this alongside the experiments, rather than make it one more freely tunable model family.

## What needs checking

1. **Reporting completion versus cost completion.**
   Count reporting delays and cost-weighted reporting delays need not be interchangeable, especially for BI.

2. **Right truncation.**
   Emerging reporting-delay estimates must account for claims not yet observed. The empirical distribution of observed delays is biased toward short delays.

3. **Recoveries on open claims.**
   If recoveries are only booked at settlement, an open claim’s current net incurred amount can overstate ultimate net cost unless expected future recoveries are represented elsewhere.

4. **Market-specific settlement and case adequacy.**
   Identical valuation maturity does not imply identical development behaviour.

5. **Declines and zero-cost claims.**
   Ensure frequency and severity definitions reconcile to the target’s exclusion of declined claims.

6. **Trend application.**
   Check both folds’ direction of trending and the separation between loss-cost inflation and weather/exposure changes.

## Important limits

- Without historical reserve snapshots, you **cannot reconstruct a genuine historical case-reserve backtest** merely by filtering today’s claim table to earlier report dates.
- A lower observed emerging maximum is not evidence of an emerging-market severity cap.
- A uniform line-level prediction adjustment will generally be cancelled by the harness’s training balance. A genuine target-level revision therefore requires an explicit owner-controlled change, not a hidden factor inside `fit_predict`.
- Do not compare scores calculated against different validation targets as if they measured model improvement.

If this audit finds a material issue, document the target correction and evaluate candidate predictions against a frozen reference plus clearly labelled alternative target scenarios.

---

# 5. The weakest spot

**The adaptation is trying to repair a transported pure-premium ranking with sparse marginal corrections, without first identifying which part of that ranking fails: frequency, severity, or observation/development.**

That is the main source of potentially recoverable lift.

Why I think so:

- The market differences are strongly **peril- and mechanism-specific**.
- The broker book changes the distribution of the very variables whose relativities may not transport.
- Count A/E provides a comparatively clean signal that is not being fully exploited.
- The successful gender adjustment suggests that a small, well-identified target-domain correction can matter substantially.
- The eleven-factor failure and free residual-boosting failures suggest that noisy cost residuals are an unreliable foundation for broad adaptation.
- Non-monotone decile A/E suggests conditional calibration problems, but does not justify an arbitrary decile correction or prove that rankings should simply be compressed.

The second weakness is **selection against a largely historical portfolio objective**. A model can be genuinely better for the 2025 new-business book and struggle to clear a threshold dominated by historical renewals. Conversely, a highly weighted target diagnostic can be too noisy to establish improvement. That tension must be exposed, not hidden behind one primary number.

---

# 6. Suggested 40-hour execution plan

| Window | Work |
|---|---|
| **Hours 0–3** | Freeze O4, seeds, targets and submission artifacts. Audit exposure/offset implementation, duplication behaviour and target-weight support. Freeze the next candidate queue. |
| **Hours 3–14** | Kaggle CPU runs frequency–severity family. Mac runs protected count residuals, hierarchical new business and anchored score recalibration. |
| **Hours 14–24** | Run target-weighted adaptation and one property hazard block. Diagnose frequency/severity results. Start source-leaf refresh only if implementation is straightforward. |
| **Hours 24–32** | Test age-band integration or the strongest remaining diagnostic-led idea. Build one predeclared arithmetic blend if a genuinely complementary family exists. |
| **Hours 32–36** | Reproduce finalists; run cluster/influence diagnostics; inspect target new-business slices and prediction distributions. |
| **Final buffer** | Human review, limited submissions, final artifact validation and governance write-up. Retain O4 unless a challenger earns promotion. |

I would target **three to five substantive submissions**, not use all available slots. Preselect the deployment champion from validation; use an unexplained public deterioration to trigger investigation, not a sequence of public-score-driven factor changes.

## `ideas`

```json
[
  {
    "name": "Frequency-severity transfer bridge",
    "mechanism": "Separate peril frequency and mean net ultimate severity, using developed priors and independently shrunk emerging adaptations.",
    "expected_gain": "Central +0.0020 primary D2; plausible trial range -0.003 to +0.007.",
    "harness_fit": "New fold-safe base family fitted by owner; attach frequency, severity and total predictions.",
    "compute": "Sequential source frequency fits and smaller claim-level severity fits.",
    "where": "Kaggle CPU",
    "hours": "2-5, excluding implementation",
    "overfit_risk": "Moderate: decomposition error, sparse severity effects and reporting assumptions.",
    "test": "Full keep rule on total cost; inspect count and severity performance by peril, new business and age; check large-claim influence."
  },
  {
    "name": "Protected count-residual adaptation",
    "mechanism": "Freeze O4 components and add one partially pooled motor count-risk block; use a proper count nuisance model if frequency bases are unavailable.",
    "expected_gain": "Central +0.0015 primary D2; plausible trial range -0.002 to +0.0045.",
    "harness_fit": "Inside fit_predict with training-only count labels.",
    "compute": "Penalized count GLMs and cross-fitted prediction ratios.",
    "where": "Mac",
    "hours": "0.1-0.5",
    "overfit_risk": "Moderate: categorical sparsity and frequency-severity compensation.",
    "test": "Full keep rule; count evidence in both folds; no clear opposing severity effect; preserve the existing gender component."
  },
  {
    "name": "Hierarchical broker-new-business adaptation",
    "mechanism": "Replace the single new-business factor with strongly pooled line, age-licensing and supported previous-insurer deviations.",
    "expected_gain": "Central +0.0013 primary D2; plausible trial range -0.002 to +0.0045.",
    "harness_fit": "Inside fit_predict; replace rather than multiply the current new-business adjustment.",
    "compute": "Empirical-Bayes or penalized hierarchical count model with severity diagnostics.",
    "where": "Mac",
    "hours": "0.2-0.7",
    "overfit_risk": "Moderate to high: historical new business may not represent broker acquisition.",
    "test": "Full keep rule; same-direction new-business deviance improvement in both folds; shrink unsupported target combinations to broad effects."
  },
  {
    "name": "Anchored risk-score recalibration",
    "mechanism": "Apply a strongly shrunk slope and limited smooth curvature to log base rate while retaining O4's explicit factors.",
    "expected_gain": "Central +0.0012 primary D2; plausible trial range -0.002 to +0.004.",
    "harness_fit": "Inside fit_predict on fixed bases and exposure.",
    "compute": "Low-dimensional offset Tweedie GLM or GAM.",
    "where": "Mac",
    "hours": "0.1-0.4",
    "overfit_risk": "Moderate: residual deciles have already been examined.",
    "test": "Full keep rule; fixed low degrees of freedom; new-business and renewal stability; improvement beyond line-level calibration."
  },
  {
    "name": "Credibility-refresh source leaves",
    "mechanism": "Retain source tree structures and refresh a restricted set of leaf predictions toward emerging outcomes under strong source-value penalties.",
    "expected_gain": "Central +0.0010 primary D2; plausible trial range -0.003 to +0.0045.",
    "harness_fit": "Owner-level new base family, initially using the LightGBM component.",
    "compute": "Source tree reuse or sequential refit, leaf-support aggregation and penalized updates.",
    "where": "Kaggle CPU",
    "hours": "1-3",
    "overfit_risk": "Moderate to high: many leaf adjustments can overfit despite fixed splits.",
    "test": "Full keep rule; restrict the refreshed component; inspect leaf support and claim influence; verify refit objective and prediction semantics."
  },
  {
    "name": "Support-aware target-weighted adaptation",
    "mechanism": "Fit adaptation using clipped or shrunk target-composition weights based on supported risk variables rather than weather-year identifiers.",
    "expected_gain": "Central +0.0008 primary D2; plausible trial range -0.002 to +0.0035.",
    "harness_fit": "Inside fit_predict using permitted target covariates and fixed evaluation weights.",
    "compute": "Small propensity or calibration-weight model and weighted adaptation fits.",
    "where": "Mac",
    "hours": "0.2-0.6",
    "overfit_risk": "Moderate: extreme weights, unsupported cells and unobserved broker selection.",
    "test": "Full keep rule with unchanged evaluation population; report ESS, target coverage and clipping sensitivity."
  },
  {
    "name": "Multi-task property hazard adaptation",
    "mechanism": "Share a few physical count-risk effects across related property lines and perils with shrunk task-specific deviations.",
    "expected_gain": "Central +0.0007 primary D2; plausible trial range -0.002 to +0.0035.",
    "harness_fit": "Inside fit_predict with peril counts and proper count offsets or nuisance models.",
    "compute": "Low-dimensional hierarchical count GLM or GAM.",
    "where": "Mac",
    "hours": "0.3-1",
    "overfit_risk": "High for unconstrained effects because property claim counts are small.",
    "test": "Full keep rule; predeclare one physical hazard block; inspect peril, region and large-claim influence."
  },
  {
    "name": "Conditional age-band teacher integration",
    "mechanism": "Average source predictions over plausible ages conditional on observed age band and licensing information instead of scoring one mean age.",
    "expected_gain": "Central +0.0006 primary D2; plausible trial range -0.0015 to +0.003.",
    "harness_fit": "Owner-generated base variant; source models can be reused if saved.",
    "compute": "Source conditional-age summaries and repeated teacher scoring.",
    "where": "Mac single source process or Kaggle CPU",
    "hours": "0.5-2",
    "overfit_risk": "Moderate: within-band age distributions may not transport.",
    "test": "Full keep rule; compare developed-conditional and simple within-band assumptions; inspect young-driver and licensing cells."
  },
  {
    "name": "Metric-correct exposure implementation",
    "mechanism": "Audit row-total versus rate objectives, exposure weighting and prediction-time offsets; test a matched objective only if warranted.",
    "expected_gain": "Central approximately +0.0004; usually zero if implementation is already appropriate, potentially material if a mismatch exists.",
    "harness_fit": "Inside fit_predict for adaptation; owner base-wrapper change or refit if the issue is in source training.",
    "compute": "Analytic checks, deterministic prediction tests and an optional matched base refit.",
    "where": "Mac, optionally Kaggle CPU for refit",
    "hours": "0.1-0.3 audit; 1-3 if refit is justified",
    "overfit_risk": "Low for the audit; moderate if exposure-specific corrections are subsequently searched.",
    "test": "Full keep rule for any changed model; short-exposure versus full-year diagnostics; stop if no implementation mismatch is found."
  },
  {
    "name": "Missingness-pattern robust source base",
    "mechanism": "Train a compatible source model with controlled masking approximating emerging joint missingness patterns.",
    "expected_gain": "Central +0.0003 primary D2; plausible trial range -0.002 to +0.0025.",
    "harness_fit": "New owner-fitted base family, initially motor only.",
    "compute": "Sequential compatible-source fit with controlled masking and no artificial independent duplicates.",
    "where": "Kaggle CPU",
    "hours": "1.5-4",
    "overfit_risk": "Moderate: artificial masking may not reproduce missing-not-at-random selection.",
    "test": "Run only with persistent missingness residuals; full keep rule; observed-versus-missing subgroup checks and limited masking strengths."
  },
  {
    "name": "Local severity-surface adaptation",
    "mechanism": "Hold frequency fixed and adapt a few source severity slopes for insured value, vehicle value and relative excess.",
    "expected_gain": "Central +0.0003 primary D2; plausible trial range -0.002 to +0.003.",
    "harness_fit": "Inside fit_predict after frequency-severity artifacts exist, otherwise an extension of that new base family.",
    "compute": "Small claim-level mean-severity GLMs with strong source-centered penalties.",
    "where": "Mac",
    "hours": "0.3-1, excluding frequency-severity base work",
    "overfit_risk": "Moderate to high: sparse tails, deductible truncation and maturity effects.",
    "test": "Full keep rule; stable severity effects across folds and maturity groups; no naive gross-loss reconstruction or arbitrary tail caps."
  },
  {
    "name": "Constrained arithmetic OOF ensemble",
    "mechanism": "Blend O4 with one preselected complementary family using a small nonnegative arithmetic weight shrunk toward zero.",
    "expected_gain": "Central +0.0002 primary D2; plausible trial range -0.001 to +0.0025.",
    "harness_fit": "Inside fit_predict after fold-safe alternative bases are attached.",
    "compute": "Small OOF blend search and paired diagnostics.",
    "where": "Mac",
    "hours": "0.1-0.3",
    "overfit_risk": "Moderate if many losing models are searched; lower with one predeclared alternative.",
    "test": "Full keep rule on the complete blend; at most three weights; require OOF complementarity rather than public-score differences."
  }
]
```
