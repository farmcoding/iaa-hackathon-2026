# Verdict

**The likely waste is another night of residual-factor searches, followed by a full frequency–severity rebuild.** Tonight produced no credible incremental winner. Removing fold R’s veto would not change that: the largest forward-fold gain was only about **0.0011**, and none had positive bootstrap q10.

The important unfinished work is narrower: establish what drives the 2025 mix uplift, quantify its metric exposure, and test at most three independent corrections.

## 1. Fold R is a stress test—not an equal proxy for deployment

The motor count block at \(k=800\) gained **0.00102 on B**, lost **0.00257 on R**, and lost **0.00077 overall**. That is evidence against this implementation, not evidence that R is suppressing a demonstrated breakthrough.

R nevertheless answers the wrong deployment question in several respects:

- It transfers from a less-developed claims year to a more-developed one.
- It reverses calendar and portfolio evolution.
- Each fit has one year of emerging experience; production has two.
- Neither historical book resembles a 77%-NB broker acquisition.

**R should cease being an equal-weight veto, but not disappear.** Its failures expose instability, maturity problems or omitted interactions that may also hurt 2025.

### Two implementation issues to investigate before interpreting the failures

**Count models with cost offsets.** A Poisson count model offset by O4 expected cost estimates frequency *relative to cost*, not a clean frequency residual. If severity varies with age, cover or vehicle, it can learn inverse-severity corrections that you subsequently apply to cost. That can actively damage predictions.

Verify that:

1. count offsets include exposure;
2. expected-count offsets come from frequency models, not normalized cost scores;
3. counts receive appropriate reporting-delay adjustment;
4. O4’s existing NB factor does not share this problem.

**Sample weights versus duplication.** Integer duplication and equivalent sample weights produce essentially the same weighted objective and gradient sums. Differences arise through minimum-row constraints, bagging, ordered categorical statistics, or splitting/encoding mistakes. Weighting is not intrinsically an antidote to memorization. The 5/20 runs are worthwhile only as a controlled implementation comparison—not as a new modelling family.

### Replacement validation

Freeze this **before examining further candidates**:

- Five customer-grouped folds, pooling 2023–24; deterministic hash assignment. This trains local adaptations on roughly 1.6 years rather than one.
- Developed-only bases can remain fixed. **Any base trained on emerging outcomes must be refitted within these folds.** Final pooled predictions are not valid OOF inputs.
- Evaluate paired deviance on the natural book and a separately reported, fully transported book—not their current 2:1 average.
- Transport using fixed row-count post-strata: **line × NB × five age bands × exposure bands \((0,.25],(.25,.75],(.75,\infty)\)**. Use 2025/training cell-proportion ratios, clipped to \([0.1,20]\), then normalize. Report empty cells, clipped target mass and ESS. Include exposure because the metric scores **row totals**, not exposure-weighted rates.
- Aggregate OOF losses before calculating D²; use one denominator per evaluation population.
- For the three candidates below: require transported gain **>0.001**, a positive **3.33rd percentile** from 2,000 paired customer-cluster bootstraps, and B deploy/level-neutral deterioration no worse than **0.001**. Report R without a hard veto.

This is a better estimand, **not fresh evidence**. These labels have already influenced development; bootstrap multiplicity adjustment does not erase that history. Do not resurrect the discarded factor stack under the new harness.

Also, importance weighting cannot manufacture support. Under an equal-exposure approximation, shifting NB from 6.4–9.1% to 77% alone leaves only **11–15% effective sample size**. Further conditioning makes precision worse.

## 2. Motor level is the biggest unresolved extrapolation—not yet the biggest D² opportunity

The comparison **607 versus 472/493** is not evidence of overpricing. It compares different portfolios.

But the uplift deserves decomposition.

With historical NB exposure share \(s_h=0.08\), target share \(s_t=0.77\), and NB relativity \(r\), the mix multiplier after historical balancing is

\[
M_{\rm NB}=\frac{1+0.77(r-1)}{1+0.08(r-1)}.
\]

Thus:

| NB relativity | Contribution to target-book uplift |
|---:|---:|
| 1.07 | 4.8% |
| 1.10 | 6.8% |
| 1.20 | 13.6% |

A shrunk factor near 1.09 explains approximately **6%**, not 26%. Roughly **18% multiplicative uplift** must then come from age and other correlated mix changes.

Moving young exposure from 38–39% to 50% gives only approximately:

- **8–9%** uplift at a young/older relativity of 2;
- **12–14%** at a relativity of 3.

Explaining an 18% residual through that binary age-share change alone requires roughly **4.5–5.5×**. Within-band age composition, licensing, vehicles and cover could explain it—but calculate the contributions rather than calling it “the younger book.”

### What can actually be estimated?

Standardize historical OOF residuals to 2025 using age × NB cells, with gender as an additional motor diagnostic. Separate:

- report-delay-adjusted frequency;
- developed cost per reported/ultimate claim;
- open-claim and case-reserve dependence.

If the quoted exposure shares also hold for motor, historical NB contributes only roughly **600 observed claims across both years**. That gives about **4% count-only sampling uncertainty** before overdispersion; severity uncertainty is appreciably larger. The shift from A/E 1.20 to 1.07 is therefore not strong evidence of a declining NB effect.

Most importantly: **historical new business is not identified as equivalent to acquired broker business**. Unlabelled covariate weighting cannot estimate an acquisition-specific conditional loss effect. The honest output is a transported estimate plus an explicit unidentifiable broker-effect sensitivity—not a confidently “corrected” 2025 mean.

The reserve sensitivity is also not trivial: ±10% case adequacy already moves motor **3.4%**, potentially exceeding several proposed refinements. Reporting-delay backtests are possible from report dates; historical case-adequacy backtests require valuation snapshots, not today’s case reserves retrospectively attached to old dates.

### What is a level error worth?

For Tweedie \(p=1.5\),

\[
d(y,\mu)=4\left(y/\sqrt{\mu}-2\sqrt y+\sqrt\mu\right).
\]

If motor’s true conditional means are \(m_i\), but predictions are \(c\,m_i\), expected D² loss is exactly

\[
4H\left(\sqrt c+c^{-1/2}-2\right),
\qquad
H=\frac{\sum_{\text{motor}}\sqrt{m_i}}{D_0},
\]

where \(D_0\) is the pooled constant-baseline deviance denominator.

| Motor level error | D² loss |
|---|---:|
| +5% / −5% | \(0.00238H / 0.00263H\) |
| +10% / −10% | \(0.00909H / 0.01110H\) |

For illustration—not an estimate of your portfolio:

- \(H=0.10\): 5% costs about **0.00025**; 10% about **0.0010**.
- \(H=0.25\): 5% costs about **0.00063**; 10% about **0.0025**.

Recovering the **0.00336** top-five gap solely by correcting a 10% motor error requires \(H\approx0.30–0.37\). Calculate \(H\) from transported validation before declaring level the dominant metric opportunity.

You can also calculate the **exact observed scalar-loss curve**, without refitting:

\[
\Delta D^2(c)=
-\frac4{D_0}\left[
(c^{-1/2}-1)\sum_M y_i/\sqrt{\mu_i}
+(\sqrt c-1)\sum_M\sqrt{\mu_i}
\right].
\]

This distinguishes a genuinely misplaced level from a visually alarming historical comparison.

## 3. Four runs tonight, in order

These are **one diagnostic package plus three candidates**. Every candidate starts independently from O4; no post-hoc stacking.

### 1. Locked support, level and implementation audit — no submission candidate

Run the validation construction above and produce:

- transported motor scalar-loss curves at **0.90, 0.95, 1.00, 1.05, 1.10**;
- \(H\), NB/age cell support, ESS and clipped target mass;
- OOF frequency and cost residuals by NB × age;
- 2025 mix decomposition, replacing **NB, age, then remaining covariates**, and repeating in reverse order to expose correlation sensitivity;
- the count-offset and duplication/weighting checks above.

These scalar settings are **diagnostics**, not a menu from which to select a submission.

**Highest expected value:** it can stop both a bad level intervention and a useless rebuild.

### 2. One motor NB **total-cost** correction

Test the missing assumption that a count-based NB adjustment is appropriate for cost.

Fit one coefficient \(\delta\), using O4 as offset:

\[
\mu_i'=\mu_i\exp(a_{\text{year}(i)}+\delta\,NB_i).
\]

Settings:

- motor only;
- Tweedie \(p=1.5\);
- unpenalized year intercepts;
- prior \(\delta\sim N(0,0.10^2)\);
- minimize \(\sum d/(2\hat\phi)+\delta^2/(2\cdot0.10^2)\), with \(\hat\phi\) fixed at O4’s training Pearson dispersion;
- apply only the NB correction at prediction, followed by the existing train-balance step;
- **no age interaction, no extra factors, no threshold search**.

This tests residual NB severity/total-cost transfer, rather than repeating the failed frequency block. Its uncertainty must remain visible under reserve sensitivities.

**Planning gain:** usually small; plausibly up to approximately **0.001–0.002** if O4’s count-to-cost assumption is materially wrong. Negative gains remain entirely plausible.

### 3. Motor age-band integration

Replace band-mean teacher evaluation with conditional age averaging:

- derive developed age distributions conditional on **age band × years-licensed buckets 0–2, 3–5, 6–10, 11–20, 21+**;
- restrict to ages consistent with the teacher’s licensing-age convention;
- shrink cell distributions toward the band distribution with **200-policy prior weight**, then enforce feasibility;
- use five equal-probability bins, evaluating the teacher at each bin’s conditional mean age;
- update all age-derived teacher inputs consistently;
- blend **50% integrated / 50% existing motor peril predictions**;
- rerun unchanged O4 adaptation and balancing.

No uniform ages, no inferred broker-specific within-band distribution, no tunable blend.

**Planning gain:** roughly **0–0.001**, potentially larger only if the current young-band substitution is seriously wrong.

### 4. One pooled-base blend—not a blend tournament

Finish the running jobs, but preregister only:

- emerging training weight **5**, not 20;
- **25% pooled / 75% existing base** for motor and buildings at the existing pre-gender/NB blend stage;
- line-normalize components on training data;
- retain O4’s contents treatment unchanged;
- retain every other O4 setting.

Require honest OOF pooled bases. Treat weight 20 as a diagnostic, not another contender after seeing results.

**Planning gain:** approximately **0–0.002**, with downside. This is the remaining queue’s best chance of useful model diversity, but the previous pooled failures substantially lower expectations.

### Kill or defer

- **Classifier shift weights inside local fits:** defer. They increase variance in precisely the small NB sample that matters and do not solve broker concept shift. First establish support and benefit from simple transport diagnostics.
- **Full per-peril frequency–severity family:** kill for this deadline. It introduces new count-development, severity-selection, tail and recovery-allocation decisions across sparse emerging perils. It is not one experiment.
- **P0/compat/pooled blend grid:** kill. Compat has already supplied negative evidence; another weight search mainly spends validation credibility.
- **Weather refinements:** keep dropped. Their estimated sub-1% line effects are below the principal unresolved uncertainties.

## 4. Is honest top five reachable?

**Possible, but not the central forecast from these results.** A defensible expectation for this restricted programme is closer to **0–0.002** than a reliable +0.0034. The candidate gains are not additive.

The credible routes are:

1. discover a substantial, independently evidenced motor extrapolation error;
2. obtain genuinely complementary pooled predictions;
3. uncover a concrete count-offset or age-mapping defect.

Another collection of weak segment factors is not a credible route.

The public gap is only **1.68 times** the quoted 0.002 score-difference noise. Crossing fifth publicly would therefore be much easier than establishing a genuine top-five improvement.

Freeze O4 as the fallback. Submit only a prespecified, validated alternative—not several motor scalars to discover the hidden book’s level. If the audit shows that 5–10% motor uncertainty buys only about 0.001 D², accept that the remaining gap requires better risk ranking, not increasingly elaborate level speculation.
