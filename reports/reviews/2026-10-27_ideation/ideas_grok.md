The gap to 5th is about +0.0034 public and to 1st about +0.0098. Those are not additive across ideas, and they will not show up one-for-one in your primary. The binding problem is that the thing you are selecting on is not the 2025 book.

## 1. Constraints, by power left on the table

Ranks are expected public/private D² left uncaptured, not primary. They overlap.

**1. Fold design, not the 0.002 threshold. About 0.004–0.008.**
Fold R trains on 2024 and scores 2023. It is not a second read of the same mechanism; it is a veto on anything that differs by year. Fire (2.3× then 0.84×) makes that veto harsh. Primary is two-thirds the historical book and one-third a 2025-reweight whose effective sample is 15–23%. Adversarial AUC 0.99 says the reweight is standing on weather year and tenure, so the weights are extreme and the slice is noise. Age 17–34 is 39% of history and 50% of 2025; new business is a minority of history and 76% of 2025. A real shift factor can be worth +0.004 on the broker book and +0.001 on primary, and your keep rule will drop it. Public noise sd ≈ 0.002 means a general loosening of the 0.002 bar would also start accepting noise. Relax the design, not the bar: forward fold B plus a propensity reweight that is forbidden to use weather and raw tenure; fold R only as a veto on year dummies; a separate, pre-registered path for structural 2025 level effects that folds cannot identify. Cap that path at two submissions.

**2. Balance-on-train plus a base that cannot see the 2025 level shift. About 0.002–0.006.**
The harness deletes any constant. Level reaches 2025 only through mix × row-level relativities. Hail 11.4 versus 14–17 is mostly a year constant, so it is almost unidentified in emerging data, and a post-hoc test scalar is the wrong place to put it. A hand-set coefficient on the row’s own weather survives balance: it is constant on train (removed) and lower on 2025 (kept). Your hail-days trial failed for this reason, not because 2025 weather does not matter. Confirm the on-level date in the same pass. If training actuals are not on-leveled through 2025, one extra year of the 3/5/9% trend is a pure level miss and must sit on row features or after balance, not as a train-fit scalar. The Tweedie-MLE scalar being noise does not test this.

**3. Developed base pushed through band-mean age, with developed severity shape. About 0.003–0.006.**
Motor deciles are not monotone (fold B: decile 1 at 0.44, deciles 4–5 at 1.34–1.37, decile 8 at 0.73). That is a transferred ranking, not a level problem. In-band mean age assumes the developed age curve and a flat within-band mix. The 2025 book is younger, so both the ranking and the mix-driven level are wrong if emerging young-driver frequency is steeper than that curve. BI severity 0.45–0.62× developed, largest emerging BI 156k versus 11.9m, means the developed BI shape is the wrong severity even after a mean peril factor. Schema-compatible and pooled refits already lost; do not spend the 48 hours re-litigating a full refit until a temperature check says the residual is still in the base.

**4. Adaptation stopped at a few scalars after the tree failure, and one-ways were fit on cost. About 0.002–0.005.**
Free LightGBM on the offset overfit, and eleven stacked cost factors went +0.0068 on folds and −0.0029 on the board. The wrong lesson is “only gender and a new-business count factor.” Gender worked because it was one gated, low-cardinality factor. Cost one-ways are large-claim noise, especially on buildings (~500 claims). The same factors on frequency, one at a time, with a complement of credibility, were not what failed. A joint 6–8 factor GLM diluted gender; a ridge Tweedie that is required to keep gender is a different model.

**5. Property treated as too noisy to model. About 0.001–0.003 pooled, more if flood/storm dominate deviance.**
Non-standard construction, energy rating E versus A, trees, and flood zone 3 miss in the same direction on both folds. At p=1.5 a few large property claims can outweigh a lot of motor frequency. This needs a frequency fit and a large penalty, not a tree.

**6. One developed-frame process, Kaggle unused. About 0.001–0.004, low confidence.**
RAM is blocking a proper sample-weighted refit. It is not blocking the ideas above. Row duplication at weight 5 already told you the failure mode (memorisation). One Kaggle CPU refit with `sample_weight`, after the emerging tests, is enough. GPU does not help this table.

**7. Borrowed development pattern. About 0.000–0.002.**
Identical valuation dates make a borrowed report-delay pattern reasonable. Case adequacy ±10% moves motor level 3.4%, and balance removes average bias. What it does not remove is differential bias by BI share and start month. Blend patterns; do not rebuild the programme around chain ladder on two emerging years.

**8. Keep-rule threshold and admissibility exclusions. Near zero net, leave them.**
The threshold is doing useful private-split protection. Ten submissions a day for three days can overfit a sd-0.002 board by several thousandths; the max of ~30 noise draws is on the order of +0.005 public that will not hold. Exclusions were re-tested on 25 Oct. Do not put premium, campaign, branch, or customer-level history back in. `hist_*_policy` is the admissible claim history; it is zero for most of the test book, so do not expect it to move 2025.

Safe relaxations, in order: fix the propensity features and the role of fold R; allow two pre-registered structural submissions that folds cannot identify; fit residual factors on frequency; one sample-weighted Kaggle refit if temperature shrink shows the base is still mis-ranked after age and new-business corrections.

## 2. White space

Already used, and not the gap: gradient boosting blends, EBM, credibility target encoding, monotone constraints, Tweedie, chain ladder, a one-factor offset GLM, fixed-k credibility, train balance. Neural tabular, spline-GLM blends, isotonic, and free residual trees were tried and lost. Do not rerun them.

Fits this problem:

- **Offset Tweedie GLM / GAM (p=1.5).** Optimises the metric, low variance, right sample size. The adaptation layer should be this, not another booster.
- **Frequency–severity, frequency first.** Emerging/developed ratios differ by peril (frequency up, severity down, BI severity down more). A cost multiplier blurs that. Severity should be a peril-level credibility number; frequency should carry the rating factors.
- **Bühlmann / empirical Bayes toward the developed relativity.** You use fixed k. The missing piece is the complement: developed relativities for age, make, occupation, not the grand mean, and credibility on frequency.
- **Structural domain adjustment, not generic domain adaptation.** AUC 0.99 importance weights will explode. Do not fit a domain-adversarial net. Reweight only on overlapping, transferable covariates, and impose weather from outside the two emerging years.
- **Temperature shrink of relativities.** Part 1 isotonic failed because it is too flexible. One power parameter is the version that matches non-monotone, over-spread deciles.
- **Two-regime new-business model with partial pooling.** A count scalar cannot fix different slopes on a book that is 76% new business.

Poor fit here: TabM, RealMLP, AutoML, bandit search over architectures, distillation stacks, multi-task neural nets, full Bayesian MCMC. Hierarchical Bayes is the right prior story but Bühlmann is the version you can finish. Point the existing agent at a pre-registered factor list; do not let it search.

## 3. Ideas

Ranked by expected effect on the 2025 book, not on primary. Overlapping ideas do not add. A realistic package that works (temperature + age + weather + new business, GLM only if the single factors survive) is about **+0.005 to +0.012 public** if the mechanisms are real, and about **+0.002** if only the shrink is real. That brackets 5th through 1st; it does not justify thirty submissions.

Work order: temperature and the on-level check first (they tell you whether the base ranking is the problem), then age, weather, previous insurer alone, then new business, then one ridge GLM that nests gender, property last, Kaggle refit only if deciles stay broken.

1. **Age-band frequency relativities.** Drop in-band mean age as the emerging age signal. Estimate motor claim-frequency factors for the five bands, complement = the developed age curve at band midpoint, k so that a band needs on the order of 100 claims for half credibility. Pre-register monotone 17–24 > 25–34 > 35–49. Optional `named_young_driver` × young band if counts support it. This is the mix shift validation under-weights: 11 pp more young exposure, and young bands are exactly where a developed curve fed a band mean is flattest relative to a real book. Primary might move only +0.001 to +0.003; 2025 more, because the mix shift is larger on the test book. `fit_predict`, Mac, ~3 h. Overfit risk low if the monotone gate is real. Test: same sign both folds, 2025-reweighted gain at least as large as natural-book gain, bootstrap 10th percentile of the gain > 0, reject if non-monotone.

2. **Weather as a row-level structural coefficient.** Do not refit hail elasticity on 2023 vs 2024. Take elasticities from the developed book, or a pre-registered range (glass/hail vs hail days 0.4–0.8; storm vs wind and hail; flood vs rainfall × flood zone). Apply inside `fit_predict` as a function of the row’s weather so train balance removes the train constant and 2025’s milder year falls out. Primary ≈ 0 is not a failure. Expected 2025 gain +0.002 to +0.006 if weather perils are a real cost share, else near zero — check the peril cost shares before spending the submission. Mac, ~4 h, no full refit. Overfit risk low only if you do not estimate the elasticity on the emerging year contrast. Test: forward fold not down more than 0.002 at either end of the range; submit the midpoint once; this is one of the two structural submissions, not a keep-rule candidate.

3. **Relativity temperature.** `mu * (p/mu)^α` by line, α in {0.6, 0.75, 0.9}, inside `fit_predict`, harness rebalance left as it is. Decile A/E from 0.44 to 1.37 is over-spread score, and p=1.5 punishes that even when rank order is partly wrong. One parameter. Expected +0.002 to +0.004 on primary and a similar amount on 2025 if the developed ranking is over-dispersed in general; ~0 if the wiggle is pure segment bias that age and previous insurer then absorb. Mac, ~1.5 h. Overfit risk low. Test: choose α on fold B, freeze it, require fold R not down more than 0.003. If it fails that, the decile chart is segment bias, not spread, and you should not ship a shrink.

4. **New-business regime, not a tighter k.** Keep the renewal path on the current model. On new business only, a frequency residual with a hard shrink toward zero: age band, cover, fuel, region, vehicle age, plus the existing line scalar. Buildings scalar stays heavily shrunk (A/E 1.06 then 0.69). Contents scalar: allow only a modest move from the current factor; 1.18 then 1.43 will overfit a free level. This transfers because the test book is 76% new business, not because fold R likes it. Expected +0.002 to +0.005 on 2025, less on primary. `fit_predict`, Mac, ~4 h. Overfit risk moderate on the small new-business claim set. Test: fit on one year, confirm sign on the other; report new-business claim counts; veto any line whose new-business factor flips sign.

5. **Previous insurer, alone, on frequency.** Largest segment miss (0.47–1.69) and a plausible broker-book selection factor. It was probably buried in the eleven-factor stack that the board rejected. One Bühlmann frequency factor, complement = line frequency, rare levels collapsed, unknown gets no load. Expected +0.001 to +0.004 if those codes exist in the 2025 portfolio; ~0 if the takeover arrives under a new code or “unknown”. `fit_predict`, Mac, ~2 h. Overfit risk moderate. Test: same direction on both folds for levels with at least ~50 claims; claim-count likelihood-ratio gate, as with gender; ship only alone, never as part of a stack of leftovers.

6. **Ridge offset Tweedie, nesting gender.** p=1.5, offset = log(current prediction), not a new booster. Motor: age band, new business, fuel, cover, region, missed payments, and the existing gender term. Buildings: construction, energy rating, flood zone, trees, with a large ridge. Alpha in {0.1, 1, 10}, picked on fold B. This is the joint model you already tried, with three constraints you did not have: frequency-scale or Tweedie with an offset, gender nested so it cannot be diluted, coefficients required to match pre-registered signs. Expected +0.002 to +0.005 if several A/E gaps are real, and it replaces ideas 1, 4 and 5 rather than stacking on top of them. `fit_predict`, Mac, ~5 h. Overfit risk moderate. Test: full keep rule against O4; drop any coefficient that flips sign across folds; if the GLM does not beat gender-only plus age-only, ship the smaller model.

7. **Peril frequency × credibility severity.** Split each motor `base_<peril>` residual into a frequency model (Poisson or Tweedie p=1, offset log base) and a severity factor that is only a credibility blend toward emerging peril severity. Glass can stay frequent, BI severity stays low, and a younger, less hail-exposed 2025 book no longer inherits a single cost multiplier. Expected +0.002 to +0.004, overlapping the peril levels you already apply; the gain is from mix, not from a new overall motor level. `fit_predict`, Mac, ~5 h. Severity overfit is the risk; group perils (BI, damage, glass/hail, theft, third party) and keep severity k large. Test: keep rule on recombined cost; severity credibility fit inside the fold, not on both years at once.

8. **Property frequency GAM, shared flood.** Buildings and contents, offset log base, ridge or a heavily penalised smooth: flood zone, elevation band, trees, construction, energy rating, distance to river. One shared flood coefficient across the two lines. Fit on claim counts. Expected +0.001 to +0.003 pooled. `fit_predict`, Mac, ~4 h. Overfit risk high if you fit cost — one flood creates energy-rating E. Test: both folds same sign on frequency A/E; keep rule on cost after the factor is locked; if a factor exists only because of one claim, drop it.

9. **2025 weights that ignore weather and tenure.** Propensity toward the 2025 mix using age band, new business, collapsed make/model, previous insurer, region, vehicle value group. Cap weights at 5. Use the weights to fit ideas 4–6, not as a new leaderboard model by themselves. Expected +0.001 to +0.003 on top of an unweighted GLM, and a more honest primary. Mac, ~3 h. Overfit risk is extreme weights. Test: effective sample above ~30% of emerging rows; if you cannot get there without weather or tenure, do not weight; the gain has to appear on the reweighted forward slice.

10. **Gender × age band, prior from developed.** Extend the existing motor gender GLM. The interaction is worth more on a book that is 50% under 35 than the folds will show. Shrink the young-male coefficient toward the developed relativity rather than freely estimating it. Expected +0.0005 to +0.002 on 2025. `fit_predict`, Mac, ~2 h. Low overfit risk. Test: likelihood-ratio gate on claim count; sign must match the developed prior; if the gate fails, keep the main effect.

11. **Blended emerging development, early or not at all.** Credibility-blend emerging report-delay with the borrowed developed pattern, by peril group. Average bias will be removed by balance; ship it only if BI-heavy and winter-inception segments move. Expected +0.000 to +0.002. Emerging claims only, Mac, ~6 h. Risk is a noisy chain ladder masquerading as a better target — the blend weight has to be claim-count credibility, not a free parameter. Test: rebuild targets, rerun O4, require primary and level-neutral fold B not down. If the first sensitivity is flat, stop.

12. **One sample-weighted base, only if deciles survive ideas 1–3.** New base: developed + emerging, `sample_weight` 3 or 5, never row copies; high `min_child_weight`; age as a band category; missingness indicators for credit, mileage, telematics; P0 as a feature or offset so the trees learn a residual. Kaggle CPU, one process, ~8 h including setup. Expected +0.001 to +0.004, and zero if temperature plus age already flattens the deciles. Overfit risk high if emerging weight is large. Test: two weights maximum, early stopping not on the scoring fold, then the unchanged adaptation must clear the full keep rule. No GPU.

13. **Make/model residual encoding.** Frequency target encoding with complement = developed make relativity, k on the order of 200 claims, rare models collapsed to body type × value group. Adversarial separation of new business on make/model means the broker mix may differ. Expected +0.0005 to +0.002. `fit_predict`, Mac, ~3 h. High cardinality is the risk; the prior and the collapse are the control. Test: encoding inside the fold; full keep rule; otherwise drop. Do not let the agent search make/model trees.

Do not resubmit cover, occupation, contents tenure, or gender-by-peril; those already failed a gate. Do not stack discarded runs. Missed payments, diesel, licence points and E01 belong inside idea 6 or not at all — they are real and small, and several will be weak on new business if payment history is missing.

## 4. Weakest spot

The transferred ranking is still in the prediction, and the validation is built so you cannot see the book you are pricing.

Decile A/E is not a calibration polish item. Decile 1 at 0.44 and the middle deciles at 1.35 means developed order, after schema mapping, is over-spread and partly mis-ordered on this market. Everything since P0 has been a level and a couple of scalars on top of that score. Balance-on-train then faithfully applies those wrong relativities to a younger, 76% new-business book, which moves the 2025 level the wrong way. Fold R and a weather-driven reweight will not reward the three corrections that match the actual shift: emerging age-band frequency, a new-business regime, and a weather coefficient the emerging years cannot identify. That is where a top-5 result has to come from. A new architecture will not do it at ~4,500 claims a year, and shopping the public board will give it back on the private split.

```json
[
  {
    "name": "age_band_frequency_relativities",
    "mechanism": "Replace band-mean age with Bühlmann claim-frequency factors by age band, complement = developed age curve, monotone 17-24 > 25-34 > 35-49, applied to motor cost.",
    "expected_gain": "+0.003 to +0.006 on 2025 D2; primary often only +0.001 to +0.003",
    "harness_fit": "fit_predict on fixed bases",
    "compute": "light",
    "where": "Mac",
    "hours": 3,
    "overfit_risk": "low if monotone and credibility k are pre-registered",
    "test": "Same sign both folds; 2025-reweighted gain >= natural gain; bootstrap 10th percentile > 0; reject if non-monotone"
  },
  {
    "name": "weather_structural_row_factor",
    "mechanism": "Hand-set peril elasticities on row-level weather (hail/glass 0.4-0.8, storm on wind/hail, flood on rainfall x flood zone), taken from developed data or a prior, not from emerging 2023 vs 2024. Survives train balance because 2025 weather differs.",
    "expected_gain": "+0.002 to +0.006 on 2025 if weather perils are a material cost share; primary about 0",
    "harness_fit": "fit_predict, must be a function of row weather not a post-hoc test scalar",
    "compute": "light",
    "where": "Mac",
    "hours": 4,
    "overfit_risk": "low only if elasticity is not fit on the two emerging years",
    "test": "Pre-register range; forward fold not down > 0.002 at either end; submit midpoint once as a structural submission"
  },
  {
    "name": "relativity_temperature",
    "mechanism": "One-parameter power shrink mu*(p/mu)^alpha by line, alpha in {0.6, 0.75, 0.9}, to correct over-spread non-monotone deciles.",
    "expected_gain": "+0.002 to +0.004 if spread is structural; ~0 if deciles are pure segment bias",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 1.5,
    "overfit_risk": "low",
    "test": "Pick alpha on fold B, freeze, require fold R not down > 0.003"
  },
  {
    "name": "nb_regime_shrinkage",
    "mechanism": "Renewal path unchanged; new business gets a hard-shrunk frequency residual (age band, cover, fuel, region, vehicle age) plus a line scalar. Buildings and contents scalars heavily constrained.",
    "expected_gain": "+0.002 to +0.005 on 2025; less on primary",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 4,
    "overfit_risk": "moderate on the small new-business claim subset",
    "test": "Fit one year, confirm sign on the other; report NB claim counts; veto a line whose NB factor flips sign"
  },
  {
    "name": "previous_insurer_frequency_credibility",
    "mechanism": "Single Bühlmann frequency factor for previous_insurer, rare levels collapsed, complement = line frequency. Not stacked with other discarded factors.",
    "expected_gain": "+0.001 to +0.004 if codes overlap the broker book; ~0 if the takeover is a new or unknown code",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 2,
    "overfit_risk": "moderate",
    "test": "Both folds, levels with >=50 claims only; claim-count LR gate; ship alone or not at all"
  },
  {
    "name": "ridge_offset_tweedie_glm",
    "mechanism": "Tweedie p=1.5 GLM, offset log(current prediction), gender nested. Motor: age band, NB, fuel, cover, region, missed payments. Buildings: construction, energy, flood zone, trees. Ridge alpha in {0.1, 1, 10}.",
    "expected_gain": "+0.002 to +0.005, replaces the single factors rather than adding to them",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 5,
    "overfit_risk": "moderate",
    "test": "Full keep rule vs O4; drop sign-flipping coefficients; ship the smaller model if the joint GLM does not beat gender-only plus age-only"
  },
  {
    "name": "freq_sev_peril_split",
    "mechanism": "Per motor peril group, frequency model with offset log(base_peril) and severity as a high-k credibility blend to emerging severity. Recombine freq x sev.",
    "expected_gain": "+0.002 to +0.004 from mix, overlapping existing peril levels",
    "harness_fit": "fit_predict using base_<peril>",
    "compute": "light",
    "where": "Mac",
    "hours": 5,
    "overfit_risk": "moderate on severity, low on frequency if perils are grouped",
    "test": "Keep rule on recombined cost; severity credibility fit inside each fold"
  },
  {
    "name": "property_frequency_gam",
    "mechanism": "Ridge/GAM on buildings and contents frequency, offset log base, shared flood coefficient: flood zone, elevation, trees, construction, energy rating, distance to river.",
    "expected_gain": "+0.001 to +0.003 pooled",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 4,
    "overfit_risk": "high if fit on cost; low-moderate on frequency with a large penalty",
    "test": "Same sign both folds on frequency; drop any factor driven by one claim; then keep rule on cost"
  },
  {
    "name": "shift_weights_excluding_weather",
    "mechanism": "Reweight emerging rows to the 2025 mix using age band, NB, collapsed make/model, previous insurer, region, vehicle value. Weather and raw tenure forbidden. Weight cap 5. Use to fit the GLM/NB factors.",
    "expected_gain": "+0.001 to +0.003 beyond the unweighted version of the same GLM",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 3,
    "overfit_risk": "moderate if weights are extreme",
    "test": "ESS >= ~30%; otherwise do not weight; gain must show on the reweighted forward slice"
  },
  {
    "name": "gender_x_ageband",
    "mechanism": "Extend the existing motor gender offset GLM with gender x age band, coefficient shrunk toward the developed young-male relativity.",
    "expected_gain": "+0.0005 to +0.002 on 2025",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 2,
    "overfit_risk": "low",
    "test": "Claim-count LR gate; sign must match developed prior; else keep main effect only"
  },
  {
    "name": "blended_development_target",
    "mechanism": "Credibility-blend emerging report-delay patterns with developed patterns by peril group; rerun O4 on the new ultimates. Do this early or not at all.",
    "expected_gain": "+0.000 to +0.002",
    "harness_fit": "target change in the harness, not a developed-frame refit",
    "compute": "moderate, emerging claims only",
    "where": "Mac",
    "hours": 6,
    "overfit_risk": "moderate if the emerging chain ladder is given too much weight",
    "test": "O4 primary and level-neutral fold B not down; stop if the first sensitivity is flat"
  },
  {
    "name": "sample_weight_pooled_refit",
    "mechanism": "One new base: developed+emerging trees with sample_weight 3 or 5, not row duplication; age as bands; missingness indicators; P0 as offset/feature; high min_child_weight. Only if deciles remain broken after age, NB and temperature.",
    "expected_gain": "+0.001 to +0.004, or ~0 if earlier ideas already fix the ranking",
    "harness_fit": "new base family, then unchanged O4 adaptation",
    "compute": "heavy, one process",
    "where": "Kaggle CPU",
    "hours": 8,
    "overfit_risk": "high if emerging weight or tree depth is large",
    "test": "Two weights max; early stopping off the scoring fold; full keep rule"
  },
  {
    "name": "make_model_shrunk_encoding",
    "mechanism": "Frequency target encoding of make_model with complement = developed make relativity, k ~ 200 claims, rare models collapsed to body type x value group.",
    "expected_gain": "+0.0005 to +0.002",
    "harness_fit": "fit_predict",
    "compute": "light",
    "where": "Mac",
    "hours": 3,
    "overfit_risk": "high cardinality, controlled only by the prior and collapse",
    "test": "Encoding fit inside the fold; full keep rule or drop"
  }
]
```
