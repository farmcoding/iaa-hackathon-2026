# Adversarial Review — Round 2

You are about to waste your final 40 hours trying to force a 65k-row emerging dataset to re-learn rating relativities that your 3M-row developed base already knows, using a mathematically misspecified offset and a broken validation veto.

Stop running multi-factor local blocks. Below is the exact diagnosis of why tonight’s runs failed, the math of your 2025 level risk, what to kill, and the five pre-registered runs that can realistically get you the +0.0034 needed for Top 5.

---

## 1. What the Fold-R Failures Actually Mean (and Why Your Local Fits Failed)

Tonight’s pattern—every local experiment barely nudges Fold B ($+0.0010$ at best) and gets crushed on Fold R ($-0.0026$ to $-0.0153$)—is **not** mysterious. It is the result of two modelling bugs on your part and two structural asymmetries in Fold R.

### Bug 1: You are fitting Poisson regressions on *claim counts* using $\log(\text{predicted cost})$ as the offset
Look at what you ran tonight for the Motor count block, Property count block, and New-Business regime (and what is already inside O4’s NB factor):
$$\mathbb{E}[N_i] = \exp\left(\log(\hat{\mu}_i^{\text{O4}}) + \mathbf{x}_i^\top \boldsymbol{\beta}\right)$$
Since $\hat{\mu}_i^{\text{O4}} = \hat{N}_i \cdot \hat{S}_i$ is a **total cost** prediction, setting $\text{offset} = \log(\hat{\mu}_i^{\text{O4}})$ in a claim-count regression estimates (up to a line-normalizing constant $c$):
$$\exp(\mathbf{x}_i^\top \boldsymbol{\beta}) \propto \frac{\mathbb{E}[N_i]}{\hat{\mu}_i^{\text{O4}}} = \left(\frac{\mathbb{E}[N_i]}{\hat{N}_i}\right) \cdot \frac{1}{\hat{S}_i}$$
**You are dividing by predicted severity $\hat{S}_i$.** Every segment with high severity and low frequency (comprehensive vs. TPFT cover, flood zone 3, subsidence/trees, non-standard construction/fire, older BI-heavy age bands) has *fewer claims per dollar of predicted cost*, so your Poisson count model systematically slashes its rate. Every segment with low severity and high frequency (glass/hail, minor own-damage, young drivers with high compulsory/voluntary excess) has *more claims per dollar of predicted cost*, so your count model systematically inflates its rate. When you then multiply your **cost** prediction by $\exp(\mathbf{x}_i^\top \boldsymbol{\beta})$, you corrupt the severity structure of `P0`.

### Bug 2: Your count-block feature list is leaked across both folds
The 10 motor variables and 9 property variables you tested tonight are the **exact segments you listed in Section A.2** after inspecting where 2023 and 2024 A/E residuals had the same sign. Selecting ~45 factor levels by looking at 2023+2024 and then tuning $k \in \{50, 200, 800\}$ on those same two years is data snooping. That is why your 11-factor stack gained $+0.0068$ on folds and lost $-0.0029$ on the public board on 25 Oct. Tonight’s count blocks are just a ridge-regularized version of the same failed 11-factor stack.

### Why Fold R (Train 2024 $\rightarrow$ Score 2023) is a flawed veto
Even for valid mechanisms, Fold R fails for three structural reasons:
1. **Maturity & Recovery Asymmetry at 31 Dec 2024**: Both policy years are valued at 31 Dec 2024. That means **2023 claims are 12–24 months mature** (mostly settled, recoveries booked, small chain-ladder multipliers). **2024 claims are 0–12 months mature** (open case reserves, **zero recoveries booked on unsettled claims**, and large $1/\text{CDF}$ multipliers blowing up Q3/Q4 2024 claims). Fold R trains your local adaptation on noisy, green 2024 ultimates and tests on settled 2023 ultimates.
2. **The Buildings Regime Shock (585 in 2023 vs. 352 in 2024)**: Driven by 49 fire claims in 2023 ($2.3\times$ developed) vs. 21 in 2024 ($0.84\times$ developed) plus milder weather, trended buildings cost dropped **40%** in 2024. Fitting shape/temperature parameters on 2024 (`t = 1.26` on buildings) and predicting 2023 (`585` level) guarantees a Fold R crash.
3. **The $12\times$ Importance-Weight Multiplier on 2023’s Tiny NB Slice**: One-third of your fold metric reweights the validation year to match 2025 (**77% NB**, 50% young). In 2023 (Fold R’s test set), NB is only **6.4%** of exposure (~1,200 motor rows, ~300 buildings rows). To make 6.4% NB look like 77% NB, the classifier-odds weights multiply those ~1,200 motor rows and ~300 buildings rows by **$12\times$ to $20\times$** (driving ESS down to ~15%). Three random claims in 2023’s 6.4% NB slice swing Fold R’s weighted score by $\pm 0.004$.

### How to fix your validation right now (without letting fold noise in)
* **Stop using Fold R’s $12\times$-weighted score as a hard veto.**
* **Use Pooled Out-of-Fold (OOF) Level-Neutral $D^2$ for relativities**: Concatenate the level-neutral OOF predictions for 2023 (from Fold R) and 2024 (from Fold B) into a single 65k-row frame *before* computing $D^2$, alongside unweighted pooled OOF $D^2$ on the `new_business == 1` subset (~5,000 rows across both years) and `age_band in ('17-24', '25-34')` (~25,000 rows). This doubles the evaluation sample size and removes the 2023-vs-2024 buildings level swing from relativity evaluation.
* **Enforce a strict parameter budget instead of Fold R vetoes**: Never fit more than **2–3 local parameters per line** on emerging data, and only on features where the developed base is **schema-blind** (see Section 3).

---

## 2. The 2025 Level Risk: Exact Math and Why Motor’s +26% Jump is Suspect

### How much $D^2$ is a 5–10% (or 20%) level error worth?
For Tweedie power $p = 1.5$, the unit deviance is:
$$d(y, \mu) = 4\left(\frac{y}{\sqrt{\mu}} + \sqrt{\mu} - 2\sqrt{y}\right)$$
Suppose a line’s predictions $\hat{\mu}_i$ are calibrated so the Tweedie-optimal scalar is $c=1$, meaning $\sum_i y_i / \sqrt{\hat{\mu}_i} = \sum_i \sqrt{\hat{\mu}_i} \equiv S_{\text{line}}$. If your 2025 level on that line is off by a multiplicative factor $c = 1 + \delta$, total deviance increases by:
$$\Delta D(c) = 4 S_{\text{line}} \left(\frac{1}{\sqrt{1+\delta}} + \sqrt{1+\delta} - 2\right) \approx S_{\text{line}} \, \delta^2$$
Dividing by the pooled null deviance $D_{\text{null}} = 8 \sum_{\text{all}} (\sqrt{\bar{y}} - \sqrt{y_i}) \approx 100 N$ (calibrated directly against your Fold B `0.14199` vs. level-neutral Fold B `0.15356`, where the 2023$\rightarrow$2024 buildings/motor level shift cost **0.01157** in $D^2$):
$$\Delta D^2_{\text{line}}(\delta) \approx -\kappa_{\text{line}} \, \delta^2, \quad \text{where } \kappa_{\text{motor}} \approx 0.11, \;\; \kappa_{\text{buildings}} \approx 0.045, \;\; \kappa_{\text{all lines}} \approx 0.17$$

Plug in the numbers:
* **5% motor level error ($\delta = 0.05$)**: costs **$-0.00028$** $D^2$ (inside noise).
* **10% motor level error ($\delta = 0.10$)**: costs **$-0.00110$** $D^2$ (one-third of the gap to 5th place).
* **20% motor level error ($\delta = 0.20$)**: costs **$-0.00440$** $D^2$ (more than the entire $+0.0034$ needed for Top 5).
* **25% buildings level error ($\delta = 0.25$)**: O4 predicts **487** on 2025 buildings vs. **352** in 2024 (and 2025 hail weather is milder than both 2023 and 2024). If true 2025 buildings is near trended 2024 ($\sim 385$), that $+26\%$ overprediction costs **$-0.0030$** $D^2$ by itself!

### Why O4’s Motor 2025 Level (607 vs. 472/493, +26%) is Almost Certainly Inflated
Decompose the $+26\%$ jump ($607 / 482 = 1.259$):
1. **$\sim +6.5\%$ comes from your emerging NB count-offset factor** ($k=200$ on A/E $1.20 / 1.07$, applied as NB jumps from $8\%$ in train to $77\%$ in 2025). Because `P0` *already* includes `new_business`, `customer_tenure`, and `ncb_years`, and because you fitted the extra NB factor on **claim counts** with a $\log(\text{cost})$ offset, if NB claims in emerging are skewed toward smaller frequency perils (glass/own damage) rather than large BI claims, this $+6.5\%$ is partly phantom double-counting.
2. **$\sim +18.2\%$ comes from `P0`’s mix shift on the 2025 book.** Why does `P0` jump $+18.2\%$? Because in emerging data, `credit_band`, `annual_mileage`, and `telematics` are **100% missing**, and `driver_age` is **coarse-banded** (`17-24` mapped to a single number). In a tree ensemble trained on developed data where only 2.5% of rows are missing, feeding `credit = NaN, telematics = NaN, mileage = NaN` for a book that is 50% aged 17–34 and 77% NB routes policies down adverse missing-value splits.
3. **2025 is a Broker Portfolio Transfer, not adverse-selected retail shoppers**: Section A.2 shows that within 2025, `new_business` vs. `renewal` is separable with AUC 0.74 using `make_model`, `previous_insurer`, and `years_licensed`. Seasoned drivers transferred from a broker book (`new_business = 1`, but with established `years_licensed` and `ncb_years > 0`) do not carry the full adverse-selection loading of a direct retail shopper with 0 NCB.
4. **How to estimate the true 2025 mix level without touching the LB**: Check the 2025 motor and buildings mix index implied by **`base_compat`** (which was trained on developed data *without* credit, mileage, telematics and *with* banded ages) vs. **`P0`**, and check emerging 2023–24 NB A/E on **Tweedie/capped cost** instead of claim counts. If `base_compat` predicts a 2025 motor mix jump of $+11\%$ instead of `P0`’s $+18\%$, that $7\%$ gap is pure missing-feature tree distortion in `P0`.

---

## 3. Queue Audit: What to Kill vs. The 5 Pre-Registered Runs for Tonight

### KILL LIST (Drop these immediately)
1. **KILL: Covariate-shift weights inside O4’s local fits.** With ESS at 15–23% ($\sim 5\text{k}$ rows), weighting local GLMs by classifier odds multiplies variance by $5\times$ and has zero effect on correctly specified additive features.
2. **KILL: Any further multi-factor emerging count or cost blocks.** 65k rows cannot re-estimate 10 rating factors on top of a 3M-row GBM/EBM blend.
3. **KILL: Building a 30-model developed Frequency-Severity base family from scratch.** Refitting 15 perils $\times$ 2 models $\times$ 3 learners on 3M rows will eat 20+ hours of compute and engineering time. You can get **90% of the Frequency-Severity benefit in 1 second inside `fit_predict`** (see Run 2 below).
4. **CAUTION on the running `base_pooled` (sample weight 5 and 20)**: Passing `sample_weight = 20` to LightGBM avoids duplicate-row splits on `min_data_in_leaf`, **but** it multiplies the Tweedie hessian by $20\times$, allowing 15 emerging rows to pass a `min_sum_hessian_in_leaf` threshold meant for 300 rows. Do not use `base_pooled` standalone; only blend it at low weight ($0.15\text{--}0.25$) inside O4 if its level-neutral pooled OOF beats O4.

---

### THE 5 PRE-REGISTERED RUNS FOR TONIGHT (In Priority Order)

#### Run 1: Blend `base_compat` with `P0` inside O4 (`fit_predict`, ~2 seconds)
* **Why you missed this**: Look at Section A.4: back when `P0` scored `0.11554` on primary, a 50:50 blend of `P0 + base_compat` scored **`0.11792` ($+0.00238$ primary gain!)**—and you abandoned it before L1 and O4 were built! `base_compat` was trained on developed data *with banded ages and without credit/mileage/telematics*. On the 2025 test book (50% young, 77% NB), `base_compat` does not suffer from `P0`’s missing-telematics/credit split bias or age-band step distortion, and it provides genuine ensemble diversity.
* **Pre-registered settings**: Before per-peril credibility calibration inside O4, replace each peril’s base with $(1 - w_L)\,\text{P0}_p + w_L\,\text{base\_compat}_p$ (proportional per-peril scaling of `base_compat` to match `P0` peril shares if `base_compat` is line-level). Test 3 weight pairs $(w_{\text{motor}}, w_{\text{buildings}})$:
  1. `(0.25, 0.25)`
  2. `(0.40, 0.30)`
  3. `(0.50, 0.50)`
* **Expected 2025 LB gain**: **$+0.0015$ to $+0.0025$**.

#### Run 2: Instant Per-Peril Frequency $\times$ Severity Credibility Decomposition (`fit_predict`, ~1 second)
* **Mechanism**: Emerging motor has $1.4\text{--}1.8\times$ developed frequency and $0.5\text{--}0.7\times$ developed severity; buildings had 49 fire claims in 2023 vs. 21 in 2024. Currently, O4 calibrates each peril’s emerging level on **total cost** with $k=100$ claims. In Bühlmann-Straub credibility, claim count CV is $1/\sqrt{N}$, whereas heavy-tailed severity CV is $\sim 4/\sqrt{N}$, requiring **$10\times$ higher $k$ on severity than on frequency**. With $k=100$ on total cost, a single 156k BI claim or the 2023 fire spike drags the peril multiplier by 25% (causing the 585 vs. 352 buildings level swing and O4’s inflated 487 buildings level in 2025).
* **How to do it in 1 second without refitting bases**: Factor each peril’s emerging training ratio $\frac{\sum_i y_{i,p}}{\sum_i \hat{\mu}_{i,p}^{\text{base}}}$ into a **frequency factor** (shrunk toward the line’s frequency factor with $k_{\text{freq}}$) times a **severity factor** (emerging mean severity $\bar{S}_{p,\text{em}}$ shrunk toward the line’s severity ratio times developed peril severity $\bar{S}_{p,\text{dev}}$ with $k_{\text{sev}} \gg k_{\text{freq}}$, or equivalently cap individual peril claims at the 98th percentile before computing the peril relativity and restore the capped tail uniformly at the line level):
  $$M_p = \left(\frac{N_p + k_{\text{freq}} M_{\text{line}}^{\text{freq}}}{\hat{N}_p^{\text{line-norm}} + k_{\text{freq}}}\right) \times \left(\frac{N_p (\bar{S}_p / \bar{S}_{p,\text{dev}}) + k_{\text{sev}} M_{\text{line}}^{\text{sev}}}{N_p + k_{\text{sev}}}\right)$$
* **Pre-registered settings**: Test $(k_{\text{freq}}, k_{\text{sev}})$ at:
  1. `(50, 400)`
  2. `(75, 600)`
  3. `(100, 1000)` (or 98th-percentile peril claim cap with $k=100$).
* **Expected 2025 LB gain**: **$+0.0010$ to $+0.0020$** (especially by fixing the 2025 Buildings level).

#### Run 3: Conditional Age Teacher Averaging Given `years_licensed` and `ncb_years` (Harness scoring pass, ~15–20 min on Mac/Kaggle)
* **Mechanism**: Simply averaging `P0` over unconditional in-band ages misses the strongest kinematic constraint in your schema: **exact `years_licensed` and `ncb_years` are present in the emerging market!** A driver in the `17-24` age band with `years_licensed = 0` is 17–18 years old (extreme risk). A driver in `17-24` with `years_licensed = 6` **must be 23–24 years old** ($\text{age} \ge 17 + \text{years\_licensed}$). Feeding `age = 21` to `P0` for both of them underprices the 17-year-old novice and overprices the 23-year-old experienced driver across 50% of the 2025 book.
* **Pre-registered settings**: From the developed dataset, compute the empirical 20th, 50th, and 80th percentiles of exact `driver_age` within each `(driver_age_band, min(years_licensed, 8), min(ncb_years, 5))` cell (and `policyholder_age` given `policyholder_age_band` and `years_at_address`). Score `P0` on the emerging rows at those three conditional age quantiles $(q_{20}, q_{50}, q_{80})$ and replace `P0` with $\frac{1}{3}(\text{P0}_{q20} + \text{P0}_{q50} + \text{P0}_{q80})$. Test inside O4 with blend weights `1.0` (full replacement), `0.75`, and `0.50` with `base_compat`.
* **Expected 2025 LB gain**: **$+0.0010$ to $+0.0018$**.

#### Run 4: Fix O4’s NB Factor — Cost A/E Instead of Count A/E + Broker Transfer Conditioning (`fit_predict`, ~2 seconds)
* **Mechanism**: Stop applying a Poisson *claim-count* multiplier (fitted with a $\log(\text{cost})$ offset) uniformly to 160,000 broker-transferred policies in 2025.
* **Pre-registered settings**:
  1. Fit the emerging NB credibility factor on **99%-capped Tweedie cost A/E** ($\sum y_{i,\text{capped}} / \sum \hat{\mu}_i^{\text{base}}$) instead of claim counts, with $k = 300$ claim-equivalents.
  2. Split `new_business == 1` into **unseasoned NB** (`new_business == 1` & `ncb_years == 0` / `years_licensed <= 2`) vs. **seasoned/transferred NB** (`new_business == 1` & `ncb_years > 0`), with $k = 200$.
  3. Apply Tweedie-optimal line scaling $c_{\text{line}} = \left(\frac{\sum y_i / \sqrt{\hat{\mu}_i}}{\sum \sqrt{\hat{\mu}_i}}\right)^2$ on train instead of $\sum \hat{\mu}_i = \sum y_i$ once the 2025 NB multiplier is fixed.
* **Expected 2025 LB gain**: **$+0.0008$ to $+0.0018$**.

#### Run 5: Schema-Blind-Only Local GLM (`gender x young_driver`, `construction_type`, `previous_insurer`) (`fit_predict`, ~2 seconds)
* **Mechanism**: Why did `gender` give you $+0.0074$ on the public LB while every other local factor failed? Because **Developed did not have `gender`**, so `P0` was 100% blind to it. Stop fitting local factors on variables `P0` already knows (`cover`, `fuel`, `elevation`, `licence_points`). Restrict local adaptation *exclusively* to features where `P0` sees `NaN` or unmapped codes:
  1. **Motor**: Add the single classic actuarial interaction `gender * I(driver_age_band == '17-24')` (1 degree of freedom, ridge log-link Tweedie $p=1.5$ offset GLM).
  2. **Buildings**: Empirical-Bayes cost-A/E factor ($k=150$ claim-equivalents, 98%-capped cost) on `construction_type` (which uses different codes in emerging, explaining the $1.39\text{--}1.46$ A/E on non-standard construction).
  3. **Motor**: Empirical-Bayes cost-A/E factor ($k=300$ claim-equivalents, 99%-capped cost) on `previous_insurer` (which has $0.47\text{--}1.69$ A/E in both folds and is the primary identifier of the 2025 broker sub-portfolios).
* **Expected 2025 LB gain**: **$+0.0008$ to $+0.0015$**.

---

## 4. Can You Reach Top 5 (+0.0034 to 0.16662) Honestly in 40 Hours?

**Yes—Top 5 (`0.1666+`) is realistically reachable; 1st place (`0.17303`, $+0.0098$) is not.**

Here is the realistic accounting of where you stand:
* Every big jump in your history came from fixing a **schema or transfer gap** between Developed and Emerging (`P0` per-peril transfer $\rightarrow$ `0.15195`; adding schema-blind `gender` $\rightarrow$ `0.15933`; peril credibility + contents pooled + NB level $\rightarrow$ `0.16326`).
* You still have three unexploited schema/transfer gaps sitting right in front of you:
  1. `P0` is distorting the 50%-young, 77%-NB 2025 book because of 100%-missing `credit/mileage/telematics` and unconditional band-mean age mapping. Blending **`base_compat`** (Run 1) + **Conditional Age Teacher Averaging using `years_licensed`/`ncb_years`** (Run 3) directly fixes this and is worth **$+0.0020$ to $+0.0035$** combined.
  2. Your emerging adaptation layer is currently conflating count residuals with cost residuals (Bug 1) and using total-cost credibility on heavy-tailed perils (causing the 585 vs. 352 buildings swing). **Freq $\times$ Sev peril credibility** (Run 2) + **Cost-based NB calibration** (Run 4) is worth **$+0.0012$ to $+0.0025$**.
  3. Filling the remaining schema-blind `NaN`s (`gender x 17-24`, unmapped `construction_type`, `previous_insurer` on capped cost, Run 5) is worth **$+0.0008$ to $+0.0015$**.

Accounting for overlap, executing Runs 1 through 5 gives an expected honest gain of **$+0.0035$ to $+0.0055$**, landing you at **`0.1668–0.1688` (3rd to 5th place)**. Better still, Remember that **60% of the final competition score is judged by the panel** (25% effective use of AI, 20% validation/governance, 15% report/video, plus the panel’s view of adaptation). Documenting how you caught the Poisson count-on-cost-offset bias, separated frequency vs. severity credibility constants ($k_{\text{freq}} \ll k_{\text{sev}}$), and used kinematic `years_licensed` constraints to de-bias coarse age bands is textbook actuarial science that will score near-maximum marks with the judging panel.
