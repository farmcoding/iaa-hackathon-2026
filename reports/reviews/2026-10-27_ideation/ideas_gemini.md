### 1. Ranking the Constraints by Cost in Predictive Power ($D^2$)

To move from **0.16326 (9th)** into the **Top 5 (> 0.16662, +0.0034)** or contend for **1st (0.17303, +0.0098)**, we first need to see how your current rules interact with the structure of the data. Note that from `P0` to `O4`, a **+0.00663** gain on your Primary metric produced **+0.01131** on the Public Leaderboard (~1.7x amplification on the 2025 test book).

Here are your current constraints ranked by how much 2025 test $D^2$ each is likely costing:

#### Rank 1: The "Single-Mechanism > +0.0020 Pooled $D^2$" Keep Rule & Ban on Combining Sub-Margin Signals (Validation / Loop Governance)
*   **Estimated Cost**: **0.0030 – 0.0055 $D^2$**
*   **Reasoning**: The competition metric is pooled across three lines: Motor (~75% of deviance weight), Buildings (~15%), and Contents (~10%). Consider the arithmetic of a pooled metric:
    *   If you find a genuine structural improvement on Buildings that lifts Buildings $D^2$ by a massive **+0.0100** (1 full percentage point), it moves the **pooled Primary score by only +0.0015**—and gets discarded by your keep rule.
    *   If you find a strong single rating factor on Motor (e.g., `previous_insurer`, `region`, `elevation_m`, `missed_payments_12m`), a typical true single-factor lift in personal lines pricing is **+0.0005 to +0.0012** pooled $D^2$. Tested in isolation against a +0.0020 hurdle, **100% of legitimate single-feature and single-property-line improvements fail**.
    *   Why did your earlier attempt at stacking 11 sub-margin factors fail on the board (-0.0029)? Not because sub-0.0020 signals are noise, but because: (1) they were estimated marginally/sequentially rather than jointly with strong shrinkage, (2) they were estimated on high-variance *ultimate cost* rather than low-variance *claim counts*, and (3) selecting factors post-hoc that looked good on Folds B and R accumulated multiple-testing selection bias.
*   **How to Relax Safely**:
    1. Keep the +0.0020 pooled hurdle for accepting a **new deployment candidate**, *but* allow a **single pre-registered, jointly heavily regularized multi-factor block** (e.g., a high-penalty Ridge Poisson GLM on claim counts across the known domain-shift features) to be tested as **one unified mechanism**.
    2. For line-specific models (Buildings or Contents), evaluate the keep rule on **within-line $D^2$** (requiring > +0.0035 within-line $D^2$ and paired bootstrap 10th percentile > 0 on that line) before pooling.

#### Rank 2: Fixing `base_<peril>` as a Rigid Unit-Slope Offset ($\beta = 1.0$) with Naive Missing/Banded Schema Mapping (Modelling)
*   **Estimated Cost**: **0.0025 – 0.0045 $D^2$**
*   **Reasoning**: Look at your Motor decile diagnostic in Section 2: across *both* folds, predicted-rate deciles are severely non-monotone in A/E (**Decile 1 = 0.44**, **Deciles 4–5 = 1.34–1.37**, **Deciles 8–9 = 0.73–0.78**). You are overpredicting the safest decile by >2x, underpredicting the middle of the book by 35%, and overpredicting the top deciles by 30%. Three structural causes drive this:
    1. **Missing-branch routing in `base_<peril>`**: In the Developed market, `credit_band`, `annual_mileage`, and `telematics` are present on ~97.5% of rows, and missingness is ~2.5% on other fields. When you push Emerging rows through `base_<peril>` with 100% `NaN` for credit/mileage/telematics and 12% `NaN` elsewhere, every tree in LightGBM/CatBoost/EBM routes every policy down the default `NaN` branches learned from the 2.5% missingness in Developed data.
    2. **Flat in-band mean age imputation**: Pushing all `17-24` drivers into `base_<peril>` as age `20.5` ignores the fact that `years_licensed` and `ncb_years` are exact integers that identify whether a `17-24` driver is actually 17 (`years_licensed = 0`) or 24 (`years_licensed = 7`). Worse, feeding `age = 20.5` with `years_licensed = 6` creates an impossible combination (licensed at age 14.5) in the Developed trees.
    3. **Peril severity compression**: Emerging BI severity is 0.45–0.62x Developed (max claim 156k vs 11.9m) while frequency is 1.2–1.8x higher. This flattens and curves the true log-cost relationship relative to $\log(\text{base})$.
*   **How to Relax Safely**: Allow a 2- to 3-parameter parametric recalibration curve on $\log(\text{base})$ inside `fit_predict` (Idea 1), and replace flat in-band age means and raw `NaN`s with a deterministic Developed-data feature bridge when scoring `base_<peril>` (Idea 3).

#### Rank 3: Fitting Emerging Residual Adaptations on High-Variance Ultimate Cost Instead of Claim Counts (Modelling)
*   **Estimated Cost**: **0.0020 – 0.0035 $D^2$**
*   **Reasoning**: A single emerging training fold has ~32k rows (~1,750 motor, ~250 buildings, ~250 contents claims). For a compound Poisson-Gamma process where claim severity has $\text{CV}_{\text{sev}} \approx 3$, the variance of row cost residuals is $1 + \text{CV}_{\text{sev}}^2 \approx 10\times$ higher than the variance of claim-count residuals. When you fitted joint Tweedie cost GLMs with 6–8 factors or boosted trees on cost, random 50k–150k BI and fire claims corrupted the factor weights ("extra factors diluted gender"). Notice that the **only** two segment adjustments that survived into `O4` (`gender` gated by a count LRT, and `new_business` as a count credibility factor) both relied on **claim counts**! Meanwhile, `base_<peril>` already captures severity relativities (`vehicle_value_group`, `sum_insured`) from 3M developed rows.
*   **How to Relax Safely**: Fit all multi-factor emerging residual adjustments on **claim counts** (Poisson log-link with offset = $\log(\text{base})$ normalized to counts) with strong L2 regularization, and apply the resulting frequency multiplier to the cost base (Idea 2).

#### Rank 4: Local 16 GB RAM Bottleneck Restricting Base Refits (Compute)
*   **Estimated Cost**: **0.0015 – 0.0030 $D^2$**
*   **Reasoning**: Because loading the 3M-row Developed frame takes 7–11 GB RAM and 10–40 minutes on the Mac mini, exploration has been restricted to the 380-line `fit_predict` script on frozen bases. Yet `base_compat` blended 50:50 with `P0` already showed **+0.00238** on Primary (0.11792 vs 0.11554) before `O4` existed, and `base_pooled` was crippled by row-duplication weighting (which bypassed `min_data_in_leaf` and memorised claims).
*   **How to Relax Safely**: Use your unused **Kaggle GPU/CPU notebooks (30 GB RAM)** to run 2–3 targeted base jobs (downcasting float64 to float32 and object to category shrinks the 3M frame to ~1.2 GB RAM), export only the scored 274k-row emerging prediction matrices (~15 MB Parquet) back to the Mac mini, and let the fast local harness consume them as precomputed base columns.

#### Rank 5: "Balance on Train" Without Adjusting for Realised Weather & Open-Claim Recovery Maturity (Level / Target Governance)
*   **Estimated Cost**: **0.0015 – 0.0025 $D^2$**
*   **Reasoning**: "Balance on train" forces the line level to match 2023–24 actuals and relies on model relativities to move the 2025 level. However: (a) 2025 weather is milder (`hail_days` drops from 14–17 in 2023–24 to **11.4** in 2025), and tree-based `base_<peril>` models trained on Developed data (`hail_days` ~6) saturate flat at their top split (~9–10 hail days), so the model fails to adjust 2025 downward for milder weather; (b) in 2024, open claims at 31 Dec 2024 have **zero recoveries booked** ("recoveries are booked only when a claim settles"), creating a maturity distortion between 2023 and 2024 in Fold B vs Fold R.
*   **How to Relax Safely**: Inject an explicit log-linear excess-weather adjustment inside `fit_predict` before balancing on train (Idea 6) and adjust open claims for expected settled recovery ratios (Idea 8).

#### Rank 6: Admissibility Exclusions (Governance)
*   **Estimated Cost**: **0.0000 – 0.0005 $D^2$ (Do NOT relax)**
*   **Reasoning**: Excluding leaked fields (`customer_service_contacts`, `renewal_decision`) and proxy/rate-leakage fields (`written_premium`, `email_domain`, `campaign_code`, `marketing_segment`, `branch`) is sound actuarial governance—and your 25 Oct re-test confirmed they did not help. With 20% of judging on governance and 40% including panel view, **keep all admissibility exclusions strictly intact**.

---

### 2. White Space Analysis: Untapped Disciplines and What Fits

| Discipline / Method | Status in Pipeline | Fit for Next 40h | Why It Fits (or Doesn't) |
| :--- | :--- | :--- | :--- |
| **Parametric / Spline Calibration Curve on Log-Base** | Untapped in Part 2 (only isotonic was tried in Part 1) | **VERY HIGH** | Isotonic regression overfits Tweedie tails with step functions. A smooth 2- or 3-DOF quadratic/spline curve on $\log(\text{base})$ directly fixes the non-monotone Motor deciles (0.44 / 1.35 / 0.75) with near-zero overfit risk. |
| **Frequency–Severity Residual Decomposition** | Partially tapped (2 count factors in O4) | **VERY HIGH** | Severity relativities transfer cleanly from the 3M Developed rows (up to a peril scalar), whereas emerging residual signal sits in **frequency**, where Poisson count estimation has ~10x lower variance than Tweedie cost. |
| **Cross-Market Latent Feature Bridge (Transfer Imputation)** | Untapped (currently flat band mean + `NaN`) | **VERY HIGH** | Imputing exact age from `(age_band, years_licensed, ncb_years)` and marginalizing/imputing credit & mileage before scoring `base_<peril>` requires **zero refitting** of the 3M-row models and fixes the 2025 youth shift (50% aged 17–34). |
| **Hierarchical / Multi-Factor L2 Mixed-Effects (Joint Credibility)** | Untapped on counts (failed on cost with low penalty) | **HIGH** | Simultaneously estimating `previous_insurer`, `region`, `elevation_m`, `missed_payments_12m`, `construction_type`, `energy_rating`, and `trees_within_5m` in a heavily L2-penalized Poisson offset GLM prevents collinear dilution of `gender`. |
| **Cross-Line Pooling of Shared Customer/Property Factors** | Untapped | **HIGH** | Buildings (~500 claims/yr) and Contents (~500 claims/yr) share identical property/location features, and all 3 lines share `missed_payments_12m`, `elevation_m`, and `region`. Pooling counts across lines doubles to 9x's sample size. |
| **Physical Weather Extrapolation (Hybrid Tree + Linear Tail)** | Failed when fitted on 1 emerging year alone | **HIGH** | Trees trained on Developed (`hail_days` ~6) clamp flat above 10 hail days. Adding a physical log-linear excess-weather slope bridges 2023–24 (14–17 hail days; 2023 wildfire spike) to 2025 (11.4 hail days). |
| **Actuarial Target Engineering (Open-Claim Recovery + BF IBNR)** | Untapped (raw chain-ladder used) | **MEDIUM-HIGH** | Open claims at 31 Dec 2024 have 0 recoveries booked, and dividing H2 2024 claims by small CDFs inflates row target variance. Bornhuetter-Ferguson (BF) row targets stabilize training without altering validation targets. |
| **Covariate-Shift Importance Weighting / Domain Adaptation** | Used in eval metric (1/3 weight), untapped in training | **MEDIUM-HIGH** | Interacting `gender` with young drivers (`<35`, 50% of 2025) and weighting `fit_predict` toward the 2025 age/new-business mix aligns parameters with the test book. |
| **Native-Weighted Pooled GBDT (`sample_weight` + `min_child_weight`)** | Untapped (row duplication was used and memorised claims) | **MEDIUM-HIGH** | Native `sample_weight = 12` with scaled `min_child_weight` on Kaggle GPU lets pooled trees learn emerging interactions (`region`, `previous_insurer`, `gender`) without leaf memorization. |
| **Neural Tabular (TabM / RealMLP), Multi-Task Nets, Distillation, AutoML** | Tried in Part 1 (no gain) | **LOW (Skip)** | High variance on 65k noisy Tweedie rows, opaque failure modes, and poor ROI inside a 40-hour window. |

---

### 3. Concrete, Testable Ideas (Ranked by Expected Gain on the 2025 Book)

#### Idea 1: Non-Linear Log-Base Recalibration Curve (Fixing the Non-Monotone Motor Deciles)
*   **Mechanism**: Inside `fit_predict`, replace the rigid unit-slope offset $\log(\text{base}_i)$ on Motor (and test on Buildings/Contents) with a 2-degree-of-freedom parametric recalibration curve on the centered log-annualized rate $z_i = \log(\text{base}_i / \text{exposure}_i) - \bar{z}$:
    $$\log(\hat{\mu}_i) = \log(\text{exposure}_i) + \beta_0 + \beta_1 z_i + \beta_2 z_i^2$$
    (or a 2-piecewise hinge basis $\beta_1 z_i + \beta_2 [z_i - \tau]_+ + \beta_3 [\tau - z_i]_+$ at the median $\tau = 0$), fitted via Ridge Poisson/Tweedie GLM with penalty on $(\beta_1 - 1)^2 + \beta_2^2$. In addition, allow the BI peril base (`base_motor_bi`, where emerging max severity is 156k vs 11.9m developed) to take its own log-slope $\beta_{\text{BI}} < 1$ before summing the motor perils.
*   **Why It Transfers to the 2025 Broker Book**: Across both validation folds, Motor A/E is 0.44 in Decile 1, 1.34–1.37 in Deciles 4–5, and 0.73–0.78 in Deciles 8–9. A concave quadratic/hinge curve ($\beta_2 < 0$) pulls down both tails (Decile 1 and Deciles 8–10) while lifting the middle deciles (Deciles 4–7). Because the 2025 broker book shifts 50% of exposure into ages 17–34 (which populate Deciles 5–10), correcting the top-decile overprediction and mid-decile underprediction is crucial for the 2025 mix shift.
*   **Expected Primary Gain**: **+0.0025 to +0.0045**
*   **Harness Fit**: Pure `fit_predict` change (~20 lines of code on fixed bases).
*   **Compute & Where**: Mac mini CPU, **0.2 hours** (~2 minutes per run).
*   **Overfitting Risk & Test**: Very low (only 2 parameters across 32,000 rows per fold). Verify that Motor predicted-rate decile A/E dispersion shrinks in both Fold B and Fold R and that level-neutral Fold B improves by > +0.0015.

#### Idea 2: Differential-Penalty Poisson Count-Residual GLM (Capturing the Missed Segment Relativities Without Diluting Gender)
*   **Mechanism**: Earlier joint GLMs failed because they were fitted on high-variance **Tweedie cost** with uniform penalties, allowing noisy cost residuals on multi-level factors to shift the `gender` coefficient. Instead, fit a **Poisson log-link GLM on claim counts (`claim_count`)** using $\log(\text{base\_after\_O4})$ (rescaled to mean claim frequency) as a **fixed offset** (so `gender` and `new_business` are already locked in, OR with penalty $\lambda_{\text{gender}} = 0.01$ and a heavy penalty $\lambda_{\text{factors}} \in \{150, 300, 600\}$ on standardized one-hot indicators of the missed factors):
    *   *Motor*: `previous_insurer` (levels with $n \ge 200$), `region` (E01–E06), `missed_payments_12m`, `elevation_m` binned (`0-3m`, `3-15m`, `15m+`), `cover_type`, `fuel_type`, `licence_points` (`0`, `1-2`, `3+`), `occupation` (grouped by empirical Bayes count residual).
    *   *Property (pooled across Buildings + Contents counts = ~1,000 claims/yr)*: `construction_type`, `energy_rating` (ordinal A–G + categorical), `trees_within_5m`, `flood_zone`, `elevation_m` binned, `missed_payments_12m`, `years_at_address` (`0-3` vs `4+`), `region`.
    Multiply the cost prediction by the fitted frequency multiplier $\exp(X_i \hat{\beta}_{\text{count}})$.
*   **Why It Transfers to the 2025 Broker Book**: `previous_insurer` (A/E 0.47–1.69 in both folds) and `region (E01-E06)` are completely unseen by the Developed `base_<peril>` model, and `previous_insurer` is one of the top 3 features separating the 2025 broker book (AUC 0.74). Fitting on claim counts reduces estimation variance ~10x relative to cost, and locking `gender` in the offset makes dilution mathematically impossible.
*   **Expected Primary Gain**: **+0.0030 to +0.0055**
*   **Harness Fit**: Pure `fit_predict` change (requires `claim_count` on `train`, which `O4` already uses for its LRT gate and new-business factor).
*   **Compute & Where**: Mac mini CPU, **0.5 hours** (~3 minutes per run across 3 settings of $\lambda_{\text{factors}}$).
*   **Overfitting Risk & Test**: Low–Medium. Test 3 pre-registered shrinkage strengths ($\lambda \in \{150, 300, 600\}$) as a single unified mechanism; require paired bootstrap 10th percentile > 0 and improvements on both natural and 2025-reweighted metrics.

#### Idea 3: Cross-Market Feature Bridge for `base_<peril>` (Imputing Exact Age from Licensing/NCB & Marginalizing Missing Credit/Mileage)
*   **Mechanism**: You do **not** need to refit the 3M-row Developed models to fix the schema distortion when scoring Emerging rows through `base_<peril>`:
    1. **Exact Driver Age Reconstruction**: Within each `driver_age_band` $[a_{\min}, a_{\max}]$, exact age is tightly bounded below by licensing age (`17 + years_licensed`) and `17 + ncb_years`. On the 3M Developed frame (or via a lookup table by `(driver_age_band, years_licensed, ncb_years, named_young_driver, marital_status)`), compute the exact conditional mean $\mathbb{E}[\text{driver\_age} \mid \text{band}, \text{years\_licensed}, \text{ncb\_years}]$. Do the same for `policyholder_age_band` using `customer_tenure` and `years_at_address`.
    2. **Eliminating `NaN` Branch Routing on Missing Fields**: Instead of passing `NaN` for `credit_band`, `annual_mileage`, `telematics`, and the 12% missing emerging columns into `base_<peril>`, train a lightweight Ridge/LightGBM bridge on the Developed frame to impute expected `credit_band` (from `missed_payments_12m`, `payment_method`, `tenure`, `occupation`, `voluntary_excess`) and `annual_mileage` (from `vehicle_use`, `fuel_type`, `engine_size_cc`, `urban_density`, `occupation`), and impute the 12% missing emerging fields with their Developed medians/modes (or score `base_<peril>` across 3 quantile profiles of the missing fields and average).
*   **Why It Transfers to the 2025 Broker Book**: In 2025, drivers aged 17–34 jump from 39% to **50% of exposure**, and `years_licensed` is a primary separator of the 2025 broker book (AUC 0.74). Replacing the flat `20.5` age mean with `years_licensed`/`ncb_years`-resolved age restores the steep 17-to-24 risk gradient across half of the 2025 test book without touching a single tree split.
*   **Expected Primary Gain**: **+0.0020 to +0.0040**
*   **Harness Fit**: Re-score the saved Part 1 `base_<peril>` models on the 274k emerging rows (65k train + 209k test) with bridged inputs and attach as `base_bridged_<peril>`.
*   **Compute & Where**: Mac mini (or Kaggle CPU), **1.0 hour** (no 3M-row training needed; only a lookup table + scoring 274k rows).
*   **Overfitting Risk & Test**: Near zero (zero parameters fitted on Emerging data; purely a deterministic schema bridge). Test by swapping `base_<peril>` for `base_bridged_<peril>` inside `O4`.

#### Idea 4: Convex Multi-Base Ensemble Underneath `O4` (`base_<peril>` + `base_compat_<peril>` + `base_pooled`)
*   **Mechanism**: On 25 Oct, blending `base_compat` 50:50 with `P0` lifted Primary by **+0.00238** (`0.11554 -> 0.11792`), because `base_compat` was trained *without* credit, mileage, and telematics—forcing its trees to put full split weight on the features actually observed in Emerging (`years_licensed`, `ncb_years`, `missed_payments_12m`, `elevation_m`, `energy_rating`, `trees_within_5m`). However, that blend was tested *without* `O4`'s adaptation layer.
    1. Immediately test blending the existing `base_compat` and `base_pooled` columns with `P0` *inside* `O4` across all three lines: $\text{Base}_L = w_1 \text{P0}_L + w_2 \text{base\_compat}_L + w_3 \text{base\_pooled}_L$ (e.g., $(0.65, 0.25, 0.10)$ on Motor, $(0.60, 0.25, 0.15)$ on Buildings, $(0.40, 0.20, 0.40)$ on Contents).
    2. On Kaggle CPU/GPU, refit `base_compat` **per peril** (`base_compat_<peril>`) so each peril gets schema-compatible trees before `O4`'s per-peril credibility levels are applied.
*   **Why It Transfers to the 2025 Broker Book**: Reduces tree variance and eliminates missing-feature split attenuation across all 209k test rows.
*   **Expected Primary Gain**: **+0.0020 to +0.0035**
*   **Harness Fit**: Step 1 is a 5-line `fit_predict` change using already-attached columns; Step 2 is a base refit on Kaggle exported to the Mac.
*   **Compute & Where**: Step 1: Mac mini (**0.2 hours**). Step 2: Kaggle CPU/GPU (**1.5 hours**).
*   **Overfitting Risk & Test**: Very low (2 fixed blend weights per line). Test Step 1 immediately against `O4` under the keep rule.

#### Idea 5: Deconfounding the 2025 Broker Transfer (`new_business` $\times$ `previous_insurer` / `ncb_years` / `channel` Hierarchy)
*   **Mechanism**: In 2023–24, `new_business` was 14% organic walk-in business, mixing (a) previously uninsured novices (`previous_insurer` = None/missing or high-risk, `ncb_years = 0`, A/E up to 1.69) and (b) experienced drivers switching insurers (`previous_insurer` = prime insurer, `ncb_years > 0`, A/E down to 0.47). In 2025, `new_business` jumps to **76%** because the insurer acquired a **broker portfolio** on 1 Jan 2025. `O4`'s flat 1-parameter `new_business` surcharge assigns the historical organic new-business penalty indiscriminately to all 76% of the 2025 book.
    Replace the 1-parameter `new_business` factor in `O4` with a hierarchical Empirical Bayes claim-count relativity ($k=150$) segmented by `new_business` $\times$ `previous_insurer` (and `channel` / `ncb_years > 0`), so transferred policies with established `previous_insurer` and `ncb_years > 0` get their true lower relativity while unseasoned/high-risk `previous_insurer` levels get their true surcharge.
*   **Why It Transfers to the 2025 Broker Book**: Directly models the exact acquisition mechanism of the 2025 test set (where `previous_insurer` is a top driver of the AUC 0.74 renewal-vs-NB split and has a 0.47–1.69 A/E spread).
*   **Expected Primary Gain**: **+0.0020 to +0.0035** (with an even larger gain on the 2025 Leaderboard where `new_business` is 76%).
*   **Harness Fit**: Pure `fit_predict` change (~25 lines replacing the flat `new_business` scalar).
*   **Compute & Where**: Mac mini CPU, **0.3 hours**.
*   **Overfitting Risk & Test**: Low with $k=150$ count credibility shrinkage. Verify that the 1/3 reweighted validation score (which upweights `new_business`) jumps more than the natural validation score.

#### Idea 6: Unclamped Physical Weather Extrapolation (`hail_days`, `wildfire_index`, `dry_summer_index`, `wind_exposure_index`)
*   **Mechanism**: `base_<peril>` trees were trained on Developed data where `hail_days` ~6 and `wildfire_index` ~0.15. In Emerging, `hail_days` was 14–17 in 2023–24 and drops to **11.4** in 2025; `wildfire_index` is ~0.50 (and Buildings fire claims swung from 49 in 2023 to 21 in 2024). Because tree models clamp flat above their highest training split (~9–10 hail days in Developed), `base_<peril>` predicts almost the exact same weather risk for 11.4 hail days as for 16 hail days—and "balance on train" locks the 2023–24 severe weather level into 2025!
    Instead of estimating a noisy weather elasticity from a single emerging year (which confounds weather with region), compute the physical excess-weather log-slope $\gamma_p$ on peril claim frequency across Developed + Emerging (or use bounded physical count elasticities on excess weather above the Developed saturation threshold: $\Delta \text{hail}_i = \max(0, \text{hail\_days}_i - 9.0)$ on `motor_glass_hail` and `bldg_storm`, and `wildfire_index * dry_summer_index` on `bldg_fire`) inside `fit_predict` *before* balancing on train.
*   **Why It Transfers to the 2025 Broker Book**: In Folds B and R, this reconciles the 2023 vs 2024 Buildings fire/storm level swing (stabilizing Fold B vs Fold R). In 2025, when `hail_days` drops from ~15.5 to 11.4, applying the multiplier before "balance on train" automatically scales 2025 hail and storm predictions down to match 2025's milder realised weather.
*   **Expected Primary Gain**: **+0.0015 to +0.0025** on Folds B & R (**+0.0025 to +0.0045** on 2025 LB).
*   **Harness Fit**: Pure `fit_predict` change on peril bases before line balancing.
*   **Compute & Where**: Mac mini CPU, **0.4 hours**.
*   **Overfitting Risk & Test**: Low if constrained to the physical peril matches (`hail_days` $\to$ hail/storm perils; `wildfire_index` $\to$ fire peril) with shrinkage $k=200$. Check that Fold B and Fold R Buildings $D^2$ gap narrows.

#### Idea 7: Native-Weighted Pooled Base (`base_pooled_w12_compat`) on Kaggle GPU
*   **Mechanism**: In `base_pooled`, weighting emerging rows at 1 gave them only 2.1% of the 3.06M rows, while weighting at 5 via *row duplication* broke tree leaf-size constraints and memorised claims. On Kaggle (30 GB RAM + GPU), refit the pooled model on the **schema-compatible feature set + `market_flag` + `gender` + `region` + `previous_insurer` + `construction_type`**, passing **native `sample_weight = 12.0`** on Emerging rows and scaling `min_child_weight` (sum of hessians in leaf) and `reg_lambda` by $12\times$ for emerging splits. This gives Emerging ~20% of total split gain while strictly forbidding any leaf with fewer than 200 unique emerging policies.
*   **Why It Transfers to the 2025 Broker Book**: Allows the GBDT/EBM ensemble to learn non-linear interactions between `gender`, `driver_age_band` (50% young in 2025), `previous_insurer` (76% NB in 2025), and `region (E01-E06)` regularized by 3M developed policies.
*   **Expected Primary Gain**: **+0.0020 to +0.0035**
*   **Harness Fit**: New base column (`base_pooled_w12`) generated on Kaggle (with strict out-of-fold 2023/2024 scoring) and blended inside `fit_predict`.
*   **Compute & Where**: Kaggle GPU/CPU notebook, **1.5 hours**.
*   **Overfitting Risk & Test**: Medium if `min_child_weight` is too small; low when `min_data_in_leaf >= 250` unique rows and `min_child_weight` is scaled. Test as a drop-in replacement for `base_pooled` in `O4` and extend blend to Motor and Buildings.

#### Idea 8: Young-Driver $\times$ Gender Interaction & Covariate-Shift Importance Weighting in `fit_predict`
*   **Mechanism**: Two parts inside `fit_predict`:
    1. **Actuarial Age $\times$ Gender Interaction**: In motor pricing globally, the Male vs Female frequency/severity gap is heavily concentrated in young drivers (`17-24` and `25-34` plus `named_young_driver = 1`) and flattens out above age 35. Currently, `O4` fits a single constant `gender` coefficient on a 2023–24 book that is 61% aged 35+, which understates the gender differential on the 17–34 cohort that makes up **50% of the 2025 book**. Add a penalized interaction `gender * is_under_35` (and `gender * named_young_driver`) gated by a claim-count LRT.
    2. **Stabilized Importance Weights**: Weight the GLM/credibility estimation inside `fit_predict` by clipped propensity odds $w_i = \text{clip}(\hat{p}_{2025}(x_i)/\hat{p}_{\text{train}}(x_i), 0.4, 2.5)$ so all emerging parameters optimize for the 2025 age/broker distribution while keeping effective sample size $> 70\%$.
*   **Why It Transfers to the 2025 Broker Book**: Directly aligns the `gender` and base-calibration parameters with the 50% under-35 / 76% new-business 2025 portfolio.
*   **Expected Primary Gain**: **+0.0015 to +0.0030**
*   **Harness Fit**: Pure `fit_predict` change.
*   **Compute & Where**: Mac mini CPU, **0.3 hours**.
*   **Overfitting Risk & Test**: Very low (1 extra interaction DOF on Motor). Check both the natural and 1/3 reweighted components of the Primary score.

#### Idea 9: Bornhuetter-Ferguson (BF) Training Target Stabilization & Open-Claim Recovery Adjustment
*   **Mechanism**: At 31 Dec 2024, 2024 policies are 0–12 months old, whereas 2023 policies are 12–24 months old. Two target distortions arise:
    1. **Unbooked Recoveries on Open Claims**: Section 1 notes: *"Open claims carry case estimates; recoveries are booked only when a claim settles."* Thus, open claims at 31 Dec 2024 (concentrated in 2024!) have `recoveries == 0`. If the chain-ladder factor was derived without conditioning on open vs settled status, 2024 open claims in high-recovery perils (Motor OD/TPPD) are overstated relative to 2023 settled claims.
    2. **Late-Year Chain-Ladder Variance Explosion**: Dividing a November/December claim by a small completion factor ($\text{CDF}_m \approx 0.40$) multiplies random claim spikes by $2.5\times$ while leaving policies with 0 reported claims at 0. Inside `fit_predict` (modifying *only* the training target vector $y_{\text{train}}$ used to fit GLMs/levels, while leaving the harness validation target untouched!), replace pure chain-ladder $Y_{i,p}^{\text{CL}} = \text{Reported}_{i,p} / \text{CDF}_m$ with the **Bornhuetter-Ferguson (BF)** target:
       $$Y_{i,p}^{\text{BF}} = \text{ReportedAdjusted}_{i,p} + (1 - \text{CDF}_m) \cdot \hat{\mu}_{i,p}^{\text{prior}}$$
       rescaled to preserve the peril total.
*   **Why It Transfers to the 2025 Broker Book**: In final submission, you train on 2023 + 2024 where H2 2024 is the freshest but least mature data. BF smoothing eliminates artificial target leverage on late-2024 claims when fitting peril levels and GLM coefficients.
*   **Expected Primary Gain**: **+0.0015 to +0.0025**
*   **Harness Fit**: Pure `fit_predict` change (constructing $y_{\text{train}}^{\text{BF}}$ inside `fit_predict`).
*   **Compute & Where**: Mac mini CPU, **0.5 hours**.
*   **Overfitting Risk & Test**: Low. Check whether Fold R (which trains on immature 2024 and scores mature 2023) shows a marked jump in stability and $D^2$.

#### Idea 10: Cross-Line Customer & Geospatial Risk Sharing (`missed_payments_12m`, `elevation_m`, `region`, `prior_claims_5y`)
*   **Mechanism**: Buildings (~500 claims/yr) and Contents (~500 claims/yr) are too sparse individually for standalone factors to clear the +0.0020 pooled bar. However, look at Section 2: `missed_payments_12m=1` has A/E **1.18–1.23 on Motor** AND **1.36–2.05 on Buildings**! Low `elevation_m` (`0-3m` vs `15-60m`) has A/E **1.15 vs 0.81 on Motor** AND drives flood/storm risk on **Buildings and Contents** (`flood_zone 3` A/E 1.29–1.61)!
    Estimate a **single pooled customer/geospatial frequency factor** across all 3 lines combined (~4,500 claims/yr) for the shared variables (`missed_payments_12m`, `elevation_m`, `flood_zone`, `region E01-E06`, `prior_claims_5y`, `hist_n_policy`), and apply it simultaneously to Motor, Buildings, and Contents.
*   **Why It Transfers to the 2025 Broker Book**: Pooling all three lines increases the sample size to ~4,500 claims/yr, shrinking estimation variance to near zero while moving all three lines in unison to clear the +0.0020 pooled hurdle.
*   **Expected Primary Gain**: **+0.0015 to +0.0030**
*   **Harness Fit**: Pure `fit_predict` change (can be combined directly into Idea 2's Poisson count GLM).
*   **Compute & Where**: Mac mini CPU, **0.3 hours**.
*   **Overfitting Risk & Test**: Very low (shared parameters across 65k rows). Verify that all three lines show positive within-line $D^2$ gains.

#### Idea 11: Exposure Duration Hazard & Part-Year Policy Correction
*   **Mechanism**: Section 1 states: *"The prediction is a row total (exposure = years on cover, rows can be part-year)."* In personal lines, part-year policies (`exposure < 0.95`) in historical data are a mix of mid-term cancellations (often associated with non-payment or total-loss claims, where a policy cancels after a write-off so exposure is truncated while a full claim sits on the row!) and short-term/seasonal policies. Meanwhile, in the 2025 test book, check the distribution of `exposure` and `start_month`. If historical part-year policies have an elevated annualized A/E (because of post-claim cancellation truncation or seasonal exposure in high-hail months), `balance on train` and GLM offsets are distorted by part-year rows. Test an exposure-elasticity parameter $\text{exposure}^\alpha$ (or a `part_year` indicator in the count GLM + weighting GLM estimation by `exposure` rather than raw rows) and a `start_month` seasonal weather alignment.
*   **Why It Transfers to the 2025 Broker Book**: Prevents mid-term cancellation bias in 2023–24 training data from distorting the 2025 annual broker transfer book.
*   **Expected Primary Gain**: **+0.0010 to +0.0020**
*   **Harness Fit**: Pure `fit_predict` diagnostic and 5-line adjustment.
*   **Compute & Where**: Mac mini CPU, **0.2 hours**.
*   **Overfitting Risk & Test**: Low (1 parameter). Check A/E of `exposure < 1.0` vs `exposure == 1.0` across both folds first; only keep if consistent in B and R.

#### Idea 12: Depth-2 Monotone Poisson Count-Residual Boosting (Constrained GAM-Tree Layer)
*   **Mechanism**: Your earlier LightGBM residual boost failed because it was fitted on **Tweedie cost** with unconstrained depth (allowing trees to isolate individual large claims). If a non-linear interaction remains after Ideas 1, 2, and 5, fit a **Depth-2 (max_leaves = 3 or 4) Poisson count-residual LightGBM** with `offset = log(expected_count_after_GLM)`, `min_child_samples = 1500`, `reg_lambda = 200.0`, `learning_rate = 0.03`, `n_estimators = 40`, restricted strictly to the 12 core domain-shift features (`log_base`, `driver_age_band`, `years_licensed`, `gender`, `new_business`, `previous_insurer`, `region`, `elevation_m`, `missed_payments_12m`, `construction_type`, `energy_rating`, `trees_within_5m`).
*   **Why It Transfers to the 2025 Broker Book**: Depth-2 trees on Poisson counts with `min_child_samples = 1500` (~5% of a fold) cannot memorize claims, yet can capture any 2-way interaction between the 2025 broker shift features (`new_business * previous_insurer`, `gender * driver_age_band`, `flood_zone * elevation_m`) that a main-effects GLM misses.
*   **Expected Primary Gain**: **+0.0010 to +0.0025**
*   **Harness Fit**: Pure `fit_predict` change.
*   **Compute & Where**: Mac mini CPU, **0.3 hours**.
*   **Overfitting Risk & Test**: Medium. Only test *after* the linear Poisson count GLM (Idea 2) is locked in; require bootstrap 10th percentile > 0 and < 0.0015 gap between train and valid $D^2$ gain.

---

### 4. The Weakest Spot in the Analysis So Far

The single largest amount of lift (+0.0050 to +0.0090 $D^2$) is currently trapped by **a mathematical Catch-22 between your base calibration, your loss function on the adaptation layer, and your keep rule**:

1. **The Base Engine (`base_<peril>`) Is Globally Miscalibrated in Slope and Shape on Emerging Data**:
   Your own diagnostics prove this beyond doubt: Motor predicted-rate deciles have A/E of **0.44 in Decile 1**, **1.34–1.37 in Deciles 4–5**, and **0.73–0.78 in Deciles 8–9** in *both* validation folds. Why? Because pushing Emerging rows into a Developed tree model with `NaN` for credit/mileage/telematics (plus 12% missingness elsewhere) routes every row down the Developed market's 2.5%-missingness branches, while plugging in flat in-band mean age (`20.5` for all 17–24 drivers) blinds the model to `years_licensed` and `ncb_years`, and Emerging's truncated BI severity (156k max vs 11.9m) compresses the upper tail. Yet `O4` treats $\log(\text{base})$ as a fixed offset with slope $= 1.0$.

2. **The Residual Layer Was Tested One Factor at a Time on Noisy Cost Against a Pooled 3-Line Hurdle**:
   *   When you tried to fix the glaring segment misses (`previous_insurer` 0.47–1.69, `region E01` 1.15, `elevation_m` 1.15 vs 0.81, `missed_payments_12m` 1.20–2.05, `construction_type` 1.42, `energy_rating` 1.65 vs 0.65, `trees_within_5m` 1.48) **one factor at a time**, none could clear **+0.0020 on the 3-line pooled metric**—because a single rating factor or a single property line (10–15% of pooled deviance) mathematically cannot move a pooled 3-line metric by +0.0020 on its own.
   *   When you tried combining 6–8 factors in a joint GLM, you fitted them on **Tweedie ultimate cost** (whose residual variance on 32k rows is ~10x higher than claim-count variance due to random large BI and fire claims) with uniform regularization, which corrupted the weights and diluted `gender`.

**How to break the Catch-22 in the next 12 hours (Recommended Execution Sequence)**:
1. **First (Mac mini, 1 hour)**: Fix the non-monotone base shape inside `fit_predict` using **Idea 1** (quadratic/hinge recalibration curve on $\log(\text{base})$ + separate BI slope) and **Idea 4 Step 1** (blending the existing `base_compat` and `base_pooled` columns into `O4`).
2. **Second (Mac mini, 2 hours)**: Lock `O4`'s gender + recalibrated base into the offset, and fit the missed emerging factors (`previous_insurer` $\times$ `new_business`, `region`, `elevation_m`, `missed_payments_12m`, `construction_type`, `energy_rating`, `trees_within_5m`, `flood_zone`) **jointly on Poisson claim counts** with heavy L2 regularization ($\lambda \in [150, 500]$) pooled across shared cross-line features (**Ideas 2, 5, 8, and 10**), plus the physical excess-weather adjustment (**Idea 6**).
3. **Third (In parallel on Kaggle CPU/GPU, 2–3 hours)**: Run the deterministic feature bridge re-scoring of `base_<peril>` using exact age from `years_licensed`/`ncb_years` (**Idea 3**) and refit `base_compat_<peril>` and native-weighted `base_pooled_w12` (**Ideas 4 & 7**), exporting the scored 274k-row base matrices back to your Mac mini for Day 2 blending.

```json
[
  {
    "name": "Non-linear log-base recalibration curve",
    "mechanism": "Replace the rigid unit-slope log(base) offset on Motor (and test on Property) with a ridge-penalized quadratic or 2-slope hinge curve on centered log annualized base rate plus a separate log-slope on the BI peril base to fix the non-monotone decile A/E (0.44 in decile 1, 1.35 in deciles 4-5, 0.75 in deciles 8-9).",
    "expected_gain": "+0.0025 to +0.0045",
    "harness_fit": "fit_predict change on fixed bases",
    "compute": "Light (GLM fit on 32k rows)",
    "where": "Mac",
    "hours": 0.2,
    "over
