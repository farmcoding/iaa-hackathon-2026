# Questions and hypotheses register

Open questions and working hypotheses, from Karl and from the work. Each entry keeps
its evidence so the trail stays readable. Status: open / exploring / answered / parked.
The dashboard renders this file.

| ID | Date | Raised by | Question or hypothesis | Status | Evidence and next step |
|---|---|---|---|---|---|
| Q1 | 2026-10-18 | Karl | What jurisdiction is motor liability? | answered | Reads as England and Wales (inference): unlimited third-party injury cover, Ogden rate, PPOs, whiplash tariff. Flood zones 1-5 don't match the UK scheme. research/sources.md J1-J2 |
| Q2 | 2026-10-18 | Karl | Can large losses be capped from public data? | answered | Only the tail shape transfers (currency is fictional). Cap from own data: 7,583 BI claims, 19 above 250k. Backlog 2.11, 3.5 |
| Q3 | 2026-10-18 | Karl | Do we have loss history to estimate ultimates? | answered | Yes: report-delay chain ladder plus BI frequency x severity; one valuation date, so case-reserve adequacy is an assumption. Owner track 3.2-3.5 tests alternatives |
| Q4 | 2026-10-18 | Karl | Should good models be held back from submission? | answered | No benefit: ties go to the earlier entry, the public board is a sample, the final pick is manual |
| Q5 | 2026-10-18 | Karl | Why a 900-second compute budget; can it double? | answered | About six times the baseline run, so 6-8 experiments an hour and no runaway runs. Doubled to 1,800 s on 2026-10-18 |
| Q6 | 2026-10-18 | Karl | Can we use Kaggle GPUs, and run GPU fits in parallel? | open | About 30 GPU-hours a week (secondary sources); no API shows the balance. Local M4 Metal is free and tried first. Ask before every GPU run with run time and balance. Backlog Tier 4 |
| Q7 | 2026-10-18 | Karl | Are Richman-circle methods repeatable from public repos? | answered | Usable: RonRichman/frmtpl-scaling-laws (GPL-2.0), lorentzenchr/credibility (Apache-2.0), model-diagnostics (MIT). Not usable: CredibilityTransformer (non-commercial), AITools4Actuaries and SAV tutorials (no licence) |
| Q8 | 2026-10-18 | Karl | Have predictors been treated properly (feature generation, binning, outliers)? | exploring | So far minimal: raw features into the GBM, native missing values, no binning (trees don't need it), no predictor winsorising, no engineered interactions. Backlog 2.1, 2.2, 2.8-2.10 |
| H1 | 2026-10-18 | Work | Fold A overstates gains; fold B tracks the leaderboard | exploring | Baseline to best: fold A +0.024, fold B +0.005, public +0.006. Primary now weights B twice. Recheck with every submission |
| H2 | 2026-10-18 | Work | The gap to the leader (0.017) is mainly ranking within lines, not line levels | open | Test with a per-line trended constant submission (backlog 3.8), awaiting approval |
| H3 | 2026-10-18 | Work | Motor bodily injury noise limits motor D² | open | Backlog 1.1 |
| H4 | 2026-10-18 | Work | Anchoring hides weather-driven years for buildings | open | Fold B buildings A/E 1.049; backlog 1.2 |
| H5 | 2026-10-18 | Work | Model under-predicts out of time because some features drift with calendar time | open | Without anchoring A/E 1.15 (fold A) and 1.09 (fold B). Backlog 3.7 |
