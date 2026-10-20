"""Generate the stage-0 candidate feature registry (owner item 3.35).

One row per column that reaches `harness.dataset()`, plus dedicated rows for the
columns removed earlier, in `features.build()`, as leaks (`features.FORBIDDEN`).
Columns are classified by role, family, availability at a real quote date,
competition admissibility, real-world admissibility, a fairness flag, which lines
currently use them (read from `experiment._feature_cols`, `LINE_DROP` and `PRUNED`),
and a reason when excluded. The family/availability/real-world/fairness calls are
this review's judgement (cited to docs/pipeline_review.md and reports/assumptions_log.md
where that judgement was made); role, used-on-lines and the code-level exclusion
reason are read straight from the frozen source so the registry stays accurate as the
code changes.

Usage: PYTHONPATH=src python scripts/feature_registry.py
Writes docs/feature_registry.csv and docs/feature_registry.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from iaa import data, experiment, features, harness

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "docs" / "feature_registry.csv"
OUT_MD = ROOT / "docs" / "feature_registry.md"

TARGET_PREFIXES = ("cost_", "n_", "ult_", "comp_")
IDENTIFIER_COLS = {"row_id", "policy_id", "customer_id", "address_id"}
STRUCTURAL_COLS = {"period_start", "period_end", "split", "year", "line", "exposure"}

# --- family: this review's grouping, cited to docs/pipeline_review.md section 2.2 (the
# family/redundancy map) and the EDA groups in scripts/feature_study.py. Fallback "other".
FAMILY: dict[str, str] = {}
for c in (
    "rainfall_mm",
    "hail_days",
    "freezing_days",
    "mean_temperature_c",
    "dry_summer_index",
    "wildfire_index",
    "wind_exposure_index",
):
    FAMILY[c] = "weather"
for c in (
    "distance_to_river_km",
    "elevation_m",
    "flood_zone",
    "soil_type",
    "urban_density",
    "region",
    "years_at_address",
):
    FAMILY[c] = "geography"
for c in (
    "customer_age",
    "policyholder_age",
    "marital_status",
    "occupation",
    "household_size",
    "email_domain",
    "credit_band",
    "customer_tenure",
    "tenure",
    "works_from_home",
    "lodgers",
    "missed_payments_12m",
    "payment_method",
    "discount_applied",
):
    FAMILY[c] = "customer"
for c in (
    "driver_age",
    "vehicle_age",
    "make_model",
    "body_type",
    "engine_size_cc",
    "fuel_type",
    "transmission",
    "seats",
    "vehicle_colour",
    "vehicle_value_group",
    "vehicle_use",
    "years_licensed",
    "named_young_driver",
    "driver_assist",
    "telematics_opt_in",
    "telematics_score",
    "annual_mileage",
    "ncb_years",
    "voluntary_excess",
    "compulsory_excess",
    "licence_points",
    "prior_claims_5y",
    "overnight_parking",
    "new_business",
    "renewal_count",
    "cover_type",
):
    FAMILY[c] = "motor risk/policy"
for c in (
    "property_type",
    "property_age",
    "year_built",
    "construction_type",
    "roof_type",
    "listed_building",
    "basement",
    "bedrooms",
    "sum_insured",
    "smoke_alarms",
    "security_level",
    "solar_panels",
    "trees_within_5m",
    "days_unoccupied",
    "mortgage",
    "occupancy",
    "energy_rating",
    "high_value_items",
    "accidental_damage_cover",
    "bicycle_cover",
):
    FAMILY[c] = "property risk/policy"
for c in (
    "branch_id",
    "campaign_code",
    "marketing_segment",
    "marketing_consent",
    "document_preference",
    "days_quote_to_inception",
    "previous_insurer",
    "written_premium",
    "start_month",
    "channel",
    "lines_held",
):
    FAMILY[c] = "policy admin/sales"
for c in ("hist_n_customer", "hist_cost_customer", "hist_n_policy", "hist_cost_policy"):
    FAMILY[c] = "claim history"

# --- availability at a real quote date (this review's judgement; default "yes").
AVAILABILITY: dict[str, str] = {
    "written_premium": "yes, but it is the incumbent's existing price, not available for a "
    "genuinely new risk (docs/pipeline_review.md section 2.1)",
    "telematics_opt_in": "yes",
    "telematics_score": "only for the ~12.6% of customers opted in; missing otherwise",
}
for c in (
    "rainfall_mm",
    "hail_days",
    "freezing_days",
    "mean_temperature_c",
    "dry_summer_index",
    "wildfire_index",
    "wind_exposure_index",
):
    AVAILABILITY[c] = (
        "partial: an address's climate normal is knowable at quote; the row's own-year "
        "realized weather (joined by [address_id, year], features.py) is not, and for the "
        "2025 test rows it is host-supplied (docs/pipeline_review.md stage-1 admissibility row)"
    )
for c in ("hist_n_customer", "hist_cost_customer", "hist_n_policy", "hist_cost_policy"):
    AVAILABILITY[c] = "yes, as of the row's period_start (features.claim_history)"

# --- competition admissibility (this review's judgement; default "yes" for anything that
# is not a target-derived or train-only leak). FORBIDDEN columns are "no" with a reason.
COMPETITION_ADMISSIBLE: dict[str, str] = {
    "written_premium": "yes: a supplied test field, fixed at inception, with no same-period "
    "claim signal found (docs/pipeline_review.md section 2.1); the owner drops it by choice, "
    "not because it is inadmissible",
}

# --- real-world admissibility for a tariff built outside the competition (this review's
# judgement, cited to docs/pipeline_review.md). Default "yes" for physical/behavioural risk
# factors; flagged otherwise.
REAL_WORLD_ADMISSIBLE: dict[str, str] = {
    "written_premium": "no as a rating input for the tariff being built: it is an output of "
    "someone else's pricing, not a risk characteristic (docs/pipeline_review.md section 2.1). "
    "Usable after modelling for adequacy and dislocation analysis",
}
for c in ("customer_age", "policyholder_age", "driver_age", "marital_status", "occupation"):
    REAL_WORLD_ADMISSIBLE[c] = (
        "contested: a common rating factor but jurisdiction-dependent restrictions apply "
        "(e.g. age/gender rating rules); needs a fairness review before a real tariff "
        "(docs/pipeline_review.md fairness row)"
    )
REAL_WORLD_ADMISSIBLE["email_domain"] = (
    "no: found independent of every other customer field and unstable between years "
    "(Cramer's V < 0.01), i.e. noise, not a risk factor (owner 2026-10-19, "
    "reports/reviews/2026-10-19_features.md)"
)
REAL_WORLD_ADMISSIBLE["credit_band"] = (
    "contested: predictive in this data but credit-based rating is restricted or banned "
    "for some lines in some jurisdictions; needs a fairness and regulatory review"
)
for c in (
    "rainfall_mm",
    "hail_days",
    "freezing_days",
    "mean_temperature_c",
    "dry_summer_index",
    "wildfire_index",
    "wind_exposure_index",
    "flood_zone",
    "distance_to_river_km",
    "elevation_m",
    "soil_type",
    "region",
    "urban_density",
):
    REAL_WORLD_ADMISSIBLE[c] = (
        "yes as a physical hazard factor, but between 66% and 96% of its variance is "
        "between addresses rather than between years (docs/pipeline_review.md section 2.2), "
        "so it also acts as a location fingerprint; review alongside the fairness fields"
    )
for c in (
    "branch_id",
    "campaign_code",
    "marketing_segment",
    "marketing_consent",
    "document_preference",
    "days_quote_to_inception",
):
    REAL_WORLD_ADMISSIBLE[c] = (
        "no: describes how the policy was sold, not the risk (owner 2026-10-19, "
        "reports/reviews/2026-10-19_features.md)"
    )

# --- fairness flag (this review's judgement, cited to the pipeline_review fairness row).
FAIRNESS_FLAGGED = {
    "customer_age",
    "policyholder_age",
    "driver_age",
    "marital_status",
    "occupation",
    "email_domain",
    "credit_band",
    "rainfall_mm",
    "hail_days",
    "freezing_days",
    "mean_temperature_c",
    "dry_summer_index",
    "wildfire_index",
    "wind_exposure_index",
    "flood_zone",
    "distance_to_river_km",
    "elevation_m",
    "soil_type",
    "region",
    "urban_density",
}

FIELDS = [
    "column",
    "role",
    "family",
    "availability_at_quote",
    "competition_admissible",
    "real_world_admissible",
    "fairness_flag",
    "used_on_lines",
    "reason_if_excluded",
]


def _feature_cols_by_line(df: pd.DataFrame, cfg: dict) -> dict[str, set[str]]:
    """Precompute once per line, not once per column: `_feature_cols` only reads the
    first row's `line` value and the per-column notna-any, so a small sample per line
    is enough and avoids re-filtering the full multi-million-row frame per column."""
    out = {}
    for line in data.LINES:
        sample = df.loc[df["line"] == line].head(500)
        out[line] = set(experiment._feature_cols(sample, cfg=cfg)) if len(sample) else set()
    return out


def _nonnull_by_line(df: pd.DataFrame) -> pd.DataFrame:
    """One pass: whether each column has any non-null value, per line."""
    return df.groupby("line", observed=True).apply(lambda g: g.notna().any(), include_groups=False)


def _used_on_lines(col: str, feature_cols_by_line: dict[str, set[str]]) -> list[str]:
    return [line for line in data.LINES if col in feature_cols_by_line[line]]


def _exclusion_reason(col: str, nonnull_by_line: pd.DataFrame) -> str:
    if col in features.FORBIDDEN:
        return "leak, removed in features.build before harness.dataset (features.FORBIDDEN)"
    if col in experiment.NON_FEATURES:
        if col in IDENTIFIER_COLS or col in STRUCTURAL_COLS:
            return "harness/dataset bookkeeping, not a rating input (experiment.NON_FEATURES)"
        return (
            "owner-excluded, not a rating factor (experiment.NON_FEATURES; see the inline "
            "comment in src/iaa/experiment.py and reports/assumptions_log.md)"
        )
    per_line = []
    for line in data.LINES:
        if col in experiment.LINE_DROP.get(line, set()):
            per_line.append(f"{line}: owner-dropped (experiment.LINE_DROP)")
        elif col in experiment.PRUNED.get(line, []):
            per_line.append(
                f"{line}: pruned, <=0.0001 D2 on shuffle (reports/features/model_card.json)"
            )
        elif col in nonnull_by_line.columns and not nonnull_by_line.loc[line, col]:
            per_line.append(f"{line}: not populated for this line")
    return "; ".join(per_line)


def build_registry() -> pd.DataFrame:
    df = harness.dataset()
    cfg = {"harness": {"use_premium": False}}  # Part 1 default; see configs/market_dev.yaml
    feature_cols_by_line = _feature_cols_by_line(df, cfg)
    nonnull_by_line = _nonnull_by_line(df)
    rows = []
    for col in df.columns:
        if col in IDENTIFIER_COLS:
            role = "identifier"
        elif col.startswith(TARGET_PREFIXES) or col == "y_ult":
            role = "target"
        elif col in STRUCTURAL_COLS or col in experiment.NON_FEATURES:
            role = "excluded"
        else:
            role = "feature"
        used = _used_on_lines(col, feature_cols_by_line) if role == "feature" else []
        reason = "" if role in ("identifier", "target") else _exclusion_reason(col, nonnull_by_line)
        rows.append(
            {
                "column": col,
                "role": role,
                "family": FAMILY.get(col, "other"),
                "availability_at_quote": AVAILABILITY.get(col, "yes"),
                "competition_admissible": COMPETITION_ADMISSIBLE.get(col, "yes"),
                "real_world_admissible": REAL_WORLD_ADMISSIBLE.get(col, "yes"),
                "fairness_flag": "yes" if col in FAIRNESS_FLAGGED else "no",
                "used_on_lines": ", ".join(used),
                "reason_if_excluded": reason,
            }
        )
    for col in sorted(features.FORBIDDEN):
        rows.append(
            {
                "column": col,
                "role": "excluded",
                "family": FAMILY.get(col, "other"),
                "availability_at_quote": AVAILABILITY.get(
                    col, "n/a, removed before features.build finishes"
                ),
                "competition_admissible": "no",
                "real_world_admissible": "no",
                "fairness_flag": "no",
                "used_on_lines": "",
                "reason_if_excluded": "leak, never reaches harness.dataset (features.FORBIDDEN); "
                "see the comment above features.FORBIDDEN in src/iaa/features.py and "
                "reports/data_quality.md",
            }
        )
    return pd.DataFrame(rows, columns=FIELDS).sort_values(["role", "column"]).reset_index(drop=True)


def write(reg: pd.DataFrame) -> None:
    reg.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    lines = [
        "# Feature registry (owner item 3.35)",
        "",
        "Generated by `scripts/feature_registry.py`. One row per column reaching "
        "`harness.dataset()`, plus the columns removed earlier as leaks "
        "(`features.FORBIDDEN`). Role, used-on-lines and the code-level exclusion reason "
        "are read from the frozen source; family, availability, admissibility and the "
        "fairness flag are this review's judgement, cited to `docs/pipeline_review.md` and "
        "`reports/assumptions_log.md`.",
        "",
        f"Counts by role: {reg['role'].value_counts().astype(int).to_dict()}",
        "",
        "| Column | Role | Family | Used on lines | Fairness flag | Reason if excluded |",
        "|---|---|---|---|---|---|",
    ]
    for r in reg.itertuples():
        lines.append(
            f"| {r.column} | {r.role} | {r.family} | {r.used_on_lines} | {r.fairness_flag} | "
            f"{r.reason_if_excluded} |"
        )
    lines.append("")
    lines.append(
        "Full detail, including availability at quote, competition admissibility and "
        "real-world admissibility, is in `feature_registry.csv`."
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    reg = build_registry()
    write(reg)
    print(f"wrote {OUT_CSV} and {OUT_MD} ({len(reg)} rows)")


if __name__ == "__main__":
    main()
