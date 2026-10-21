
### Consistency check -- captured models vs. the model's own fit

- `fold_B.pred_d2` (pooled D^2 recomputed from the captured sub-models): 0.1426
- `model_card.json['pooled_d2']` (same metric, scripts/model_card.py's own run): 0.1426
- dataset fingerprint this pack used: 902ed0cd8066
- dataset fingerprint model_card.json used: 902ed0cd8066

Exact match (diff 0.00e+00): the sub-models captured by this pack are the ones that produced the model's real predictions, on the same dataset fingerprint as model_card.json.

### All flagged predictors (fold B)

| line | sub-model | predictor | flags |
|---|---|---|---|
| motor | bi_frequency | years_licensed | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | vehicle_value_group | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | vehicle_age | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | make_model | non-monotone wiggle (6 sign changes) a regulator would query |
| motor | bi_frequency | engine_size_cc | non-monotone wiggle (3 sign changes) a regulator would query; 45% of sampled ICE curves disagree with the model's own average direction |
| motor | bi_frequency | body_type | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | seats | 43% of sampled ICE curves disagree with the model's own average direction |
| motor | bi_frequency | vehicle_colour | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | telematics_score | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | region | non-monotone wiggle (7 sign changes) a regulator would query |
| motor | bi_frequency | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | freezing_days | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | dry_summer_index | non-monotone wiggle (4 sign changes) a regulator would query |
| motor | bi_frequency | wildfire_index | non-monotone wiggle (4 sign changes) a regulator would query; 36% of sampled ICE curves disagree with the model's own average direction; 36% of sampled ICE curves disagree with the actuarial prior direction |
| motor | bi_frequency | mean_temperature_c | non-monotone wiggle (4 sign changes) a regulator would query; 38% of sampled ICE curves disagree with the model's own average direction |
| motor | bi_frequency | start_month | model direction contradicts the observed one-way relativity on 2021-23 |
| motor | other_perils | make_model | non-monotone wiggle (6 sign changes) a regulator would query |
| motor | other_perils | engine_size_cc | non-monotone wiggle (5 sign changes) a regulator would query |
| motor | other_perils | seats | 44% of sampled ICE curves disagree with the model's own average direction |
| motor | other_perils | vehicle_colour | non-monotone wiggle (5 sign changes) a regulator would query |
| motor | other_perils | telematics_score | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | other_perils | region | non-monotone wiggle (7 sign changes) a regulator would query |
| motor | other_perils | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | other_perils | freezing_days | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | other_perils | dry_summer_index | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (6 sign changes) a regulator would query |
| motor | other_perils | wildfire_index | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 31% of sampled ICE curves disagree with the model's own average direction; 68% of sampled ICE curves disagree with the actuarial prior direction |
| motor | other_perils | mean_temperature_c | model direction contradicts the observed one-way relativity on 2021-23; 32% of sampled ICE curves disagree with the model's own average direction |
| buildings | lgbm | renewal_count | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | construction_type | non-monotone wiggle (5 sign changes) a regulator would query |
| buildings | lgbm | occupation | non-monotone wiggle (6 sign changes) a regulator would query |
| buildings | lgbm | credit_band | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | years_at_address | non-monotone wiggle (3 sign changes) a regulator would query; 42% of sampled ICE curves disagree with the model's own average direction |
| buildings | lgbm | distance_to_river_km | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | elevation_m | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | soil_type | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | year_built | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | energy_rating | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | wildfire_index | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | rainfall_mm | non-monotone wiggle (3 sign changes) a regulator would query; 73% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | lgbm | mean_temperature_c | non-monotone wiggle (2 sign changes) a regulator would query; 46% of sampled ICE curves disagree with the model's own average direction |
| buildings | lgbm | start_month | 34% of sampled ICE curves disagree with the model's own average direction |
| buildings | ebm | renewal_count | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | ebm | construction_type | non-monotone wiggle (5 sign changes) a regulator would query |
| buildings | ebm | occupation | non-monotone wiggle (9 sign changes) a regulator would query |
| buildings | ebm | credit_band | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | ebm | years_at_address | model direction contradicts the observed one-way relativity on 2021-23 |
| buildings | ebm | distance_to_river_km | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | ebm | energy_rating | non-monotone wiggle (4 sign changes) a regulator would query |
| buildings | ebm | wildfire_index | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | ebm | rainfall_mm | 100% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | ebm | mean_temperature_c | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | discount_applied | model direction contradicts the observed one-way relativity on 2021-23 |
| contents | lgbm | credit_band | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | region | non-monotone wiggle (7 sign changes) a regulator would query |
| contents | lgbm | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | year_built | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | lgbm | energy_rating | non-monotone wiggle (5 sign changes) a regulator would query |
| contents | lgbm | freezing_days | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | lgbm | wildfire_index | model direction opposes the 'up' prior; non-monotone wiggle (2 sign changes) a regulator would query; 95% of sampled ICE curves disagree with the actuarial prior direction |
| contents | ebm | credit_band | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | ebm | region | non-monotone wiggle (5 sign changes) a regulator would query |
| contents | ebm | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | ebm | year_built | non-monotone wiggle (4 sign changes) a regulator would query |
| contents | ebm | property_type | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | ebm | energy_rating | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | ebm | freezing_days | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | ebm | wildfire_index | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 100% of sampled ICE curves disagree with the actuarial prior direction |
| contents | ebm | rainfall_mm | non-monotone wiggle (4 sign changes) a regulator would query |

### motor / bi_frequency

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| ncb_years | down | True | -1.00 | -1.00 | -0.99 | 0.00 | 0.00 | - |
| driver_age | U | False | -0.60 | -0.62 | -0.48 | - | 0.00 | - |
| annual_mileage | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| years_licensed | down | False | -0.33 | -0.33 | -0.40 | 0.06 | 0.06 | non-monotone wiggle (3 sign changes) a regulator would query |
| cover_type | - | False | - | - | - | - | - | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| vehicle_age | - | False | -0.95 | -0.95 | -0.50 | - | 0.24 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| telematics_score | - | False | -0.95 | -0.98 | -0.90 | - | 0.01 | non-monotone wiggle (2 sign changes) a regulator would query |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| credit_band | up | True | - | - | - | - | - | - |
| hist_n_policy | up | False | - | 1.00 | 1.00 | 0.01 | 0.01 | - |
| missed_payments_12m | up | False | - | 1.00 | - | 0.02 | 0.02 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| new_business | - | False | - | - | - | - | - | - |
| licence_points | up | True | 1.00 | 1.00 | - | 0.00 | 0.00 | - |
| vehicle_value_group | up | False | 0.69 | 0.90 | - | 0.20 | 0.20 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| vehicle_colour | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| start_month | - | False | -1.00 | -1.00 | - | - | 0.00 | model direction contradicts the observed one-way relativity on 2021-23 |
| hist_cost_policy | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 0.83 | 0.69 | - | 0.07 | 0.07 | non-monotone wiggle (4 sign changes) a regulator would query |
| wildfire_index | up | False | 0.74 | 0.71 | - | 0.36 | 0.36 | non-monotone wiggle (4 sign changes) a regulator would query; 36% of sampled ICE curves disagree with the model's own average direction; 36% of sampled ICE curves disagree with the actuarial prior direction |
| overnight_parking | - | False | - | - | - | - | - | - |
| make_model | - | False | - | - | - | - | - | non-monotone wiggle (6 sign changes) a regulator would query |
| payment_method | - | False | - | - | - | - | - | - |
| freezing_days | up | False | 0.81 | 0.81 | - | 0.04 | 0.04 | non-monotone wiggle (3 sign changes) a regulator would query |
| engine_size_cc | - | False | 0.17 | 0.17 | - | - | 0.45 | non-monotone wiggle (3 sign changes) a regulator would query; 45% of sampled ICE curves disagree with the model's own average direction |
| body_type | - | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| seats | - | False | 0.50 | 0.50 | - | - | 0.43 | 43% of sampled ICE curves disagree with the model's own average direction |
| fuel_type | - | False | - | - | - | - | - | - |
| named_young_driver | up | False | - | - | - | - | - | - |
| mean_temperature_c | - | False | 0.07 | 0.24 | - | - | 0.38 | non-monotone wiggle (4 sign changes) a regulator would query; 38% of sampled ICE curves disagree with the model's own average direction |
| channel | - | False | - | - | - | - | - | - |
| ncb_years_new | - | True | -1.00 | -1.00 | - | - | 0.00 | - |

### motor / other_perils

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| ncb_years | down | True | -1.00 | -1.00 | -1.00 | 0.00 | 0.00 | - |
| driver_age | U | False | -0.38 | -0.38 | -0.38 | - | 0.01 | - |
| annual_mileage | up | False | 0.94 | 0.94 | 0.94 | 0.00 | 0.00 | - |
| years_licensed | down | False | -0.69 | -0.69 | -0.48 | 0.00 | 0.00 | - |
| cover_type | - | False | - | - | - | - | - | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| vehicle_age | - | False | -1.00 | -1.00 | -1.00 | - | 0.01 | - |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| telematics_score | - | False | -0.95 | -0.98 | -0.98 | - | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| credit_band | up | True | - | - | - | - | - | - |
| hist_n_policy | up | False | - | 1.00 | 1.00 | 0.01 | 0.01 | - |
| missed_payments_12m | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| new_business | - | False | - | - | - | - | - | - |
| licence_points | up | True | 1.00 | 1.00 | - | 0.00 | 0.00 | - |
| vehicle_value_group | up | False | 1.00 | 1.00 | - | 0.01 | 0.01 | - |
| vehicle_colour | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| start_month | - | False | 0.50 | 0.50 | - | - | 0.24 | - |
| hist_cost_policy | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 0.88 | 0.86 | - | 0.28 | 0.28 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (6 sign changes) a regulator would query |
| wildfire_index | up | False | -0.31 | -0.31 | - | 0.68 | 0.31 | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 31% of sampled ICE curves disagree with the model's own average direction; 68% of sampled ICE curves disagree with the actuarial prior direction |
| overnight_parking | - | False | - | - | - | - | - | - |
| make_model | - | False | - | - | - | - | - | non-monotone wiggle (6 sign changes) a regulator would query |
| payment_method | - | False | - | - | - | - | - | - |
| freezing_days | up | False | 0.67 | 0.55 | - | 0.02 | 0.02 | non-monotone wiggle (3 sign changes) a regulator would query |
| engine_size_cc | - | False | 0.69 | 0.74 | - | - | 0.30 | non-monotone wiggle (5 sign changes) a regulator would query |
| body_type | - | False | - | - | - | - | - | - |
| seats | - | False | 0.50 | 0.50 | - | - | 0.44 | 44% of sampled ICE curves disagree with the model's own average direction |
| fuel_type | - | False | - | - | - | - | - | - |
| named_young_driver | up | False | - | - | - | - | - | - |
| mean_temperature_c | - | False | 0.26 | 0.33 | - | - | 0.32 | model direction contradicts the observed one-way relativity on 2021-23; 32% of sampled ICE curves disagree with the model's own average direction |
| channel | - | False | - | - | - | - | - | - |
| ncb_years_new | - | True | -1.00 | -1.00 | - | - | 0.00 | - |

### buildings / lgbm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| flood_zone | up | True | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.69 | -0.69 | -0.62 | 0.04 | 0.04 | non-monotone wiggle (3 sign changes) a regulator would query |
| sum_insured | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| distance_to_river_km | down | False | -0.98 | -0.98 | -1.00 | 0.03 | 0.03 | non-monotone wiggle (2 sign changes) a regulator would query |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| year_built | down | False | -0.95 | -0.95 | -0.98 | 0.01 | 0.01 | non-monotone wiggle (3 sign changes) a regulator would query |
| property_age | up | False | 0.98 | 0.98 | 0.98 | 0.00 | 0.00 | - |
| soil_type | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| occupancy | - | False | - | - | - | - | - | - |
| missed_payments_12m | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 1.00 | 1.00 | 1.00 | 0.06 | 0.06 | - |
| construction_type | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| renewal_count | - | False | -0.98 | -0.98 | - | - | 0.07 | non-monotone wiggle (2 sign changes) a regulator would query |
| channel | - | False | - | - | - | - | - | - |
| rainfall_mm | up | False | -0.19 | -0.05 | - | 0.73 | 0.27 | non-monotone wiggle (3 sign changes) a regulator would query; 73% of sampled ICE curves disagree with the actuarial prior direction |
| payment_method | - | False | - | - | - | - | - | - |
| wind_exposure_index | - | False | -0.93 | -0.93 | - | - | 0.04 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| basement | - | False | - | - | - | - | - | - |
| wildfire_index | up | False | 0.90 | 0.95 | - | 0.07 | 0.07 | non-monotone wiggle (3 sign changes) a regulator would query |
| mean_temperature_c | - | False | 0.19 | 0.21 | - | - | 0.46 | non-monotone wiggle (2 sign changes) a regulator would query; 46% of sampled ICE curves disagree with the model's own average direction |
| listed_building | - | False | - | - | - | - | - | - |
| years_at_address | - | False | -0.12 | -0.12 | - | - | 0.42 | non-monotone wiggle (3 sign changes) a regulator would query; 42% of sampled ICE curves disagree with the model's own average direction |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| occupation | - | False | - | - | - | - | - | non-monotone wiggle (6 sign changes) a regulator would query |
| start_month | - | False | - | -1.00 | - | - | 0.34 | 34% of sampled ICE curves disagree with the model's own average direction |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.00 | - |

### buildings / ebm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| flood_zone | up | False | 0.50 | 0.50 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.98 | -0.98 | -0.93 | 0.00 | 0.00 | - |
| sum_insured | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| distance_to_river_km | down | False | -0.55 | -0.43 | -0.98 | 0.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query |
| prior_claims_5y | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| year_built | down | False | -0.98 | -0.98 | -0.98 | 0.00 | 0.00 | - |
| property_age | up | False | 0.93 | 0.93 | 0.95 | 0.00 | 0.00 | - |
| soil_type | - | False | - | - | - | - | - | - |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (4 sign changes) a regulator would query |
| occupancy | - | False | - | - | - | - | - | - |
| missed_payments_12m | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| construction_type | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| renewal_count | - | False | -0.96 | -0.96 | - | - | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| channel | - | False | - | - | - | - | - | - |
| rainfall_mm | up | False | -0.02 | -0.02 | - | 1.00 | 0.00 | 100% of sampled ICE curves disagree with the actuarial prior direction |
| payment_method | - | False | - | - | - | - | - | - |
| wind_exposure_index | - | False | -0.93 | -0.93 | - | - | 0.00 | - |
| voluntary_excess | down | False | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| basement | - | False | - | - | - | - | - | - |
| wildfire_index | up | False | 0.86 | 0.86 | - | 0.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query |
| mean_temperature_c | - | False | -0.74 | -0.83 | - | - | 0.00 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (2 sign changes) a regulator would query |
| listed_building | - | False | - | - | - | - | - | - |
| years_at_address | - | False | 0.26 | 0.26 | - | - | 0.00 | model direction contradicts the observed one-way relativity on 2021-23 |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| occupation | - | False | - | - | - | - | - | non-monotone wiggle (9 sign changes) a regulator would query |
| start_month | - | False | - | 1.00 | - | - | 0.00 | - |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.00 | - |

### contents / lgbm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| sum_insured | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| flood_zone | up | True | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.83 | -0.83 | -0.93 | 0.00 | 0.00 | - |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | -1.00 | 0.00 | 0.00 | - |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| high_value_items | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| customer_age | - | False | -0.31 | -0.31 | -0.31 | - | 0.11 | - |
| policyholder_age | U | False | -0.31 | -0.31 | -0.24 | - | 0.11 | - |
| freezing_days | up | False | 0.95 | 0.95 | 0.93 | 0.01 | 0.01 | non-monotone wiggle (3 sign changes) a regulator would query |
| payment_method | - | False | - | - | - | - | - | - |
| missed_payments_12m | up | False | - | 1.00 | - | 0.01 | 0.01 | - |
| rainfall_mm | up | False | 0.64 | 0.64 | - | 0.25 | 0.25 | - |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| property_type | - | False | - | - | - | - | - | - |
| wildfire_index | up | False | -0.98 | -0.98 | - | 0.95 | 0.05 | model direction opposes the 'up' prior; non-monotone wiggle (2 sign changes) a regulator would query; 95% of sampled ICE curves disagree with the actuarial prior direction |
| channel | - | False | - | - | - | - | - | - |
| security_level | down | False | - | - | - | - | - | - |
| discount_applied | - | False | -0.50 | -0.50 | - | - | 0.25 | model direction contradicts the observed one-way relativity on 2021-23 |
| works_from_home | - | False | - | - | - | - | - | - |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| solar_panels | - | False | - | - | - | - | - | - |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.11 | - |
| lodgers | up | False | - | - | - | - | - | - |
| trees_within_5m | up | False | - | - | - | - | - | - |
| tenure | - | False | - | - | - | - | - | - |
| bicycle_cover | - | False | - | - | - | - | - | - |
| year_built | down | False | -0.64 | -0.64 | - | 0.19 | 0.19 | non-monotone wiggle (3 sign changes) a regulator would query |

### contents / ebm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| sum_insured | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| flood_zone | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.69 | -0.69 | -0.76 | 0.00 | 0.00 | - |
| prior_claims_5y | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| voluntary_excess | down | False | -0.80 | -0.80 | -1.00 | 0.00 | 0.00 | - |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| high_value_items | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| customer_age | - | False | -0.17 | -0.17 | -0.31 | - | 0.00 | - |
| policyholder_age | U | False | -0.31 | -0.31 | -0.17 | - | 0.00 | - |
| freezing_days | up | False | 0.71 | 0.71 | 0.90 | 0.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query |
| payment_method | - | False | - | - | - | - | - | - |
| missed_payments_12m | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| rainfall_mm | up | False | 0.40 | 0.40 | - | 0.00 | 0.00 | non-monotone wiggle (4 sign changes) a regulator would query |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| property_type | - | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| wildfire_index | up | False | -0.95 | -0.95 | - | 1.00 | 0.00 | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 100% of sampled ICE curves disagree with the actuarial prior direction |
| channel | - | False | - | - | - | - | - | - |
| security_level | down | False | - | - | - | - | - | - |
| discount_applied | - | False | 0.50 | 0.50 | - | - | 0.00 | - |
| works_from_home | - | False | - | - | - | - | - | - |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| solar_panels | - | False | - | - | - | - | - | - |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.00 | - |
| lodgers | up | False | - | - | - | - | - | - |
| trees_within_5m | up | False | - | - | - | - | - | - |
| tenure | - | False | - | - | - | - | - | - |
| bicycle_cover | - | False | - | - | - | - | - | - |
| year_built | down | False | -0.86 | -0.86 | - | 0.00 | 0.00 | non-monotone wiggle (4 sign changes) a regulator would query |

### Interaction sanity check -- flagged pairs (log-rate deviation > log(1.10))

| line | sub-model | pair | max log deviation | flagged |
|---|---|---|---|---|
| motor | bi_frequency | ncb_years x driver_age | 0.063 | no |
| motor | bi_frequency | ncb_years x annual_mileage | 0.036 | no |
| motor | bi_frequency | ncb_years x years_licensed | 0.034 | no |
| motor | bi_frequency | ncb_years x cover_type | 0.004 | no |
| motor | bi_frequency | driver_age x annual_mileage | 0.023 | no |
| motor | bi_frequency | driver_age x years_licensed | 0.035 | no |
| motor | bi_frequency | driver_age x cover_type | 0.011 | no |
| motor | bi_frequency | annual_mileage x years_licensed | 0.020 | no |
| motor | bi_frequency | annual_mileage x cover_type | 0.019 | no |
| motor | bi_frequency | years_licensed x cover_type | 0.009 | no |
| motor | other_perils | ncb_years x driver_age | 0.040 | no |
| motor | other_perils | ncb_years x annual_mileage | 0.038 | no |
| motor | other_perils | ncb_years x years_licensed | 0.029 | no |
| motor | other_perils | ncb_years x cover_type | 0.087 | no |
| motor | other_perils | driver_age x annual_mileage | 0.027 | no |
| motor | other_perils | driver_age x years_licensed | 0.042 | no |
| motor | other_perils | driver_age x cover_type | 0.042 | no |
| motor | other_perils | annual_mileage x years_licensed | 0.008 | no |
| motor | other_perils | annual_mileage x cover_type | 0.052 | no |
| motor | other_perils | years_licensed x cover_type | 0.026 | no |
| buildings | lgbm | flood_zone x elevation_m | 0.099 | YES |
| buildings | lgbm | flood_zone x sum_insured | 0.015 | no |
| buildings | lgbm | flood_zone x distance_to_river_km | 0.057 | no |
| buildings | lgbm | flood_zone x prior_claims_5y | 0.009 | no |
| buildings | lgbm | elevation_m x sum_insured | 0.022 | no |
| buildings | lgbm | elevation_m x distance_to_river_km | 0.060 | no |
| buildings | lgbm | elevation_m x prior_claims_5y | 0.036 | no |
| buildings | lgbm | sum_insured x distance_to_river_km | 0.028 | no |
| buildings | lgbm | sum_insured x prior_claims_5y | 0.038 | no |
| buildings | lgbm | distance_to_river_km x prior_claims_5y | 0.015 | no |
| contents | lgbm | sum_insured x flood_zone | 0.040 | no |
| contents | lgbm | sum_insured x elevation_m | 0.067 | no |
| contents | lgbm | sum_insured x prior_claims_5y | 0.036 | no |
| contents | lgbm | sum_insured x voluntary_excess | 0.030 | no |
| contents | lgbm | flood_zone x elevation_m | 0.132 | YES |
| contents | lgbm | flood_zone x prior_claims_5y | 0.007 | no |
| contents | lgbm | flood_zone x voluntary_excess | 0.011 | no |
| contents | lgbm | elevation_m x prior_claims_5y | 0.014 | no |
| contents | lgbm | elevation_m x voluntary_excess | 0.007 | no |
| contents | lgbm | prior_claims_5y x voluntary_excess | 0.002 | no |

### Surrogate Tweedie GLM per line

| line | R^2 (weighted) | D^2 (Tweedie) vs. full model | intercept rate |
|---|---|---|---|
| motor | 0.637 | 0.863 | 1246.3 |
| buildings | 0.531 | 0.825 | 444.6 |
| contents | 0.467 | 0.818 | 109.1 |

**motor surrogate relativities, grouped by feature (reference band = 1.00):**


*ncb_years* (reference: `(-0.001, 3.0]` = 1.00)

| band | relativity |
|---|---|
| (3.0, 5.0] | 0.68 |
| (5.0, 9.0] | 0.58 |
| (9.0, 13.0] | 0.52 |
| (13.0, 15.0] | 0.49 |

*driver_age* (reference: `(16.999, 27.0]` = 1.00)

| band | relativity |
|---|---|
| (27.0, 40.0] | 0.82 |
| (40.0, 57.0] | 0.74 |
| (57.0, 74.0] | 0.74 |
| (74.0, 91.0] | 0.85 |

*annual_mileage* (reference: `(-0.001, 6800.0]` = 1.00)

| band | relativity |
|---|---|
| (6800.0, 10000.0] | 1.19 |
| (10000.0, 15000.0] | 1.42 |
| (15000.0, 75000.0] | 1.63 |

*years_licensed* (reference: `(-0.001, 6.0]` = 1.00)

| band | relativity |
|---|---|
| (6.0, 19.0] | 0.82 |
| (19.0, 36.0] | 0.75 |
| (36.0, 53.0] | 0.75 |
| (53.0, 87.0] | 0.86 |

*cover_type* (reference: `comprehensive` = 1.00)

| band | relativity |
|---|---|
| third_party_fire_theft | 0.75 |
| third_party_only | 0.68 |

*region* (reference: `R01` = 1.00)

| band | relativity |
|---|---|
| R03 | 1.25 |
| R05 | 0.87 |
| R10 | 1.02 |
| R11 | 1.03 |
| R12 | 1.09 |
| _other_ | 1.01 |

*vehicle_age* (reference: `(-0.001, 5.0]` = 1.00)

| band | relativity |
|---|---|
| (5.0, 7.0] | 0.97 |
| (7.0, 9.0] | 0.91 |
| (9.0, 12.0] | 0.86 |
| (12.0, 28.0] | 0.82 |

*prior_claims_5y* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 7.0] | 1.42 |

*telematics_score* (reference: `(0.999, 47.0]` = 1.00)

| band | relativity |
|---|---|
| (47.0, 56.0] | 0.89 |
| (56.0, 64.0] | 0.83 |
| (64.0, 73.0] | 0.75 |
| (73.0, 100.0] | 0.66 |

*urban_density* (reference: `suburban` = 1.00)

| band | relativity |
|---|---|
| rural | 0.78 |
| suburban | 0.87 |
| urban | 0.91 |

*credit_band* (reference: `B` = 1.00)

| band | relativity |
|---|---|
| B | 0.99 |
| C | 0.99 |
| D | 1.07 |
| E | 1.29 |
| _other_ | 1.11 |

**buildings surrogate relativities, grouped by feature (reference band = 1.00):**


*flood_zone* (reference: `(0.999, 2.0]` = 1.00)

| band | relativity |
|---|---|
| (2.0, 5.0] | 1.65 |

*elevation_m* (reference: `(-0.001, 3.2]` = 1.00)

| band | relativity |
|---|---|
| (3.2, 6.3] | 0.63 |
| (6.3, 10.3] | 0.62 |
| (10.3, 17.1] | 0.62 |
| (17.1, 196.9] | 0.60 |

*sum_insured* (reference: `(38999.999, 161000.0]` = 1.00)

| band | relativity |
|---|---|
| (161000.0, 210000.0] | 1.38 |
| (210000.0, 257000.0] | 1.74 |
| (257000.0, 319000.0] | 1.94 |
| (319000.0, 588000000.0] | 2.33 |

*distance_to_river_km* (reference: `(0.009000000000000001, 0.82]` = 1.00)

| band | relativity |
|---|---|
| (0.82, 1.78] | 0.73 |
| (1.78, 3.06] | 0.74 |
| (3.06, 5.06] | 0.75 |
| (5.06, 30.0] | 0.75 |

*prior_claims_5y* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 5.0] | 1.50 |

*year_built* (reference: `(1861.999, 1917.0]` = 1.00)

| band | relativity |
|---|---|
| (1917.0, 1952.0] | 0.92 |
| (1952.0, 1977.0] | 0.88 |
| (1977.0, 2001.0] | 0.86 |
| (2001.0, 2021.0] | 0.87 |

*property_age* (reference: `(2.999, 23.0]` = 1.00)

| band | relativity |
|---|---|
| (23.0, 47.0] | 1.03 |
| (47.0, 72.0] | 1.07 |
| (72.0, 107.0] | 1.19 |
| (107.0, 162.0] | 1.12 |

*soil_type* (reference: `clay` = 1.00)

| band | relativity |
|---|---|
| chalk | 1.01 |
| clay | 1.15 |
| loam | 1.01 |
| rock | 1.00 |
| sand | 1.03 |

*energy_rating* (reference: `C` = 1.00)

| band | relativity |
|---|---|
| B | 0.98 |
| C | 1.03 |
| D | 1.01 |
| E | 1.19 |
| F | 1.25 |
| _other_ | 1.21 |

*occupancy* (reference: `owner_occupied` = 1.00)

| band | relativity |
|---|---|
| owner_occupied | 0.78 |

*dry_summer_index* (reference: `(-0.001, 0.403]` = 1.00)

| band | relativity |
|---|---|
| (0.403, 0.578] | 1.08 |
| (0.578, 0.703] | 1.07 |
| (0.703, 0.798] | 1.12 |
| (0.798, 1.0] | 1.17 |

**contents surrogate relativities, grouped by feature (reference band = 1.00):**


*sum_insured* (reference: `(3999.999, 24800.0]` = 1.00)

| band | relativity |
|---|---|
| (24800.0, 35000.0] | 1.33 |
| (35000.0, 45700.0] | 1.64 |
| (45700.0, 60700.0] | 2.07 |
| (60700.0, 152600000.0] | 2.53 |

*flood_zone* (reference: `(0.999, 2.0]` = 1.00)

| band | relativity |
|---|---|
| (2.0, 5.0] | 1.87 |

*elevation_m* (reference: `(-0.001, 3.2]` = 1.00)

| band | relativity |
|---|---|
| (3.2, 6.3] | 0.67 |
| (6.3, 10.3] | 0.65 |
| (10.3, 17.1] | 0.64 |
| (17.1, 196.9] | 0.65 |

*prior_claims_5y* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 5.0] | 1.41 |

*voluntary_excess* (reference: `(-0.001, 250.0]` = 1.00)

| band | relativity |
|---|---|
| (250.0, 500.0] | 0.75 |
| (500.0, 1000.0] | 0.77 |

*urban_density* (reference: `suburban` = 1.00)

| band | relativity |
|---|---|
| rural | 0.77 |
| suburban | 0.76 |
| urban | 0.86 |

*high_value_items* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 2.0] | 1.06 |
| (2.0, 5.0] | 1.18 |

*region* (reference: `R01` = 1.00)

| band | relativity |
|---|---|
| R03 | 1.09 |
| R05 | 0.95 |
| R10 | 0.95 |
| R11 | 1.08 |
| R12 | 1.05 |
| _other_ | 0.98 |

*accidental_damage_cover* (reference: `False` = 1.00)

| band | relativity |
|---|---|
| True | 1.19 |

*customer_age* (reference: `(17.072, 27.515]` = 1.00)

| band | relativity |
|---|---|
| (27.515, 40.539] | 0.90 |
| (40.539, 57.358] | 0.83 |
| (57.358, 74.146] | 0.86 |
| (74.146, 91.934] | 0.89 |

*policyholder_age* (reference: `(16.999, 27.0]` = 1.00)

| band | relativity |
|---|---|
| (27.0, 40.0] | 0.90 |
| (40.0, 57.0] | 0.89 |
| (57.0, 74.0] | 0.92 |
| (74.0, 91.0] | 0.98 |

*freezing_days* (reference: `(-0.001, 17.0]` = 1.00)

| band | relativity |
|---|---|
| (17.0, 25.0] | 0.96 |
| (25.0, 43.0] | 0.99 |
| (43.0, 67.0] | 1.17 |
| (67.0, 104.0] | 1.08 |
