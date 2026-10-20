
### Consistency check -- captured models vs. the model's own fit

- `fold_B.pred_d2` (pooled D^2 recomputed from the captured sub-models): 0.1415
- `model_card.json['pooled_d2']` (same metric, scripts/model_card.py's own run): 0.1415
- dataset fingerprint this pack used: 2e6c0e5ff80c
- dataset fingerprint model_card.json used: 2e6c0e5ff80c

Exact match (diff 0.00e+00): the sub-models captured by this pack are the ones that produced the model's real predictions, on the same dataset fingerprint as model_card.json.

### All flagged predictors (fold B)

| line | sub-model | predictor | flags |
|---|---|---|---|
| motor | bi_frequency | years_licensed | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | vehicle_value_group | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | vehicle_age | model direction contradicts the observed one-way relativity on 2021-23 |
| motor | bi_frequency | make_model | non-monotone wiggle (10 sign changes) a regulator would query |
| motor | bi_frequency | engine_size_cc | non-monotone wiggle (5 sign changes) a regulator would query; 44% of sampled ICE curves disagree with the model's own average direction |
| motor | bi_frequency | body_type | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | seats | 37% of sampled ICE curves disagree with the model's own average direction |
| motor | bi_frequency | vehicle_colour | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | bi_frequency | telematics_score | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | credit_band | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | region | non-monotone wiggle (7 sign changes) a regulator would query |
| motor | bi_frequency | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | bi_frequency | distance_to_river_km | non-monotone wiggle (5 sign changes) a regulator would query; 39% of sampled ICE curves disagree with the model's own average direction; 39% of sampled ICE curves disagree with the actuarial prior direction |
| motor | bi_frequency | dry_summer_index | non-monotone wiggle (4 sign changes) a regulator would query |
| motor | bi_frequency | wildfire_index | non-monotone wiggle (4 sign changes) a regulator would query; 33% of sampled ICE curves disagree with the model's own average direction; 33% of sampled ICE curves disagree with the actuarial prior direction |
| motor | bi_frequency | mean_temperature_c | non-monotone wiggle (4 sign changes) a regulator would query; 34% of sampled ICE curves disagree with the model's own average direction |
| motor | bi_frequency | start_month | model direction contradicts the observed one-way relativity on 2021-23 |
| motor | other_perils | make_model | non-monotone wiggle (6 sign changes) a regulator would query |
| motor | other_perils | engine_size_cc | non-monotone wiggle (5 sign changes) a regulator would query |
| motor | other_perils | seats | 47% of sampled ICE curves disagree with the model's own average direction |
| motor | other_perils | vehicle_colour | non-monotone wiggle (5 sign changes) a regulator would query |
| motor | other_perils | telematics_score | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | other_perils | region | non-monotone wiggle (7 sign changes) a regulator would query |
| motor | other_perils | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| motor | other_perils | distance_to_river_km | model direction opposes the 'down' prior; non-monotone wiggle (4 sign changes) a regulator would query; 40% of sampled ICE curves disagree with the model's own average direction; 59% of sampled ICE curves disagree with the actuarial prior direction |
| motor | other_perils | freezing_days | non-monotone wiggle (3 sign changes) a regulator would query |
| motor | other_perils | dry_summer_index | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (4 sign changes) a regulator would query |
| motor | other_perils | wildfire_index | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 31% of sampled ICE curves disagree with the model's own average direction; 68% of sampled ICE curves disagree with the actuarial prior direction |
| motor | other_perils | mean_temperature_c | 44% of sampled ICE curves disagree with the model's own average direction |
| buildings | lgbm | renewal_count | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | sum_insured | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | construction_type | non-monotone wiggle (5 sign changes) a regulator would query |
| buildings | lgbm | occupation | non-monotone wiggle (4 sign changes) a regulator would query |
| buildings | lgbm | credit_band | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | years_at_address | non-monotone wiggle (4 sign changes) a regulator would query; 43% of sampled ICE curves disagree with the model's own average direction |
| buildings | lgbm | distance_to_river_km | non-monotone wiggle (5 sign changes) a regulator would query; 37% of sampled ICE curves disagree with the model's own average direction; 37% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | lgbm | elevation_m | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | soil_type | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | year_built | non-monotone wiggle (5 sign changes) a regulator would query |
| buildings | lgbm | energy_rating | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | lgbm | dry_summer_index | non-monotone wiggle (5 sign changes) a regulator would query |
| buildings | lgbm | wildfire_index | non-monotone wiggle (3 sign changes) a regulator would query; 34% of sampled ICE curves disagree with the model's own average direction; 34% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | lgbm | rainfall_mm | model direction opposes the 'up' prior; model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (5 sign changes) a regulator would query; 71% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | lgbm | mean_temperature_c | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | lgbm | property_age | non-monotone wiggle (4 sign changes) a regulator would query |
| buildings | lgbm | hist_n_customer | model direction opposes the 'up' prior; 76% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | ebm | renewal_count | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | ebm | construction_type | non-monotone wiggle (7 sign changes) a regulator would query |
| buildings | ebm | occupation | non-monotone wiggle (9 sign changes) a regulator would query |
| buildings | ebm | credit_band | non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | ebm | years_at_address | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (2 sign changes) a regulator would query |
| buildings | ebm | distance_to_river_km | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | ebm | energy_rating | non-monotone wiggle (4 sign changes) a regulator would query |
| buildings | ebm | wildfire_index | non-monotone wiggle (3 sign changes) a regulator would query |
| buildings | ebm | rainfall_mm | non-monotone wiggle (3 sign changes) a regulator would query; 100% of sampled ICE curves disagree with the actuarial prior direction |
| buildings | ebm | mean_temperature_c | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | discount_applied | 35% of sampled ICE curves disagree with the model's own average direction |
| contents | lgbm | policyholder_age | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | sum_insured | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | credit_band | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | lgbm | region | non-monotone wiggle (7 sign changes) a regulator would query |
| contents | lgbm | urban_density | non-monotone wiggle (2 sign changes) a regulator would query |
| contents | lgbm | year_built | non-monotone wiggle (4 sign changes) a regulator would query; 35% of sampled ICE curves disagree with the model's own average direction; 35% of sampled ICE curves disagree with the actuarial prior direction |
| contents | lgbm | energy_rating | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | lgbm | freezing_days | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | lgbm | wildfire_index | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 73% of sampled ICE curves disagree with the actuarial prior direction |
| contents | lgbm | rainfall_mm | non-monotone wiggle (3 sign changes) a regulator would query |
| contents | ebm | credit_band | non-monotone wiggle (3 sign changes) a regulator would query |
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
| driver_age | U | False | -0.62 | -0.50 | -0.48 | - | 0.00 | - |
| annual_mileage | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| years_licensed | down | False | -0.33 | -0.33 | -0.36 | 0.05 | 0.05 | non-monotone wiggle (3 sign changes) a regulator would query |
| cover_type | - | False | - | - | - | - | - | - |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| vehicle_age | - | False | -0.98 | -0.98 | -0.67 | - | 0.23 | model direction contradicts the observed one-way relativity on 2021-23 |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| telematics_score | - | False | -0.98 | -0.98 | -0.90 | - | 0.01 | non-monotone wiggle (2 sign changes) a regulator would query |
| vehicle_value_group | up | False | 0.60 | 0.62 | 0.90 | 0.22 | 0.22 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| missed_payments_12m | up | False | - | 1.00 | 1.00 | 0.02 | 0.02 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| licence_points | up | True | 1.00 | 1.00 | - | 0.00 | 0.00 | - |
| vehicle_colour | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| hist_n_policy | up | False | - | 1.00 | - | 0.01 | 0.01 | - |
| dry_summer_index | up | False | 0.74 | 0.62 | - | 0.08 | 0.08 | non-monotone wiggle (4 sign changes) a regulator would query |
| hist_cost_policy | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| wildfire_index | up | False | 0.74 | 0.74 | - | 0.33 | 0.33 | non-monotone wiggle (4 sign changes) a regulator would query; 33% of sampled ICE curves disagree with the model's own average direction; 33% of sampled ICE curves disagree with the actuarial prior direction |
| freezing_days | up | False | 0.95 | 0.93 | - | 0.03 | 0.03 | - |
| make_model | - | False | - | - | - | - | - | non-monotone wiggle (10 sign changes) a regulator would query |
| body_type | - | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| overnight_parking | - | False | - | - | - | - | - | - |
| engine_size_cc | - | False | -0.02 | -0.02 | - | - | 0.44 | non-monotone wiggle (5 sign changes) a regulator would query; 44% of sampled ICE curves disagree with the model's own average direction |
| payment_method | - | False | - | - | - | - | - | - |
| distance_to_river_km | down | False | -0.24 | -0.24 | - | 0.39 | 0.39 | non-monotone wiggle (5 sign changes) a regulator would query; 39% of sampled ICE curves disagree with the model's own average direction; 39% of sampled ICE curves disagree with the actuarial prior direction |
| seats | - | False | 0.50 | 0.50 | - | - | 0.37 | 37% of sampled ICE curves disagree with the model's own average direction |
| fuel_type | - | False | - | - | - | - | - | - |
| start_month | - | False | -1.00 | -1.00 | - | - | 0.00 | model direction contradicts the observed one-way relativity on 2021-23 |
| named_young_driver | up | False | - | - | - | - | - | - |
| mean_temperature_c | - | False | 0.33 | 0.33 | - | - | 0.34 | non-monotone wiggle (4 sign changes) a regulator would query; 34% of sampled ICE curves disagree with the model's own average direction |
| channel | - | False | - | - | - | - | - | - |

### motor / other_perils

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| ncb_years | down | True | -1.00 | -1.00 | -1.00 | 0.00 | 0.00 | - |
| driver_age | U | False | -0.38 | -0.38 | -0.48 | - | 0.00 | - |
| annual_mileage | up | False | 0.94 | 0.94 | 0.94 | 0.00 | 0.00 | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| years_licensed | down | False | -0.69 | -0.69 | -0.38 | 0.00 | 0.00 | - |
| cover_type | - | False | - | - | - | - | - | - |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| vehicle_age | - | False | -1.00 | -1.00 | -1.00 | - | 0.01 | - |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| telematics_score | - | False | -0.98 | -0.98 | -1.00 | - | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| vehicle_value_group | up | False | 1.00 | 1.00 | 1.00 | 0.01 | 0.01 | - |
| missed_payments_12m | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| credit_band | up | False | - | - | - | - | - | - |
| licence_points | up | True | 1.00 | 1.00 | - | 0.00 | 0.00 | - |
| vehicle_colour | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| hist_n_policy | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 0.95 | 0.88 | - | 0.27 | 0.27 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (4 sign changes) a regulator would query |
| hist_cost_policy | up | False | - | 1.00 | - | 0.01 | 0.01 | - |
| wildfire_index | up | False | -0.48 | -0.31 | - | 0.68 | 0.31 | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 31% of sampled ICE curves disagree with the model's own average direction; 68% of sampled ICE curves disagree with the actuarial prior direction |
| freezing_days | up | False | 0.67 | 0.55 | - | 0.04 | 0.04 | non-monotone wiggle (3 sign changes) a regulator would query |
| make_model | - | False | - | - | - | - | - | non-monotone wiggle (6 sign changes) a regulator would query |
| body_type | - | False | - | - | - | - | - | - |
| overnight_parking | - | False | - | - | - | - | - | - |
| engine_size_cc | - | False | 0.67 | 0.69 | - | - | 0.28 | non-monotone wiggle (5 sign changes) a regulator would query |
| payment_method | - | False | - | - | - | - | - | - |
| distance_to_river_km | down | False | 0.40 | 0.48 | - | 0.59 | 0.40 | model direction opposes the 'down' prior; non-monotone wiggle (4 sign changes) a regulator would query; 40% of sampled ICE curves disagree with the model's own average direction; 59% of sampled ICE curves disagree with the actuarial prior direction |
| seats | - | False | 0.50 | 0.50 | - | - | 0.47 | 47% of sampled ICE curves disagree with the model's own average direction |
| fuel_type | - | False | - | - | - | - | - | - |
| start_month | - | False | 1.00 | 1.00 | - | - | 0.02 | - |
| named_young_driver | up | False | - | - | - | - | - | - |
| mean_temperature_c | - | False | 0.33 | 0.17 | - | - | 0.44 | 44% of sampled ICE curves disagree with the model's own average direction |
| channel | - | False | - | - | - | - | - | - |

### buildings / lgbm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| flood_zone | up | True | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| sum_insured | up | False | 1.00 | 0.98 | 0.98 | 0.00 | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| elevation_m | down | False | -0.55 | -0.55 | -0.69 | 0.17 | 0.17 | non-monotone wiggle (3 sign changes) a regulator would query |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 0.79 | 0.88 | 0.83 | 0.08 | 0.08 | non-monotone wiggle (5 sign changes) a regulator would query |
| construction_type | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| soil_type | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| occupancy | - | False | - | - | - | - | - | - |
| wildfire_index | up | False | 0.69 | 0.74 | 0.75 | 0.34 | 0.34 | non-monotone wiggle (3 sign changes) a regulator would query; 34% of sampled ICE curves disagree with the model's own average direction; 34% of sampled ICE curves disagree with the actuarial prior direction |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| voluntary_excess | down | True | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| channel | - | False | - | - | - | - | - | - |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.04 | - |
| year_built | down | False | -0.62 | -0.57 | - | 0.13 | 0.13 | non-monotone wiggle (5 sign changes) a regulator would query |
| years_at_address | - | False | -0.19 | -0.05 | - | - | 0.43 | non-monotone wiggle (4 sign changes) a regulator would query; 43% of sampled ICE curves disagree with the model's own average direction |
| missed_payments_12m | up | False | - | 1.00 | - | 0.01 | 0.01 | - |
| mean_temperature_c | - | False | -0.86 | -0.79 | - | - | 0.09 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (3 sign changes) a regulator would query |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| property_age | up | False | 0.88 | 0.90 | - | 0.07 | 0.07 | non-monotone wiggle (4 sign changes) a regulator would query |
| listed_building | - | False | - | - | - | - | - | - |
| rainfall_mm | up | False | -0.33 | -0.33 | - | 0.71 | 0.29 | model direction opposes the 'up' prior; model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (5 sign changes) a regulator would query; 71% of sampled ICE curves disagree with the actuarial prior direction |
| wind_exposure_index | - | False | -0.98 | -0.98 | - | - | 0.16 | - |
| payment_method | - | False | - | - | - | - | - | - |
| start_month | - | False | - | -1.00 | - | - | 0.13 | - |
| hist_n_customer | up | False | - | -1.00 | - | 0.76 | 0.24 | model direction opposes the 'up' prior; 76% of sampled ICE curves disagree with the actuarial prior direction |
| basement | - | False | - | - | - | - | - | - |
| distance_to_river_km | down | False | -0.26 | -0.62 | - | 0.37 | 0.37 | non-monotone wiggle (5 sign changes) a regulator would query; 37% of sampled ICE curves disagree with the model's own average direction; 37% of sampled ICE curves disagree with the actuarial prior direction |
| renewal_count | - | False | -0.98 | -0.98 | - | - | 0.10 | non-monotone wiggle (2 sign changes) a regulator would query |
| occupation | - | False | - | - | - | - | - | non-monotone wiggle (4 sign changes) a regulator would query |

### buildings / ebm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| flood_zone | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| sum_insured | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.93 | -0.93 | -0.93 | 0.00 | 0.00 | - |
| prior_claims_5y | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| dry_summer_index | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| construction_type | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (4 sign changes) a regulator would query |
| soil_type | - | False | - | - | - | - | - | - |
| occupancy | - | False | - | - | - | - | - | - |
| wildfire_index | up | False | 0.86 | 0.86 | 0.82 | 0.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| voluntary_excess | down | False | -1.00 | -1.00 | - | 0.00 | 0.00 | - |
| channel | - | False | - | - | - | - | - | - |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.00 | - |
| year_built | down | False | -0.98 | -0.98 | - | 0.00 | 0.00 | - |
| years_at_address | - | False | 0.36 | 0.36 | - | - | 0.00 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (2 sign changes) a regulator would query |
| missed_payments_12m | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| mean_temperature_c | - | False | -0.95 | -0.95 | - | - | 0.00 | model direction contradicts the observed one-way relativity on 2021-23; non-monotone wiggle (2 sign changes) a regulator would query |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| property_age | up | False | 0.93 | 0.93 | - | 0.00 | 0.00 | - |
| listed_building | - | False | - | - | - | - | - | - |
| rainfall_mm | up | False | -0.05 | -0.05 | - | 1.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query; 100% of sampled ICE curves disagree with the actuarial prior direction |
| wind_exposure_index | - | False | -0.93 | -0.93 | - | - | 0.00 | - |
| payment_method | - | False | - | - | - | - | - | - |
| start_month | - | False | - | 1.00 | - | - | 0.00 | - |
| hist_n_customer | up | False | - | 1.00 | - | 0.00 | 0.00 | - |
| basement | - | False | - | - | - | - | - | - |
| distance_to_river_km | down | False | -0.50 | -0.50 | - | 0.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query |
| renewal_count | - | False | -0.96 | -0.96 | - | - | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| occupation | - | False | - | - | - | - | - | non-monotone wiggle (9 sign changes) a regulator would query |

### contents / lgbm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| sum_insured | up | False | 0.98 | 0.98 | 0.98 | 0.00 | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| flood_zone | up | True | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.64 | -0.64 | -0.81 | 0.23 | 0.23 | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (7 sign changes) a regulator would query |
| prior_claims_5y | up | True | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| voluntary_excess | down | True | -1.00 | -1.00 | -1.00 | 0.00 | 0.00 | - |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| high_value_items | up | False | 1.00 | 1.00 | 0.80 | 0.08 | 0.08 | - |
| customer_age | - | False | -0.31 | -0.31 | -0.31 | - | 0.17 | - |
| missed_payments_12m | up | False | - | 1.00 | 1.00 | 0.03 | 0.03 | - |
| freezing_days | up | False | 0.88 | 0.79 | 0.81 | 0.02 | 0.02 | non-monotone wiggle (3 sign changes) a regulator would query |
| rainfall_mm | up | False | 0.24 | 0.31 | - | 0.28 | 0.28 | non-monotone wiggle (3 sign changes) a regulator would query |
| security_level | down | False | - | - | - | - | - | - |
| works_from_home | - | False | - | - | - | - | - | - |
| property_type | - | False | - | - | - | - | - | - |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| discount_applied | - | False | 0.50 | 0.50 | - | - | 0.35 | 35% of sampled ICE curves disagree with the model's own average direction |
| bedrooms | - | False | 0.90 | 0.90 | - | - | 0.28 | - |
| wildfire_index | up | False | -0.62 | -0.55 | - | 0.73 | 0.26 | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 73% of sampled ICE curves disagree with the actuarial prior direction |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| trees_within_5m | up | False | - | - | - | - | - | - |
| lodgers | up | False | - | - | - | - | - | - |
| channel | - | False | - | - | - | - | - | - |
| bicycle_cover | - | False | - | - | - | - | - | - |
| solar_panels | - | False | - | - | - | - | - | - |
| policyholder_age | U | False | 0.29 | 0.00 | - | - | 0.00 | non-monotone wiggle (2 sign changes) a regulator would query |
| tenure | - | False | - | - | - | - | - | - |
| payment_method | - | False | - | - | - | - | - | - |
| year_built | down | False | -0.81 | -0.64 | - | 0.35 | 0.35 | non-monotone wiggle (4 sign changes) a regulator would query; 35% of sampled ICE curves disagree with the model's own average direction; 35% of sampled ICE curves disagree with the actuarial prior direction |

### contents / ebm

| predictor | prior | monotone | Spearman ALE | Spearman PD | Spearman A | ICE viol (prior) | ICE viol (model) | flags |
|---|---|---|---|---|---|---|---|---|
| sum_insured | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| flood_zone | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| elevation_m | down | False | -0.69 | -0.69 | -0.74 | 0.00 | 0.00 | - |
| region | - | False | - | - | - | - | - | non-monotone wiggle (5 sign changes) a regulator would query |
| prior_claims_5y | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| voluntary_excess | down | False | -0.80 | -0.80 | -0.80 | 0.00 | 0.00 | - |
| accidental_damage_cover | - | False | - | - | - | - | - | - |
| urban_density | up | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| high_value_items | up | False | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| customer_age | - | False | -0.17 | -0.17 | -0.31 | - | 0.00 | - |
| missed_payments_12m | up | False | - | 1.00 | 1.00 | 0.00 | 0.00 | - |
| freezing_days | up | False | 0.71 | 0.71 | 0.90 | 0.00 | 0.00 | non-monotone wiggle (3 sign changes) a regulator would query |
| rainfall_mm | up | False | 0.40 | 0.40 | - | 0.00 | 0.00 | non-monotone wiggle (4 sign changes) a regulator would query |
| security_level | down | False | - | - | - | - | - | - |
| works_from_home | - | False | - | - | - | - | - | - |
| property_type | - | False | - | - | - | - | - | non-monotone wiggle (2 sign changes) a regulator would query |
| credit_band | up | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| discount_applied | - | False | 0.50 | 0.50 | - | - | 0.00 | - |
| bedrooms | - | False | 1.00 | 1.00 | - | - | 0.00 | - |
| wildfire_index | up | False | -0.95 | -0.95 | - | 1.00 | 0.00 | model direction opposes the 'up' prior; non-monotone wiggle (3 sign changes) a regulator would query; 100% of sampled ICE curves disagree with the actuarial prior direction |
| energy_rating | - | False | - | - | - | - | - | non-monotone wiggle (3 sign changes) a regulator would query |
| trees_within_5m | up | False | - | - | - | - | - | - |
| lodgers | up | False | - | - | - | - | - | - |
| channel | - | False | - | - | - | - | - | - |
| bicycle_cover | - | False | - | - | - | - | - | - |
| solar_panels | - | False | - | - | - | - | - | - |
| policyholder_age | U | False | -0.31 | -0.31 | - | - | 0.00 | - |
| tenure | - | False | - | - | - | - | - | - |
| payment_method | - | False | - | - | - | - | - | - |
| year_built | down | False | -0.86 | -0.86 | - | 0.00 | 0.00 | non-monotone wiggle (4 sign changes) a regulator would query |

### Interaction sanity check -- flagged pairs (log-rate deviation > log(1.10))

| line | sub-model | pair | max log deviation | flagged |
|---|---|---|---|---|
| motor | bi_frequency | ncb_years x driver_age | 0.067 | no |
| motor | bi_frequency | ncb_years x annual_mileage | 0.034 | no |
| motor | bi_frequency | ncb_years x region | 0.050 | no |
| motor | bi_frequency | ncb_years x years_licensed | 0.031 | no |
| motor | bi_frequency | driver_age x annual_mileage | 0.022 | no |
| motor | bi_frequency | driver_age x region | 0.045 | no |
| motor | bi_frequency | driver_age x years_licensed | 0.043 | no |
| motor | bi_frequency | annual_mileage x region | 0.019 | no |
| motor | bi_frequency | annual_mileage x years_licensed | 0.021 | no |
| motor | bi_frequency | region x years_licensed | 0.030 | no |
| motor | other_perils | ncb_years x driver_age | 0.040 | no |
| motor | other_perils | ncb_years x annual_mileage | 0.036 | no |
| motor | other_perils | ncb_years x region | 0.059 | no |
| motor | other_perils | ncb_years x years_licensed | 0.021 | no |
| motor | other_perils | driver_age x annual_mileage | 0.024 | no |
| motor | other_perils | driver_age x region | 0.069 | no |
| motor | other_perils | driver_age x years_licensed | 0.037 | no |
| motor | other_perils | annual_mileage x region | 0.022 | no |
| motor | other_perils | annual_mileage x years_licensed | 0.012 | no |
| motor | other_perils | region x years_licensed | 0.039 | no |
| buildings | lgbm | flood_zone x sum_insured | 0.017 | no |
| buildings | lgbm | flood_zone x elevation_m | 0.094 | no |
| buildings | lgbm | flood_zone x prior_claims_5y | 0.007 | no |
| buildings | lgbm | flood_zone x dry_summer_index | 0.016 | no |
| buildings | lgbm | sum_insured x elevation_m | 0.058 | no |
| buildings | lgbm | sum_insured x prior_claims_5y | 0.039 | no |
| buildings | lgbm | sum_insured x dry_summer_index | 0.015 | no |
| buildings | lgbm | elevation_m x prior_claims_5y | 0.024 | no |
| buildings | lgbm | elevation_m x dry_summer_index | 0.066 | no |
| buildings | lgbm | prior_claims_5y x dry_summer_index | 0.028 | no |
| contents | lgbm | sum_insured x flood_zone | 0.033 | no |
| contents | lgbm | sum_insured x elevation_m | 0.091 | no |
| contents | lgbm | sum_insured x region | 0.102 | YES |
| contents | lgbm | sum_insured x prior_claims_5y | 0.034 | no |
| contents | lgbm | flood_zone x elevation_m | 0.145 | YES |
| contents | lgbm | flood_zone x region | 0.031 | no |
| contents | lgbm | flood_zone x prior_claims_5y | 0.007 | no |
| contents | lgbm | elevation_m x region | 0.068 | no |
| contents | lgbm | elevation_m x prior_claims_5y | 0.011 | no |
| contents | lgbm | region x prior_claims_5y | 0.015 | no |

### Surrogate Tweedie GLM per line

| line | R^2 (weighted) | D^2 (Tweedie) vs. full model | intercept rate |
|---|---|---|---|
| motor | 0.615 | 0.854 | 1202.8 |
| buildings | 0.445 | 0.772 | 291.9 |
| contents | 0.395 | 0.782 | 107.5 |

**motor surrogate relativities, grouped by feature (reference band = 1.00):**


*ncb_years* (reference: `(-0.001, 3.0]` = 1.00)

| band | relativity |
|---|---|
| (3.0, 5.0] | 0.68 |
| (5.0, 9.0] | 0.57 |
| (9.0, 13.0] | 0.52 |
| (13.0, 15.0] | 0.50 |

*driver_age* (reference: `(16.999, 27.0]` = 1.00)

| band | relativity |
|---|---|
| (27.0, 40.0] | 0.81 |
| (40.0, 57.0] | 0.73 |
| (57.0, 74.0] | 0.73 |
| (74.0, 91.0] | 0.83 |

*annual_mileage* (reference: `(-0.001, 6800.0]` = 1.00)

| band | relativity |
|---|---|
| (6800.0, 10000.0] | 1.19 |
| (10000.0, 15000.0] | 1.42 |
| (15000.0, 75000.0] | 1.62 |

*region* (reference: `R01` = 1.00)

| band | relativity |
|---|---|
| R03 | 1.27 |
| R05 | 0.87 |
| R10 | 1.02 |
| R11 | 1.05 |
| R12 | 1.10 |
| _other_ | 1.02 |

*years_licensed* (reference: `(-0.001, 6.0]` = 1.00)

| band | relativity |
|---|---|
| (6.0, 19.0] | 0.83 |
| (19.0, 36.0] | 0.76 |
| (36.0, 53.0] | 0.75 |
| (53.0, 87.0] | 0.86 |

*cover_type* (reference: `comprehensive` = 1.00)

| band | relativity |
|---|---|
| third_party_fire_theft | 0.75 |
| third_party_only | 0.69 |

*prior_claims_5y* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 7.0] | 1.41 |

*vehicle_age* (reference: `(-0.001, 5.0]` = 1.00)

| band | relativity |
|---|---|
| (5.0, 7.0] | 0.98 |
| (7.0, 9.0] | 0.92 |
| (9.0, 12.0] | 0.87 |
| (12.0, 28.0] | 0.83 |

*urban_density* (reference: `suburban` = 1.00)

| band | relativity |
|---|---|
| rural | 0.79 |
| suburban | 0.88 |
| urban | 0.91 |

*telematics_score* (reference: `(0.999, 47.0]` = 1.00)

| band | relativity |
|---|---|
| (47.0, 56.0] | 0.89 |
| (56.0, 64.0] | 0.84 |
| (64.0, 73.0] | 0.75 |
| (73.0, 100.0] | 0.66 |

*vehicle_value_group* (reference: `(0.999, 8.0]` = 1.00)

| band | relativity |
|---|---|
| (8.0, 10.0] | 1.03 |
| (10.0, 12.0] | 1.07 |
| (12.0, 14.0] | 1.11 |
| (14.0, 20.0] | 1.17 |

**buildings surrogate relativities, grouped by feature (reference band = 1.00):**


*flood_zone* (reference: `(0.999, 2.0]` = 1.00)

| band | relativity |
|---|---|
| (2.0, 5.0] | 1.96 |

*sum_insured* (reference: `(38999.999, 161000.0]` = 1.00)

| band | relativity |
|---|---|
| (161000.0, 210000.0] | 1.40 |
| (210000.0, 257000.0] | 1.74 |
| (257000.0, 319000.0] | 1.94 |
| (319000.0, 588000000.0] | 2.35 |

*elevation_m* (reference: `(-0.001, 3.2]` = 1.00)

| band | relativity |
|---|---|
| (3.2, 6.3] | 0.62 |
| (6.3, 10.3] | 0.60 |
| (10.3, 17.1] | 0.61 |
| (17.1, 196.9] | 0.59 |

*prior_claims_5y* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 5.0] | 1.59 |

*dry_summer_index* (reference: `(-0.001, 0.403]` = 1.00)

| band | relativity |
|---|---|
| (0.403, 0.578] | 1.07 |
| (0.578, 0.703] | 1.03 |
| (0.703, 0.798] | 1.09 |
| (0.798, 1.0] | 1.11 |

*construction_type* (reference: `Brick` = 1.00)

| band | relativity |
|---|---|
| Brick | 1.01 |
| W | 1.22 |
| _other_ | 0.99 |
| brick | 1.17 |
| non_standard | 1.31 |
| stone | 1.13 |

*energy_rating* (reference: `C` = 1.00)

| band | relativity |
|---|---|
| B | 1.00 |
| C | 1.12 |
| D | 1.18 |
| E | 1.51 |
| F | 1.63 |
| _other_ | 1.53 |

*soil_type* (reference: `clay` = 1.00)

| band | relativity |
|---|---|
| chalk | 0.99 |
| clay | 1.17 |
| loam | 1.00 |
| rock | 0.98 |
| sand | 1.04 |

*occupancy* (reference: `owner_occupied` = 1.00)

| band | relativity |
|---|---|
| owner_occupied | 0.74 |

*wildfire_index* (reference: `(-0.001, 0.043]` = 1.00)

| band | relativity |
|---|---|
| (0.043, 0.125] | 1.00 |
| (0.125, 0.183] | 1.00 |
| (0.183, 0.254] | 1.06 |
| (0.254, 0.582] | 1.05 |

*credit_band* (reference: `B` = 1.00)

| band | relativity |
|---|---|
| B | 0.95 |
| C | 1.03 |
| D | 0.94 |
| E | 1.28 |

**contents surrogate relativities, grouped by feature (reference band = 1.00):**


*sum_insured* (reference: `(3999.999, 24800.0]` = 1.00)

| band | relativity |
|---|---|
| (24800.0, 35000.0] | 1.36 |
| (35000.0, 45700.0] | 1.68 |
| (45700.0, 60700.0] | 2.10 |
| (60700.0, 152600000.0] | 2.64 |

*flood_zone* (reference: `(0.999, 2.0]` = 1.00)

| band | relativity |
|---|---|
| (2.0, 5.0] | 1.97 |

*elevation_m* (reference: `(-0.001, 3.2]` = 1.00)

| band | relativity |
|---|---|
| (3.2, 6.3] | 0.69 |
| (6.3, 10.3] | 0.66 |
| (10.3, 17.1] | 0.65 |
| (17.1, 196.9] | 0.66 |

*region* (reference: `R01` = 1.00)

| band | relativity |
|---|---|
| R03 | 1.14 |
| R05 | 0.93 |
| R10 | 0.95 |
| R11 | 1.08 |
| R12 | 1.04 |
| _other_ | 0.98 |

*prior_claims_5y* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 5.0] | 1.52 |

*voluntary_excess* (reference: `(-0.001, 250.0]` = 1.00)

| band | relativity |
|---|---|
| (250.0, 500.0] | 0.71 |
| (500.0, 1000.0] | 0.76 |

*accidental_damage_cover* (reference: `False` = 1.00)

| band | relativity |
|---|---|
| True | 1.23 |

*urban_density* (reference: `suburban` = 1.00)

| band | relativity |
|---|---|
| rural | 0.72 |
| suburban | 0.73 |
| urban | 0.85 |

*high_value_items* (reference: `(-0.001, 1.0]` = 1.00)

| band | relativity |
|---|---|
| (1.0, 2.0] | 1.08 |
| (2.0, 5.0] | 1.22 |

*customer_age* (reference: `(17.072, 27.515]` = 1.00)

| band | relativity |
|---|---|
| (27.515, 40.539] | 0.79 |
| (40.539, 57.358] | 0.70 |
| (57.358, 74.146] | 0.76 |
| (74.146, 91.934] | 0.88 |

*freezing_days* (reference: `(-0.001, 17.0]` = 1.00)

| band | relativity |
|---|---|
| (17.0, 25.0] | 0.94 |
| (25.0, 43.0] | 0.99 |
| (43.0, 67.0] | 1.19 |
| (67.0, 104.0] | 1.10 |
