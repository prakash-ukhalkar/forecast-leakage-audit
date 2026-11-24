# Audit validation

Structural checks over 180 (dataset, horizon, protocol) cells: flawed protocols flagged as errors 120/126 (error or warning: 126/126); corrected protocols flagged 0/54 (false positives).

| protocol   |   cells |   flagged |   flagged_any | flawed   |   flag_rate | codes                |
|:-----------|--------:|----------:|--------------:|:---------|------------:|:---------------------|
| P0         |      18 |         0 |             0 | False    |    0        | none                 |
| P0V        |      18 |         0 |             0 | False    |    0        | none                 |
| P6B        |      18 |         0 |            18 | False    |    0        | none                 |
| P1         |      18 |        18 |            18 | True     |    1        | SPLIT-ORDER          |
| P2         |      18 |        18 |            18 | True     |    1        | FIT-LEAK             |
| P3         |      18 |        18 |            18 | True     |    1        | SPLIT-GAP            |
| P4         |      18 |        12 |            18 | True     |    0.666667 | FEAT-LOOKAHEAD       |
| P5         |      18 |        18 |            18 | True     |    1        | SELECT-LEAK          |
| P6         |      18 |        18 |            18 | True     |    1        | CV-LEAK, SPLIT-ORDER |
| P6S        |      18 |        18 |            18 | True     |    1        | CV-LEAK, SPLIT-ORDER |

## Black-box look-ahead probe

| dataset                       | builder   |   flagged_columns |   measured_max_reach |   rule_reach |
|:------------------------------|:----------|------------------:|---------------------:|-------------:|
| household_power_consumption   | causal    |                 0 |                    0 |            0 |
| household_power_consumption   | centred   |                 2 |                   29 |           29 |
| appliances_energy_prediction  | causal    |                 0 |                    0 |            0 |
| appliances_energy_prediction  | centred   |                 2 |                   11 |           11 |
| air_quality_uci               | causal    |                 0 |                    0 |            0 |
| air_quality_uci               | centred   |                 2 |                   11 |           11 |
| beijing_multisite_air_quality | causal    |                 0 |                    0 |            0 |
| beijing_multisite_air_quality | centred   |                 2 |                   11 |           11 |
| jena_climate                  | causal    |                 0 |                    0 |            0 |
| jena_climate                  | centred   |                 2 |                   71 |           71 |
| ett_h1                        | causal    |                 0 |                    0 |            0 |
| ett_h1                        | centred   |                 2 |                   11 |           11 |

## Fit-leak probe: {'P0 scaler': 'ok', 'P2 scaler': 'error'}

## Declared-window rule vs measured P4 inflation (>5%): agreement 0.93; share of empirically leaking cells that the rule marks exposed: 1.00

| dataset_id                    |   horizon |   mae_inflation_pct | rule_exposed   | empirical_leak_5pct   |
|:------------------------------|----------:|--------------------:|:---------------|:----------------------|
| air_quality_uci               |         1 |               20.43 | True           | True                  |
| air_quality_uci               |         6 |               22.95 | True           | True                  |
| air_quality_uci               |        24 |                2.35 | False          | False                 |
| appliances_energy_prediction  |         1 |               18.15 | True           | True                  |
| appliances_energy_prediction  |         6 |               28.25 | True           | True                  |
| appliances_energy_prediction  |        24 |                0.54 | False          | False                 |
| beijing_multisite_air_quality |         1 |               18.9  | True           | True                  |
| beijing_multisite_air_quality |         6 |               29.81 | True           | True                  |
| beijing_multisite_air_quality |        24 |                4.22 | False          | False                 |
| household_power_consumption   |         1 |                0.04 | True           | False                 |
| household_power_consumption   |        24 |               32.85 | True           | True                  |
| household_power_consumption   |        60 |               -8.08 | False          | False                 |
| jena_climate                  |         1 |                5.29 | True           | True                  |
| jena_climate                  |         6 |                8.63 | True           | True                  |
| jena_climate                  |       144 |               -9.19 | False          | False                 |