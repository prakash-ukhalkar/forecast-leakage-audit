# Results summary (profile `colab`)

Grid rows: 3240 (0 errored and excluded). Learned-model units: 90.

## Headline inflation (MAE %, + = flattering)
| protocol   |   n_units |   mean_mae_inflation_pct |   ci95_lo |   ci95_hi |   median_pct |   share_units_inflated |   mean_skill_gain_abs |   skill_gain_ci95_lo |   skill_gain_ci95_hi |   wilcoxon_p |   cliffs_delta_mae_vs_baseline |   p_holm | significant_holm_0.05   |
|:-----------|----------:|-------------------------:|----------:|----------:|-------------:|-----------------------:|----------------------:|---------------------:|---------------------:|-------------:|-------------------------------:|---------:|:------------------------|
| P1         |        90 |                   17.47  |    11.498 |    23.532 |       10.534 |                  0.689 |                 0.246 |                0.184 |                0.317 |        0     |                          0.125 |    0     | True                    |
| P2         |        90 |                    0.127 |    -0.514 |     0.754 |        0     |                  0.256 |                -0     |               -0.007 |                0.006 |        0.185 |                          0.004 |    0.37  | False                   |
| P3         |        90 |                   -0.139 |    -0.383 |     0.074 |        0.002 |                  0.522 |                 0.001 |               -0.002 |                0.003 |        0.722 |                          0     |    0.722 | False                   |
| P4         |        90 |                   11.677 |     7.357 |    15.708 |        3.54  |                  0.567 |                 0.085 |                0.041 |                0.127 |        0     |                          0.06  |    0     | True                    |
| P5         |        90 |                    0.458 |    -0.712 |     1.578 |        0     |                  0.3   |                 0.008 |               -0.005 |                0.023 |        0.064 |                          0.004 |    0.193 | False                   |
| P6         |        90 |                   30.212 |    24.906 |    35.434 |       23.812 |                  0.933 |                 0.305 |                0.229 |                0.393 |        0     |                          0.216 |    0     | True                    |

## P2 restricted to scale-sensitive models (ridge, mlp)
|   mean_pct |   ci95_lo |   ci95_hi |   n_units |
|-----------:|----------:|----------:|----------:|
|      0.381 |    -1.432 |      2.36 |        30 |

Caveats: P1/P6 draw different test rows than P0, so MAE inflation mixes leakage with test-period difficulty (see `skill_gain` columns, which are relative to seasonal-naive on the same rows); P6 also trains on ~80% of the data per fold vs the smaller expanding windows of P6B.

## Control arms vs P0 (not leakage)
| protocol_id   |   mean |   median |   count |
|:--------------|-------:|---------:|--------:|
| P0V           |   1.84 |     0    |      90 |
| P6B           | -23.07 |   -10.55 |      90 |

## Omnibus
Friedman chi2=160.69, p=2.29e-31

## By dataset
| dataset_id                    |     P1 |    P2 |    P3 |    P4 |    P5 |    P6 |
|:------------------------------|-------:|------:|------:|------:|------:|------:|
| air_quality_uci               |  20.75 | -0.02 |  0    | 15.24 |  1.03 | 21.48 |
| appliances_energy_prediction  |  12.18 | -1.36 |  0.01 | 15.65 |  0.57 | 23.36 |
| beijing_multisite_air_quality | -14.7  | -0.2  | -0.1  | 17.65 |  0.13 | 21.64 |
| household_power_consumption   |  27.35 |  0.77 |  0.43 |  8.27 | -1.94 | 32.53 |
| jena_climate                  |  41.78 |  1.44 | -1.04 |  1.58 |  2.51 | 52.05 |

## By family
| family        |    P1 |   P2 |    P3 |    P4 |    P5 |    P6 |
|:--------------|------:|-----:|------:|------:|------:|------:|
| linear        |  4.71 | 0.14 | -0.18 | 18.32 |  2.12 | 16.11 |
| neural        | 11.32 | 0.21 | -0.27 |  5.13 | -0.64 | 26.39 |
| tree_ensemble | 33.07 | 0    |  0.08 | 18.17 |  1.28 | 43    |

## Inflation vs autocorrelation (Spearman)
| protocol   |   n_dataset_horizons |   spearman_rho |     p |
|:-----------|---------------------:|---------------:|------:|
| P1         |                   15 |          0.2   | 0.475 |
| P2         |                   15 |          0.875 | 0     |
| P3         |                   15 |         -0.021 | 0.94  |
| P4         |                   15 |         -0.104 | 0.713 |
| P5         |                   15 |         -0.079 | 0.781 |
| P6         |                   15 |          0.225 | 0.42  |