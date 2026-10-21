# Report Outline

Title: "How evaluation-protocol errors inflate sustainability forecasting
performance: a multi-domain leakage audit mapped to SDGs 3, 7, 11 and 13."

## Abstract
TODO (after full-profile results): summarize motivation, method (P0-P6 audit across 5
datasets x 8 models), headline inflation findings, and SDG decision-impact framing.

## 1. Introduction
- TODO: motivate why evaluation-protocol errors in time-series forecasting are
  under-scrutinized relative to model novelty.
- TODO: state the core contribution -- a controlled, multi-domain audit isolating
  the inflation effect of each of six common protocol errors (P1-P6) against a
  corrected baseline (P0).
- TODO: state the SDG framing (3, 7, 11, 13) and why evaluation rigor matters for
  real-world sustainability decisions built on forecasts.

## 2. Related Work
- TODO: time-series leakage literature (data leakage taxonomies, temporal CV).
- TODO: forecasting benchmarking critiques (reproducibility, protocol
  sensitivity).
- TODO: SDG-aligned ML application literature (energy, air quality, climate).

## 3. Methodology
### 3.1 Datasets
- TODO: describe all 5 datasets (see `configs/datasets.yaml`), SDG mapping, and
  licensing.
### 3.2 Protocols (P0-P6)
- TODO: formal definitions (see `configs/protocols.yaml`), with the protocol
  timeline figures from notebook 05.
### 3.3 Models
- TODO: describe all 8 models (see `configs/models.yaml`) and the rationale for
  LightGBM over XGBoost, and scikit-learn MLP over a hand-rolled PyTorch MLP.
### 3.4 Statistical analysis
- TODO: moving-block bootstrap CIs, paired tests with multiple-comparison
  correction, Friedman + Nemenyi, effect sizes (see `configs/experiment.yaml`).

## 4. Results
- TODO: per-protocol inflation tables/figures (notebook 08 outputs).
- TODO: cross-dataset, cross-model consistency of inflation effects.
- TODO: robustness/sensitivity results (notebook 10).

## 5. SDG Decision Impact Discussion
- TODO: SDG 7 (energy) -- reserve margin / storage sizing decisions distorted by
  inflated accuracy claims.
- TODO: SDG 3/11 (air quality) -- exceedance-alert false-negative/positive risk.
- TODO: SDG 13 (climate) -- downstream climate-forecast-dependent planning risk.

## 6. Limitations
- TODO: dataset scope (5 datasets, not exhaustive); model scope; synthetic vs
  real-world deployment gap; assumptions in SDG impact translation (notebook 09).

## 7. Conclusion
- TODO: summarize headline inflation magnitudes and the practical checklist
  contribution (see `reports/checklist_leakage_sustainability_forecasting.md`).

## Appendix
- TODO: full hyperparameter grids, additional figures, per-dataset detailed
  tables.
