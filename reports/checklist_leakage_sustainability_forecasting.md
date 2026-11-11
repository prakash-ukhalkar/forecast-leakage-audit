# Checklist: Self-Auditing Your Forecasting Evaluation Protocol

A reusable, practitioner-facing tool to self-audit a time-series forecasting evaluation
pipeline for the six protocol errors (P1-P6) studied in this repository. Items derive from the
protocol definitions in `configs/protocols.yaml`.

**Where to look first.** In this repository's audit (`results/final/SUMMARY.md`; learned models,
five datasets, MAE), the errors that inflated apparent accuracy most were shuffled K-fold CV
(section 6, about +30% vs blocked CV), random splitting (section 1, about +17%) and look-ahead
features (section 4, about +12%). Scaling/imputation on the full data (section 2), a missing purge gap
(section 3) and test-set tuning (section 5) were not distinguishable from zero in that design, but
remain incorrect practice and can matter in other settings. Effect sizes depend on the dataset, model
and horizon, so treat this ordering as a prioritisation aid, not a guarantee.

## How to use this checklist

For each question, answer "yes, I avoid this" or "no / unsure" for your own
forecasting pipeline. Each "no / unsure" is a candidate source of inflated
apparent performance.

## 1. Data splitting (P1)

- [ ] Is the train/test split performed strictly in chronological order (not a
      random/shuffled split of rows or windows)?
- [ ] If windows are randomly assigned to folds for any reason, has the time-based
      adjacency of those windows been checked for leakage?

## 2. Preprocessing fit order (P2)

- [ ] Are scalers, imputers, and feature-selection steps fit ONLY on the training
      fold, never on the full dataset (train+val+test) before splitting?
- [ ] Is this re-verified for every pipeline step that calls `.fit()` on anything
      resembling global statistics (mean, std, min/max, vocabulary, categories)?
- [ ] Is there an automated guard that fails if any fitted step sees test-fold data? See
      `tests/test_no_peeking_guard.py` for a pattern that can be adapted.

## 3. Window construction at fold boundaries (P3)

- [ ] Is there a purge gap >= forecast horizon between the last training window and
      the first test window?
- [ ] Do any windows span across the train/test boundary?

## 4. Feature engineering (P4)

- [ ] Are all rolling-window features strictly trailing/causal (never centered or
      using `shift(-k)` for positive k)?
- [ ] Are lag features computed on a single, time-sorted fold, never on a
      pre-split/unsorted/merged full series?
- [ ] Have any "it just works" feature-engineering shortcuts (e.g. computing
      features before splitting "for convenience") been re-checked for this?

## 5. Model/hyperparameter selection (P5)

- [ ] Is hyperparameter tuning and model selection performed on a validation block
      that strictly precedes the test block in time?
- [ ] Has the test set been used, even once, to decide between model
      configurations (including "just checking" during development)?

## 6. Cross-validation scheme (P6)

- [ ] If cross-validation is used for model selection or performance estimation, is
      it a blocked or expanding-window CV scheme rather than standard shuffled
      K-fold?
- [ ] Has the default CV splitter of the ML library in use (often shuffled K-fold)
      been explicitly overridden for time-series data?

## 7. Decision-consequence framing

For every "no / unsure" above, state the decision it could distort. Operational uses often size a
safety margin proportional to forecast error (reserve capacity, alert thresholds, storage headroom),
so an optimistic error estimate yields a margin that is too small. Template: "Leakage in step N may
understate forecast error by about X%, so a margin sized from the reported error would be too small by
a similar proportion." Note that this follows arithmetically from the error ratio; it is not a measured
downstream outcome, which needs a task-specific metric (for example alert recall or peak-load shortfall).

## Provenance

Derived from `configs/protocols.yaml` (P0 corrected baseline, P1-P6 injected errors, and the P0V/P6B
control arms), as implemented in `src/sdg_leakage_audit/protocols/` and evaluated in
`results/final/`.
