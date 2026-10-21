# Checklist: Self-Auditing Your Forecasting Evaluation Protocol

**Status: draft.** This checklist is a deliverable of this project -- a reusable,
practitioner-facing tool to self-audit a time-series forecasting evaluation
pipeline for the six protocol errors (P1-P6) studied here. Bullet points below
are derived directly from the P0-P6 protocol definitions in
`configs/protocols.yaml`; they can be refined/expanded as the `full`-profile
results come in.

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
- [ ] TODO (Phase 2): report the automated-guard pattern used in
      `tests/test_no_peeking_guard.py` as a template practitioners can adapt.

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

## 7. SDG decision-consequence framing (TODO, Phase 2)

- [ ] TODO: translate each "no/unsure" answer above into a concrete decision-risk
      statement (e.g. "leakage in step 4 may overstate air-quality exceedance-alert
      recall, risking under-preparedness for SDG 3/11 health decisions").

## Provenance

Derived from `configs/protocols.yaml` protocol definitions P0 (corrected baseline)
through P6, as implemented in `src/sdg_leakage_audit/protocols/`. To be expanded
with quantitative severity rankings once Phase 2 experiments (notebooks 06-10) are
complete.
