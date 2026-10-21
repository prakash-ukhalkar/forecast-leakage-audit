# sdg-leakage-audit

**How evaluation-protocol errors inflate sustainability forecasting performance:
a multi-domain leakage audit mapped to SDGs 3, 7, 11 and 13.**

This is a methodological research project. We build a corrected (leakage-safe)
forecasting baseline protocol (**P0**), then inject six common evaluation-protocol
errors one at a time (**P1-P6**) and measure how much each inflates apparent
forecasting performance -- with uncertainty quantification -- across five public
datasets spanning four UN Sustainable Development Goals. The whole pipeline is
designed to be fully reproducible and runnable by anyone, with no proprietary data
or paid services required.

## SDG mapping

| SDG | Dataset(s) | Target |
|---|---|---|
| SDG 7 (Affordable and Clean Energy) | Individual Household Electric Power Consumption (UCI 235); Appliances Energy Prediction (UCI 374) | Electricity/appliance load |
| SDG 3 / SDG 11 (Good Health & Well-Being / Sustainable Cities) | Air Quality (UCI 360); Beijing Multi-Site Air-Quality (UCI 501) | CO/PM2.5 pollutant concentration |
| SDG 13 (Climate Action) | Jena Climate 2009-2016 | Temperature |

See `configs/datasets.yaml` for full source URLs, licence notes, and forecast
horizons.

## Protocols audited (P0-P6)

| id | Name |
|---|---|
| P0 | Corrected baseline (chronological split, purge gap, train-only fit, validation-block tuning) |
| P1 | Random/shuffled split |
| P2 | Scaling/imputation fit on full dataset before splitting |
| P3 | Overlapping windows across train/test boundary |
| P4 | Target-derived / future-informed features |
| P5 | Hyperparameter tuning/model selection on the test set |
| P6 | Shuffled k-fold CV instead of blocked/expanding-window CV |

Full definitions: `configs/protocols.yaml`. Implementations:
`src/sdg_leakage_audit/protocols/`.

## Project status

All 12 notebooks (`00` through `11`) are implemented and have been run end-to-end
on real data under the `fast` experiment profile (single seed, subsampled data,
small hyperparameter grid -- a pipeline-correctness smoke test, not final
results). The full test suite passes.

- `00`-`02`: environment check, data acquisition (all 5 datasets downloaded and
  checksummed into `data/MANIFEST.json`), cleaning.
- `03`-`05`: exploratory analysis, feature engineering, protocol/splitter
  validation.
- `06`-`07`: P0 baseline models and the full P1-P6 leakage-injection grid
  (checkpointed, resumable).
- `08`-`09`: statistical analysis of inflation effects (bootstrap CIs, paired
  tests, Friedman/Nemenyi, effect sizes) and translation into SDG decision-impact
  terms.
- `10`-`11`: robustness/sensitivity checks and final tables/figures.

**Next step:** re-run notebooks `06`-`11` under the `full` experiment profile
(all seeds, full data, full hyperparameter grids) to produce final results --
everything currently in `results/` demonstrates the pipeline and the
inflation-measurement method correctly, but was generated under the `fast`
profile and should not be treated as final.

## Quickstart

```powershell
# Clone, then from the repo root:
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -v
```

Or use the provided Windows script:

```powershell
.\run.ps1 setup
.\run.ps1 test
```

On Linux/macOS/WSL with `make`:

```bash
make setup
make test
```

To reproduce a notebook, activate the environment and run it top-to-bottom (each
notebook documents its own inputs/outputs/runtime at the top); notebooks must be
run in order since each depends on the previous ones' outputs under `data/` and
`results/`.

## Experiment profiles

`configs/experiment.yaml` defines two profiles:

- **fast** -- 1 seed, subsampled data, small hyperparameter grid; for smoke-testing
  the pipeline end-to-end quickly. Used for every result currently in `results/`.
- **full** -- all 5 seeds, full data, full hyperparameter grid; for final results.

## Repository structure

See `configs/`, `src/sdg_leakage_audit/`, `notebooks/`, `tests/`, `reports/` for
the main content. `data/raw`, `data/interim`, and `data/processed` are git-ignored
(populated by running the notebooks) except for `data/MANIFEST.json`, which is
committed so provenance/checksums are visible without re-downloading anything.

## Citation

See `CITATION.cff`. **Note:** author name/ORCID in `CITATION.cff` and the
copyright holder in `LICENSE` are placeholders (`TBD`) -- the repository owner
should fill these in before any public release.

## Licence

Code is MIT-licensed (see `LICENSE`). Datasets have their own, separate licences
-- see `data/README.md`.
