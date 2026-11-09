# forecast-leakage-audit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

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

See `configs/datasets.yaml` for full source URLs, license notes, and forecast
horizons.

## Protocols audited (P0-P6)

| id | Name |
|---|---|
| P0 | Corrected baseline (chronological split, purge gap, train-only fit, validation-block tuning) |
| P1 | Random/shuffled split |
| P2 | Scaling/imputation fit on full dataset before splitting |
| P3 | Overlapping windows across train/test boundary |
| P4 | Future-informed (look-ahead) features: centered rolling windows |
| P5 | Hyperparameter tuning/model selection on the test set |
| P6 | Shuffled k-fold CV instead of blocked/expanding-window CV |

Full definitions: `configs/protocols.yaml`. Implementations:
`src/sdg_leakage_audit/protocols/`.

## Project status

The full audit has been run (profile `colab`: 5 datasets x 9 protocols x 8 models x 3 horizons
x 3 seeds = 3,240 runs). Final outputs are in `results/final/` (see `results/final/SUMMARY.md`,
`tables/`, `figures/`, and per-run metrics in `metrics/colab_grid_metrics.csv`).

Protocols: P0 corrected baseline; P1-P6 each inject one evaluation error. Two control arms make the
comparisons fair: **P0V** (P0 with hyperparameters tuned on a validation block; baseline for P5) and
**P6B** (blocked expanding-window CV; baseline for P6). Inflation is reported for learned models only.

Headline (MAE inflation, learned models, 95% cluster-bootstrap CI): P6 +30.2% [24.9, 35.4],
P1 +17.5% [11.5, 23.5], P4 +11.7% [7.4, 15.7]; P2, P3 and P5 indistinguishable from zero.
Known confounds (test-period difficulty for P1/P6, training-set size for P6) are described in
`results/final/SUMMARY.md`.

- `00`-`05`: environment check, data acquisition and cleaning, EDA, features, protocol/splitter validation.
- `06`-`11`: baseline models, leakage injection, statistics, SDG impact, robustness, tables/figures.
  Their stored outputs come from an earlier small `fast`-profile smoke run; the authoritative results
  are those produced by `scripts/run_full_grid.py` + `scripts/analyze_results.py` (below).

## Reproducing the full run

```bash
# grid (resumable; one JSON checkpoint per cell) then analysis
python scripts/run_full_grid.py --profile colab --n-jobs 2
python scripts/analyze_results.py --profile colab      # -> results/final/
```

On Google Colab (GPU used only by LSTM/GRU): build the bundle with
`python scripts/make_colab_package.py`, upload `colab_package/project.zip` to
`MyDrive/sdg_audit/`, and run `notebooks/colab_run_full_grid.ipynb` cell by cell.

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

`configs/experiment.yaml` defines three profiles:

- **fast** -- 1 seed, subsampled data, small hyperparameter grid; for smoke-testing
  the pipeline end-to-end quickly.
- **colab** -- 3 seeds, last 20,000 rows per series, moderate grids; the profile used for
  the results in `results/final/`.
- **full** -- all 5 seeds, full data, full hyperparameter grid; for final results.

## Repository structure

See `configs/`, `src/sdg_leakage_audit/`, `notebooks/`, `tests/`, `reports/` for
the main content. `data/raw`, `data/interim`, and `data/processed` are git-ignored
(populated by running the notebooks) except for `data/MANIFEST.json`, which is
committed so provenance/checksums are visible without re-downloading anything.

## Citation

See [CITATION.cff](CITATION.cff) for structured citation metadata (BibTeX/APA
export available via GitHub's "Cite this repository" button).

## License

Code is MIT-licensed (see [LICENSE](LICENSE)). Datasets have their own, separate
licenses -- see [data/README.md](data/README.md).
