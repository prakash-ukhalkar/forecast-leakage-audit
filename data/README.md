# Data provenance and licensing

Real provenance records (resolved URLs, download dates, file sizes, SHA-256 hashes) are
populated by `notebooks/01_data_acquisition.ipynb` (executed 2026-10-19) and written to
`data/MANIFEST.json` via `src.sdg_leakage_audit.data.manifest`. Nothing under `data/raw/`,
`data/interim/`, `data/processed/` is committed except `.gitkeep` placeholders (see
`.gitignore`) — re-run notebook 01 to repopulate `data/raw/` locally.

## Datasets and licences

| Dataset | Source | Resolved route | Size | SDG | Licence | Notes |
|---|---|---|---|---|---|---|
| Individual household electric power consumption (UCI id 235) | UCI ML Repository | `ucimlrepo` package | 128.5 MB (2,075,259 rows) | SDG 7 | UCI Repository terms — generally open for research/academic use (repo itself CC BY 4.0) | TODO-VERIFY exact dataset-level terms |
| Appliances energy prediction (UCI id 374) | UCI ML Repository | `ucimlrepo` package | 4.4 MB (19,735 rows) | SDG 7 | UCI Repository terms | TODO-VERIFY |
| Air Quality (UCI id 360) | UCI ML Repository | `ucimlrepo` package | 0.76 MB (9,357 rows) | SDG 3 / SDG 11 | UCI Repository terms | Contains sentinel value -200 for missing readings; TODO-VERIFY licence |
| Beijing Multi-Site Air-Quality Data (UCI id 501) | UCI ML Repository | direct `archive.ics.uci.edu` zip (ucimlrepo package route rejects id=501: "not available for import") | 8.2 MB (zip) | SDG 3 / SDG 11 | UCI Repository terms | Project uses id 501 only (not id 381); TODO-VERIFY licence |
| Jena Climate 2009-2016 | Max Planck Institute for Biogeochemistry (mirrored via Google/TensorFlow) | direct zip URL | 13.6 MB (zip) | SDG 13 | Typically free for research/non-commercial use | TODO-VERIFY exact redistribution terms before any use beyond research |

Exact SHA-256 hashes and timestamps for the current local download are in
`data/MANIFEST.json` (committed) and summarized in `results/tables/data_provenance.csv`.

## Per-dataset provenance record

Each entry in `data/MANIFEST.json` (and the provenance table from notebook 01) contains:

```json
{
  "dataset_id": "<string id matching configs/datasets.yaml>",
  "url": "<original configured URL>",
  "resolved_url": "<actual URL/package call that succeeded>",
  "date_downloaded": "<ISO 8601 timestamp>",
  "size_bytes": 0,
  "sha256": "<hex digest>"
}
```

## Open Power System Data (skipped for now)

Open Power System Data is **not** part of this project's dataset set at this time; see
the commented-out TODO note in `configs/datasets.yaml` for a placeholder to revisit later.
