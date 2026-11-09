"""Run the audit grid in parallel with per-cell checkpointing (built for Google Colab).

Usage (from the project root):
    python scripts/run_full_grid.py --profile full --n-jobs 2
    python scripts/run_full_grid.py --profile full --datasets jena_climate --models ridge lightgbm

Re-running resumes: any cell with an existing JSON checkpoint is skipped.
Output: results/metrics/<profile>_grid/*.json  and  results/metrics/<profile>_grid_metrics.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# One BLAS/OpenMP thread per worker so parallel workers don't oversubscribe the CPU.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from sdg_leakage_audit.evaluation.runner import run_single_cell  # noqa: E402
from sdg_leakage_audit.utils.io import read_json, read_yaml, write_json  # noqa: E402

ALL_PROTOCOLS = ["P0", "P0V", "P1", "P2", "P3", "P4", "P5", "P6B", "P6"]
ALL_MODELS = ["seasonal_naive", "persistence", "ridge", "random_forest", "lightgbm", "mlp", "lstm", "gru"]


def _run_cell(cell: tuple, config: dict, ckpt_dir: Path) -> dict:
    dataset, protocol, model, horizon, seed = cell
    path = ckpt_dir / f"{dataset}__{protocol}__{model}__h{horizon}__s{seed}.json"
    if path.is_file():
        return read_json(path)
    try:
        res = run_single_cell(dataset, protocol, model, horizon, seed, config)
    except Exception as exc:  # one bad cell must not kill the grid; errors are retried on re-run
        return {"dataset_id": dataset, "protocol_id": protocol, "model_name": model,
                "horizon": horizon, "seed": seed, "error": repr(exc)}
    write_json(path, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["fast", "colab", "full"])
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--protocols", nargs="*", default=ALL_PROTOCOLS)
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--max-cells", type=int, help="debug: only run the first N pending cells")
    a = ap.parse_args()

    exp = read_yaml(ROOT / "configs" / "experiment.yaml")
    profile = exp["profiles"][a.profile]
    config = {k: profile[k] for k in ("data_subsample_frac", "model_param_grid", "max_rows", "candidate_param_grid") if k in profile}
    seeds = a.seeds if a.seeds is not None else profile["seeds"]
    ds_cfg = read_yaml(ROOT / "configs" / "datasets.yaml")["datasets"]
    datasets = a.datasets or list(ds_cfg)

    ckpt_dir = ROOT / "results" / "metrics" / f"{a.profile}_grid"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cells = [(d, p, m, h, s) for d in datasets for p in a.protocols for m in a.models
             for h in ds_cfg[d]["forecast_horizons"] for s in seeds]
    pending = [c for c in cells if not (ckpt_dir / f"{c[0]}__{c[1]}__{c[2]}__h{c[3]}__s{c[4]}.json").is_file()]
    if a.max_cells:
        pending = pending[: a.max_cells]
    print(f"{len(cells)} cells total, {len(cells) - len(pending)} already done, running {len(pending)} with n_jobs={a.n_jobs}")

    # Longest jobs first (big datasets, recurrent models) for better load balancing.
    heavy = {"household_power_consumption": 0, "jena_climate": 1}
    pending.sort(key=lambda c: (heavy.get(c[0], 2), c[2] not in ("lstm", "gru")))
    try:
        Parallel(n_jobs=a.n_jobs)(delayed(_run_cell)(c, config, ckpt_dir) for c in tqdm(pending, desc="cells"))
    except Exception as exc:  # a worker died (OOM / native crash): finish remaining cells serially
        print(f"Parallel run failed ({exc!r:.120}); finishing remaining cells serially (checkpoints kept).")
        left = [c for c in pending if not (ckpt_dir / f"{c[0]}__{c[1]}__{c[2]}__h{c[3]}__s{c[4]}.json").is_file()]
        for c in tqdm(left, desc="serial"):
            _run_cell(c, config, ckpt_dir)

    rows = [read_json(p) for p in sorted(ckpt_dir.glob("*.json"))]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics" / f"{a.profile}_grid_metrics.csv"
    df.to_csv(out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"Wrote {out} ({len(df)} rows, {n_err} errored)")


if __name__ == "__main__":
    main()
