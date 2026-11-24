"""Ground-truth study of evaluation leakage on simulated series.

For each simulated process the data-generating mechanism continues beyond the series that the protocols see, so the TRUE
deployment error of every fitted model can be measured on fresh future data. For a protocol p and a fitted model,

    ground-truth inflation = 100 * (MAE_future - MAE_protocol_estimate) / MAE_future ,

the relative optimism of the protocol's estimate. No protocol is used as its own reference, so the result does not depend on
choosing a baseline. For linear Gaussian processes the interpolation ceiling of audit/theory.py is the predicted upper bound.

Usage: python scripts/synthetic_groundtruth.py --n-jobs 6 [--seeds 0 1 2 3 4 5 6 7]
Output: results/synthetic/groundtruth_cells.csv, results/synthetic/ceiling.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

from sdg_leakage_audit.audit.theory import leakage_ceiling  # noqa: E402
from sdg_leakage_audit.evaluation import runner as R  # noqa: E402
from sdg_leakage_audit.features.build import build_feature_matrix  # noqa: E402
from sdg_leakage_audit.protocols.injectors import fit_imputer_correct_train_only, fit_scaler_correct_train_only  # noqa: E402
from sdg_leakage_audit.protocols.splitters import (  # noqa: E402
    blocked_cv_splits,
    chronological_split_with_gap,
    random_split,
    shuffled_kfold_splits,
    size_matched_shuffled_splits,
)
from sdg_leakage_audit.utils.io import read_yaml  # noqa: E402

OUT = ROOT / "results" / "synthetic"
HORIZONS = [1, 6, 24]
LAGS = [1, 6, 24]
N, N_FUT = 5000, 3000
MODELS = ["ridge", "random_forest", "lightgbm", "mlp"]
PROTOCOLS = ["P0", "P1", "P4", "P6", "P6B", "P6S"]
# process name -> (kind, params). AR coefficients of the stochastic part are used for the theoretical ceiling.
PROCESSES = {
    "ar1_0.3": ("ar", [0.3]), "ar1_0.6": ("ar", [0.6]), "ar1_0.9": ("ar", [0.9]), "ar1_0.98": ("ar", [0.98]),
    "seas_ar1_0.5": ("seas", [0.5]), "seas_ar1_0.9": ("seas", [0.9]),
    "ar2_osc": ("ar", [1.2, -0.5]), "tar_0.9_0.3": ("tar", [0.9, 0.3]),
}


def simulate(kind: str, par: list[float], n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(size=n + 500)
    if kind == "ar":
        y = lfilter([1.0], np.r_[1.0, -np.asarray(par)], eps)
    elif kind == "seas":
        noise = lfilter([1.0], [1.0, -par[0]], eps)
        t = np.arange(n + 500)
        y = noise + 2.0 * np.sin(2 * np.pi * t / 24) + 1.0 * np.sin(2 * np.pi * t / 168 + 1.0)
    elif kind == "tar":
        y = np.zeros(n + 500)
        for i in range(1, n + 500):
            y[i] = (par[0] if y[i - 1] > 0 else par[1]) * y[i - 1] + eps[i]
    else:
        raise ValueError(kind)
    return y[500:]


def model_fit_predict(model: str, seed: int, models_cfg: dict, Xtr: pd.DataFrame, ytr: pd.Series, Xev: list[pd.DataFrame]) -> list[np.ndarray]:
    """Fit once on the training rows and predict each evaluation matrix (same imputer/scaler recipe as the runner)."""
    imp = fit_imputer_correct_train_only(Xtr.to_numpy())
    A = imp.transform(Xtr.to_numpy())
    Es = [imp.transform(x.to_numpy()) for x in Xev]
    if model in R._SCALED_MODEL_NAMES:
        sc = fit_scaler_correct_train_only(A)
        A = sc.transform(A)
        Es = [sc.transform(e) for e in Es]
    params = R._resolve_model_params(model, models_cfg, "params_colab", seed)
    cat = np.concatenate(Es, axis=0)
    pred = R._fit_predict(model, params, A, ytr.to_numpy(), cat, "y", Xtr.columns)
    out, k = [], 0
    for e in Es:
        out.append(pred[k : k + len(e)])
        k += len(e)
    return out


def run_job(process: str, h: int, seed: int) -> list[dict]:
    kind, par = PROCESSES[process]
    models_cfg = read_yaml(ROOT / "configs" / "models.yaml")["models"]
    rng = np.random.default_rng(1000 * seed + 17)
    y = simulate(kind, par, N + N_FUT + h, rng)
    idx = pd.date_range("2020-01-01", periods=len(y), freq="h")
    df = pd.DataFrame({"y": y}, index=idx)
    feats = build_feature_matrix(df, "y", lags=LAGS, rolling_windows=LAGS, include_calendar=True)
    full = pd.concat([df, feats.drop(columns=[c for c in feats.columns if c in df.columns])], axis=1)
    # month and day-of-year carry no information here and fall outside the training range in the fresh future block
    full = full.drop(columns=[c for c in full.columns if c.startswith(("month", "day_of_year"))])

    block = full.iloc[:N]
    Xp, yp, _ = R._build_direct_horizon_xy(block, "y", h)  # protocol rows: targets inside the visible block
    Xp_leak, yp_leak, _ = R._build_direct_horizon_xy(R._apply_p4_feature_leak(block, "y"), "y", h)
    keep = Xp.index.intersection(Xp_leak.index)
    Xp_leak, yp_leak = Xp_leak.loc[keep], yp_leak.loc[keep]
    Xall, yall, _ = R._build_direct_horizon_xy(full, "y", h)
    fut = Xall.index >= idx[N + h]  # fresh rows after a purge gap of h
    Xf, yf = Xall[fut], yall[fut]
    n = len(Xp)

    rows = []
    for model in MODELS:
        for proto in PROTOCOLS:
            if proto == "P0":
                folds = [chronological_split_with_gap(n, 0.7, 0.2, h)]
            elif proto == "P1":
                folds = [random_split(n, 0.7, 0.2, seed=seed)]
            elif proto == "P4":
                folds = [chronological_split_with_gap(n, 0.7, 0.2, h)]
            elif proto == "P6":
                folds = shuffled_kfold_splits(n, 5, seed=seed)
            elif proto == "P6B":
                folds = blocked_cv_splits(n, 5, horizon=h, expanding=True)
            else:
                folds = size_matched_shuffled_splits(n, 5, horizon=h, seed=seed)
            est, tru = [], []
            for tr, te in folds:
                if proto == "P4":
                    Xtr, ytr = Xp_leak.iloc[tr], yp_leak.iloc[tr]
                    Xte, yte = Xp_leak.iloc[te], yp_leak.iloc[te]
                else:
                    Xtr, ytr, Xte, yte = Xp.iloc[tr], yp.iloc[tr], Xp.iloc[te], yp.iloc[te]
                p_est, p_fut = model_fit_predict(model, seed, models_cfg, Xtr, ytr, [Xte, Xf])
                est.append(float(np.mean(np.abs(yte.to_numpy() - p_est))))
                tru.append(float(np.mean(np.abs(yf.to_numpy() - p_fut))))
            mae_est, mae_true = float(np.mean(est)), float(np.mean(tru))
            rows.append({"process": process, "horizon": h, "seed": seed, "model": model, "protocol": proto,
                         "mae_est": mae_est, "mae_true": mae_true, "inflation_true_pct": 100 * (mae_true - mae_est) / mae_true})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(8)))
    ap.add_argument("--processes", nargs="*", default=list(PROCESSES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, h, s) for p in a.processes for h in HORIZONS for s in a.seeds]
    ck = OUT / "checkpoints"
    ck.mkdir(exist_ok=True)

    def do(p, h, s):
        f = ck / f"{p}__h{h}__s{s}.csv"
        if f.is_file():
            return pd.read_csv(f)
        d = pd.DataFrame(run_job(p, h, s))
        d.to_csv(f, index=False)
        return d

    print(f"{len(jobs)} jobs ({sum((ck / f'{p}__h{h}__s{s}.csv').is_file() for p, h, s in jobs)} done)")
    parts = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(do)(*j) for j in jobs)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / "groundtruth_cells.csv", index=False)
    # theoretical ceiling for the linear Gaussian processes
    rows = []
    for p, (kind, par) in PROCESSES.items():
        if kind in ("ar", "seas"):
            for h in HORIZONS:
                rows.append({"process": p, "horizon": h, "ceiling_mae_pct": 100 * leakage_ceiling(par, h, metric="mae"),
                             "ceiling_mse_pct": 100 * leakage_ceiling(par, h, metric="mse")})
    pd.DataFrame(rows).to_csv(OUT / "ceiling.csv", index=False)
    print("done", df.shape)


if __name__ == "__main__":
    main()
