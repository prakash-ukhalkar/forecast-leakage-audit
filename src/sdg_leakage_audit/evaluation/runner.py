"""Experiment runner: orchestrates dataset x protocol x model x horizon x seed grids.

`run_single_cell` implements all seven protocols (P0-P6); `run_grid` checkpoints
each cell to disk so a long grid run can be interrupted and resumed.

Per-protocol implementation summary (see `configs/protocols.yaml` for the audit
definitions this code implements):
  P0  Chronological split (purge gap = horizon); imputer/scaler fit train-only.
  P1  Same as P0, but the split is `splitters.random_split` instead of chronological.
  P2  Same split as P0, but imputer/scaler fit on train+test combined (leaky).
  P3  Chronological split with the purge gap forced to 0 (overlap allowed).
  P4  Same split as P0, but lag/rolling feature columns are replaced with their
      leaky (centered-rolling / shuffled-then-lagged) counterparts before fitting.
      implemented via `protocols.injectors.rolling_feature_leaky_centered` /
      `lag_feature_leaky_full_series`.
  P5  Three-way (train/val/test) split; hyperparameters chosen by scoring candidates
      directly on the test block (leaky), via `protocols.injectors.select_best_params_leaky_on_test`.
      Only meaningful for models with a tunable parameter grid (ridge, random_forest,
      lightgbm, mlp); for other models this is a no-op, identical to P0.
  P6  Metrics averaged over `splitters.shuffled_kfold_splits` folds instead of a
      single chronological split -- the CV-estimate variant of the audit. Compared
      against P0's single-split metrics by notebook 08, with that methodological
      difference noted explicitly rather than hidden.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from sdg_leakage_audit.evaluation.metrics import (
    mae,
    mase,
    rmse,
    skill_score_vs_seasonal_naive,
    smape,
)
from sdg_leakage_audit.models.factory import build_model
from sdg_leakage_audit.protocols.injectors import (
    fit_imputer_correct_train_only,
    fit_imputer_leaky_full_data,
    fit_scaler_correct_train_only,
    fit_scaler_leaky_full_data,
    lag_feature_leaky_full_series,
    rolling_feature_leaky_centered,
    select_best_params_leaky_on_test,
)
from sdg_leakage_audit.protocols.splitters import (
    chronological_split_with_gap,
    random_split,
    shuffled_kfold_splits,
)
from sdg_leakage_audit.utils.io import read_json, read_yaml, write_json
from sdg_leakage_audit.utils.logging import get_logger
from sdg_leakage_audit.utils.paths import configs_dir, data_processed_dir, to_project_relative
from sdg_leakage_audit.utils.seed import set_global_seed

logger = get_logger(__name__)

_SCALED_MODEL_NAMES = {"ridge", "mlp"}
_RECURRENT_MODEL_NAMES = {"lstm", "gru"}
_TUNABLE_MODEL_NAMES = {"ridge", "random_forest", "lightgbm", "mlp"}
_KNOWN_PROTOCOLS = {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}


def _load_dataset_frame(dataset_id: str) -> pd.DataFrame:
    """Load a dataset's engineered feature matrix (notebook 04 output).

    For the multi-station Beijing dataset, restricts to one representative station
    (the alphabetically first), consistent with the convention already used in
    notebooks 03-04, since the recurrent-sequence framing here assumes one series.
    """
    df = pd.read_parquet(data_processed_dir() / f"{dataset_id}_features.parquet")
    if "station" in df.columns:
        representative_station = sorted(df["station"].unique())[0]
        df = df.loc[df["station"] == representative_station].drop(columns=["station"])
    return df


def _apply_p4_feature_leak(df: pd.DataFrame, target_column: str, seed: int) -> pd.DataFrame:
    """P4: replace causal lag/rolling feature columns with their leaky counterparts.

    Rolling-mean columns (`{target}_roll{window}_mean`) are replaced with the
    centered rolling mean (reads values after the row it's attached to).
    Lag columns (`{target}_lag{lag}`) are replaced with a lag computed on a
    shuffled copy of the target series, so the "lag" value at each row no longer
    corresponds to that row's true temporal predecessor -- simulating the
    documented full-series-pre-split lag bug without reordering the rest of the
    DataFrame (only these specific feature columns are corrupted).

    Args:
        df: The dataset's engineered feature DataFrame (notebook 04 output).
        target_column: Name of the target column.
        seed: Seed for the shuffle used to corrupt lag features.

    Returns:
        A copy of `df` with lag/rolling feature columns replaced by leaky versions.
    """
    df = df.copy()

    for col in df.columns:
        if col.startswith(f"{target_column}_roll") and col.endswith("_mean"):
            window = int(col[len(f"{target_column}_roll") : -len("_mean")])
            df[col] = rolling_feature_leaky_centered(df[target_column], window)

    rng = np.random.default_rng(seed)
    shuffled_values = df[target_column].to_numpy().copy()
    rng.shuffle(shuffled_values)
    shuffled_series = pd.Series(shuffled_values, index=df.index)

    for col in df.columns:
        if col.startswith(f"{target_column}_lag"):
            lag = int(col[len(f"{target_column}_lag") :])
            df[col] = lag_feature_leaky_full_series(shuffled_series, lag)

    return df


def _build_direct_horizon_xy(
    df: pd.DataFrame, target_column: str, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y, origin_values) for the "direct" horizon-step-ahead regression framing.

    Each row's features (lags/rolling/calendar, all computed causally in notebook 04,
    unless P4 feature-leak overrides were already applied by the caller) predict the
    target `horizon` steps ahead of that row: `y.loc[i] == df[target_column]` at
    position `i + horizon`. Rows without a full feature history (head) or without a
    valid future target (tail) are dropped.

    Columns that are entirely NaN (e.g. a window-1 rolling standard deviation, which is
    mathematically undefined for every row) are dropped outright rather than treated as
    "missing" -- notebook 04 builds rolling std features for every configured forecast
    horizon including 1, and keeping an all-NaN column would make every row fail the
    completeness check below. Non-numeric columns (e.g. Beijing's categorical `wd` wind
    direction column, carried through unchanged from the raw loader) are also dropped --
    categorical encoding is out of scope for this baseline model set.

    Returns:
        `(X, y, origin_values)`, all aligned on the same (filtered) index;
        `origin_values` is `df[target_column]` at the forecast origin (used by the
        persistence baseline and by MASE's naive-diff reference).
    """
    y = df[target_column].shift(-horizon)
    X = df.drop(columns=[target_column])

    non_numeric_cols = X.select_dtypes(exclude="number").columns
    if len(non_numeric_cols) > 0:
        logger.debug("Dropping non-numeric feature columns: %s", list(non_numeric_cols))
        X = X.drop(columns=non_numeric_cols)

    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        logger.debug("Dropping structurally all-NaN feature columns: %s", list(all_nan_cols))
        X = X.drop(columns=all_nan_cols)

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid], df[target_column].loc[valid]


def _seasonal_forecast_for_index(
    df: pd.DataFrame, target_column: str, season_length: int, horizon: int, index: pd.Index
) -> np.ndarray:
    """Compute the seasonal-naive forecast aligned to `index` (used for skill scores)."""
    seasonal_model = build_model("seasonal_naive", {"season_length": season_length})
    full_forecast = seasonal_model.predict(df[target_column].to_numpy(), horizon)
    return pd.Series(full_forecast, index=df.index).loc[index].to_numpy()


def _resolve_model_params(model_name: str, models_cfg: dict[str, Any], param_grid_key: str, seed: int) -> dict[str, Any]:
    params = {k: v[0] for k, v in models_cfg[model_name][param_grid_key].items()}
    if model_name in {"random_forest", "lightgbm", "mlp"}:
        params["random_state"] = seed
    return params


def _fit_predict(
    model_name: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    target_column: str,
    feature_columns: pd.Index,
) -> np.ndarray:
    """Fit `model_name` on (X_train, y_train) and predict on X_eval.

    Shared by every protocol's non-baseline model path, so P1-P6 reuse exactly the
    same fit/predict logic P0 uses -- only the inputs (split, feature values, fit
    order, hyperparameters) differ per protocol.
    """
    params = dict(params)
    if model_name in _RECURRENT_MODEL_NAMES:
        lag_cols = sorted(
            (c for c in feature_columns if c.startswith(f"{target_column}_lag")),
            key=lambda c: int(c.rsplit("lag", 1)[1]),
        )
        lag_col_positions = [feature_columns.get_loc(c) for c in lag_cols]
        seq_train = X_train[:, lag_col_positions][..., None]
        seq_eval = X_eval[:, lag_col_positions][..., None]

        epochs = params.pop("epochs", 5)
        model = build_model(model_name, params)
        model.fit(seq_train, y_train, epochs=epochs)
        return model.predict(seq_eval)

    model = build_model(model_name, params)
    with warnings.catch_warnings():
        # The `fast` profile deliberately caps MLP's max_iter low for smoke-test speed,
        # so it routinely doesn't converge; that's expected here, not a bug.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    return model.predict(X_eval)


def _finalize_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_forecast_test: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        **metadata,
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "smape": smape(y_test, y_pred),
        "mase": mase(y_test, y_pred, y_train, seasonality=1),
        "skill_score": skill_score_vs_seasonal_naive(y_test, y_pred, seasonal_forecast_test),
    }
    logger.debug("run_single_cell: %s", metrics)
    return metrics


def run_single_cell(
    dataset_id: str,
    protocol_id: str,
    model_name: str,
    horizon: int,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one (dataset, protocol, model, horizon, seed) grid cell end-to-end.

    Framing: this uses "direct" multi-horizon regression -- each row's engineered
    features predict the target value `horizon` steps ahead of that row, evaluated
    independently per row. See the module docstring for what each protocol P0-P6
    changes relative to the P0 corrected baseline.

    Loads the dataset by id from `configs/datasets.yaml` and `data/processed/`;
    for a custom/non-standard feature frame (e.g. an alternate feature-window
    size, or a single Beijing station rather than the representative one used
    elsewhere), call `run_single_cell_on_frame` directly instead (used by
    notebook 10's robustness/sensitivity checks).

    Args:
        dataset_id: Key into `configs/datasets.yaml`.
        protocol_id: One of "P0".."P6".
        model_name: Key into `configs/models.yaml`.
        horizon: Forecast horizon in steps.
        seed: Random seed for this run.
        config: Merged experiment configuration (from `configs/experiment.yaml`'s
            active profile), must contain `data_subsample_frac` and
            `model_param_grid` (one of `"params_fast"`/`"params_full"`).

    Returns:
        Dict of computed metrics (mae, rmse, smape, mase, skill_score) plus
        metadata (dataset_id, protocol_id, model_name, horizon, seed, n_train, n_test).

    Raises:
        ValueError: If `protocol_id` is not one of "P0".."P6".
    """
    datasets_cfg = read_yaml(configs_dir() / "datasets.yaml")["datasets"]
    dataset_cfg = datasets_cfg[dataset_id]
    target_column = dataset_cfg["target_column"]
    season_length = dataset_cfg["season_length"]

    df = _load_dataset_frame(dataset_id)
    return run_single_cell_on_frame(df, dataset_id, target_column, season_length, protocol_id, model_name, horizon, seed, config)


def run_single_cell_on_frame(
    df: pd.DataFrame,
    dataset_id: str,
    target_column: str,
    season_length: int,
    protocol_id: str,
    model_name: str,
    horizon: int,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Core of `run_single_cell`, taking an already-loaded feature frame directly.

    Factored out so callers that need a non-standard feature frame -- e.g.
    notebook 10's window-size sensitivity check (a feature frame built with a
    different lag/rolling-window list than notebook 04's default) or its
    dataset-subset check (a single Beijing station's rows, rather than the
    representative station `run_single_cell` always picks) -- can reuse the
    exact same split/fit/metric logic without needing a matching entry in
    `configs/datasets.yaml`.

    Args:
        df: Feature DataFrame (DatetimeIndex, target column present), structured
            like notebook 04's output.
        dataset_id: Label stored in the returned metadata (does not need to be a
            real `configs/datasets.yaml` key).
        target_column: Name of the target column in `df`.
        season_length: Seasonal-naive season length for this series.
        protocol_id: One of "P0".."P6".
        model_name: Key into `configs/models.yaml`.
        horizon: Forecast horizon in steps.
        seed: Random seed for this run.
        config: Merged experiment configuration, as in `run_single_cell`.

    Returns:
        Same shape as `run_single_cell`.

    Raises:
        ValueError: If `protocol_id` is not one of "P0".."P6".
    """
    if protocol_id not in _KNOWN_PROTOCOLS:
        raise ValueError(f"Unknown protocol_id {protocol_id!r}; known: {sorted(_KNOWN_PROTOCOLS)}")

    set_global_seed(seed)
    models_cfg = read_yaml(configs_dir() / "models.yaml")["models"]

    if protocol_id == "P4":
        df = _apply_p4_feature_leak(df, target_column, seed)

    X_full, y_full, origin_full = _build_direct_horizon_xy(df, target_column, horizon)

    # The seasonal-naive reference (needed for every cell's skill_score, and as the
    # P0/P1-equivalent prediction when model_name="seasonal_naive") is undefined for
    # the earliest rows of the series (insufficient history for season_length - horizon
    # steps back). P0's chronological split never selects test rows from that region
    # (test is always the tail), but P1's random split and P6's shuffled K-fold can --
    # so rows without a defined seasonal reference are dropped here, uniformly across
    # all protocols, to keep every protocol's evaluation set on the same footing.
    seasonal_forecast_for_full = _seasonal_forecast_for_index(df, target_column, season_length, horizon, X_full.index)
    seasonally_valid = ~np.isnan(seasonal_forecast_for_full)
    if not seasonally_valid.all():
        X_full, y_full, origin_full = X_full.loc[seasonally_valid], y_full.loc[seasonally_valid], origin_full.loc[seasonally_valid]

    subsample_frac = config.get("data_subsample_frac", 1.0)
    if subsample_frac < 1.0:
        n_keep = max(200, int(len(X_full) * subsample_frac))
        X_full, y_full, origin_full = X_full.iloc[-n_keep:], y_full.iloc[-n_keep:], origin_full.iloc[-n_keep:]

    metadata = {
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "model_name": model_name,
        "horizon": horizon,
        "seed": seed,
    }

    if protocol_id == "P6":
        return _run_p6_cv_cell(X_full, y_full, origin_full, df, target_column, season_length, horizon, model_name, models_cfg, config, metadata)
    if protocol_id == "P5":
        return _run_p5_cell(X_full, y_full, origin_full, df, target_column, season_length, horizon, model_name, models_cfg, config, metadata)

    split_gap = 0 if protocol_id == "P3" else horizon
    if protocol_id == "P1":
        train_idx, test_idx = random_split(X_full, train_frac=0.7, test_frac=0.2, seed=seed)
    else:
        train_idx, test_idx = chronological_split_with_gap(X_full, train_frac=0.7, test_frac=0.2, horizon=split_gap)

    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
    origin_test = origin_full.iloc[test_idx]
    seasonal_forecast_test = _seasonal_forecast_for_index(df, target_column, season_length, horizon, y_test.index)

    metadata["n_train"] = len(X_train)
    metadata["n_test"] = len(X_test)

    if model_name == "seasonal_naive":
        y_pred = seasonal_forecast_test
    elif model_name == "persistence":
        y_pred = origin_test.to_numpy()
    else:
        if protocol_id == "P2":
            combined = np.concatenate([X_train.to_numpy(), X_test.to_numpy()], axis=0)
            imputer = fit_imputer_leaky_full_data(combined)
        else:
            imputer = fit_imputer_correct_train_only(X_train.to_numpy())
        X_train_imp = imputer.transform(X_train.to_numpy())
        X_test_imp = imputer.transform(X_test.to_numpy())

        if model_name in _SCALED_MODEL_NAMES:
            if protocol_id == "P2":
                combined_imp = np.concatenate([X_train_imp, X_test_imp], axis=0)
                scaler = fit_scaler_leaky_full_data(combined_imp)
            else:
                scaler = fit_scaler_correct_train_only(X_train_imp)
            X_train_imp = scaler.transform(X_train_imp)
            X_test_imp = scaler.transform(X_test_imp)

        params = _resolve_model_params(model_name, models_cfg, config["model_param_grid"], seed)
        y_pred = _fit_predict(model_name, params, X_train_imp, y_train.to_numpy(), X_test_imp, target_column, X_train.columns)

    return _finalize_metrics(y_test.to_numpy(), y_pred, y_train.to_numpy(), seasonal_forecast_test, metadata)


def _run_p5_cell(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    origin_full: pd.Series,
    df: pd.DataFrame,
    target_column: str,
    season_length: int,
    horizon: int,
    model_name: str,
    models_cfg: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """P5: select hyperparameters by scoring candidates directly on the test block.

    Uses the same chronological train/test split as P0 -- P5's leak is entirely
    about which data candidate hyperparameters are scored against (the test block
    itself, via `select_best_params_leaky_on_test`), not about split structure. A
    "correct" comparison arm would score on a validation block preceding test
    instead (`select_best_params_correct_on_validation`), but this function only
    implements the leaky arm under audit; a validation split is therefore
    unnecessary here.

    No-op (identical to P0) for models without a tunable parameter grid
    (seasonal_naive, persistence, lstm, gru) -- there is nothing to leak via
    selection if there is no selection happening.
    """
    train_idx, test_idx = chronological_split_with_gap(X_full, train_frac=0.7, test_frac=0.2, horizon=horizon)
    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
    origin_test = origin_full.iloc[test_idx]
    seasonal_forecast_test = _seasonal_forecast_for_index(df, target_column, season_length, horizon, y_test.index)

    metadata["n_train"] = len(X_train)
    metadata["n_test"] = len(X_test)

    if model_name == "seasonal_naive":
        return _finalize_metrics(y_test.to_numpy(), seasonal_forecast_test, y_train.to_numpy(), seasonal_forecast_test, metadata)
    if model_name == "persistence":
        return _finalize_metrics(y_test.to_numpy(), origin_test.to_numpy(), y_train.to_numpy(), seasonal_forecast_test, metadata)

    imputer = fit_imputer_correct_train_only(X_train.to_numpy())
    X_train_imp = imputer.transform(X_train.to_numpy())
    X_test_imp = imputer.transform(X_test.to_numpy())

    if model_name in _SCALED_MODEL_NAMES:
        scaler = fit_scaler_correct_train_only(X_train_imp)
        X_train_imp = scaler.transform(X_train_imp)
        X_test_imp = scaler.transform(X_test_imp)

    if model_name not in _TUNABLE_MODEL_NAMES:
        params = _resolve_model_params(model_name, models_cfg, config["model_param_grid"], seed=0)
        y_pred = _fit_predict(model_name, params, X_train_imp, y_train.to_numpy(), X_test_imp, target_column, X_train.columns)
        return _finalize_metrics(y_test.to_numpy(), y_pred, y_train.to_numpy(), seasonal_forecast_test, metadata)

    candidates = _build_p5_candidates(model_name, models_cfg)

    def score_fn(params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        preds = _fit_predict(model_name, params, X_train_imp, y_train.to_numpy(), X, target_column, X_train.columns)
        return float(mae(y, preds))

    best_params = select_best_params_leaky_on_test(candidates, score_fn, X_test_imp, y_test.to_numpy())
    y_pred = _fit_predict(model_name, best_params, X_train_imp, y_train.to_numpy(), X_test_imp, target_column, X_train.columns)
    return _finalize_metrics(y_test.to_numpy(), y_pred, y_train.to_numpy(), seasonal_forecast_test, metadata)


def _build_p5_candidates(model_name: str, models_cfg: dict[str, Any], max_candidates: int = 3) -> list[dict[str, Any]]:
    """Build up to `max_candidates` hyperparameter dicts from `params_full`, varied
    enough to make model selection meaningful regardless of the active profile --
    P5's leak is about the selection process, not about grid size."""
    full_grid = models_cfg[model_name]["params_full"]
    candidates = []
    for i in range(max_candidates):
        candidate = {k: v[i % len(v)] for k, v in full_grid.items()}
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _run_p6_cv_cell(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    origin_full: pd.Series,
    df: pd.DataFrame,
    target_column: str,
    season_length: int,
    horizon: int,
    model_name: str,
    models_cfg: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any],
    n_folds: int = 5,
) -> dict[str, Any]:
    """P6: metrics averaged over shuffled K-fold CV folds (the CV-estimate audit).

    Compared against P0's single-split metrics by notebook 08; that P0 baseline
    uses a single chronological split rather than CV, a methodological difference
    noted explicitly rather than hidden (see module docstring).
    """
    folds = shuffled_kfold_splits(X_full, n_folds=n_folds, seed=metadata["seed"])
    per_fold_metrics: list[dict[str, float]] = []
    n_train_total = n_test_total = 0

    for train_idx, test_idx in folds:
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
        origin_test = origin_full.iloc[test_idx]
        seasonal_forecast_test = _seasonal_forecast_for_index(df, target_column, season_length, horizon, y_test.index)
        n_train_total += len(X_train)
        n_test_total += len(X_test)

        if model_name == "seasonal_naive":
            y_pred = seasonal_forecast_test
        elif model_name == "persistence":
            y_pred = origin_test.to_numpy()
        else:
            imputer = fit_imputer_correct_train_only(X_train.to_numpy())
            X_train_imp = imputer.transform(X_train.to_numpy())
            X_test_imp = imputer.transform(X_test.to_numpy())
            if model_name in _SCALED_MODEL_NAMES:
                scaler = fit_scaler_correct_train_only(X_train_imp)
                X_train_imp = scaler.transform(X_train_imp)
                X_test_imp = scaler.transform(X_test_imp)
            params = _resolve_model_params(model_name, models_cfg, config["model_param_grid"], metadata["seed"])
            y_pred = _fit_predict(model_name, params, X_train_imp, y_train.to_numpy(), X_test_imp, target_column, X_train.columns)

        fold_metrics = _finalize_metrics(y_test.to_numpy(), y_pred, y_train.to_numpy(), seasonal_forecast_test, {})
        per_fold_metrics.append(fold_metrics)

    averaged = {
        metric: float(np.nanmean([fold[metric] for fold in per_fold_metrics]))
        for metric in ("mae", "rmse", "smape", "mase", "skill_score")
    }
    metadata["n_train"] = n_train_total // len(folds)
    metadata["n_test"] = n_test_total // len(folds)
    metadata["n_folds"] = len(folds)
    return {**metadata, **averaged}


def run_grid(
    datasets: list[str],
    protocols: list[str],
    models: list[str],
    horizons: list[int],
    seeds: list[int],
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    """Run the full experiment grid, checkpointing results after each cell.

    Each (dataset, protocol, model, horizon, seed) cell is written to its own JSON
    file under `checkpoint_dir` immediately after it completes. On a subsequent
    call, any cell whose checkpoint file already exists is loaded from disk instead
    of recomputed, so an interrupted grid resumes where it left off. A cell that
    raises is recorded with an `"error"` field rather than aborting the whole grid.

    Args:
        datasets: Dataset ids to include.
        protocols: Protocol ids to include.
        models: Model names to include.
        horizons: Forecast horizons to include.
        seeds: Random seeds to include.
        config: Merged experiment configuration.
        checkpoint_dir: Directory to write per-cell checkpoint files into.

    Returns:
        List of per-cell result dicts (see `run_single_cell`).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    n_loaded = n_computed = 0

    for dataset_id in datasets:
        for protocol_id in protocols:
            for model_name in models:
                for horizon in horizons:
                    for seed in seeds:
                        key = f"{dataset_id}__{protocol_id}__{model_name}__h{horizon}__s{seed}"
                        checkpoint_path = checkpoint_dir / f"{key}.json"

                        if checkpoint_path.is_file():
                            result = read_json(checkpoint_path)
                            n_loaded += 1
                        else:
                            try:
                                result = run_single_cell(dataset_id, protocol_id, model_name, horizon, seed, config)
                            except Exception as exc:  # noqa: BLE001 - one bad cell must not abort the grid
                                logger.error("Grid cell %s failed: %s", key, exc)
                                result = {
                                    "dataset_id": dataset_id,
                                    "protocol_id": protocol_id,
                                    "model_name": model_name,
                                    "horizon": horizon,
                                    "seed": seed,
                                    "error": str(exc),
                                }
                            write_json(checkpoint_path, result)
                            n_computed += 1

                        results.append(result)

    logger.info(
        "run_grid: %d cells total (%d loaded from checkpoint, %d computed) -> %s",
        len(results),
        n_loaded,
        n_computed,
        to_project_relative(checkpoint_dir),
    )
    return results
