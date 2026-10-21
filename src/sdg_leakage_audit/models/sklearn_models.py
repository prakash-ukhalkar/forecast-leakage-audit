"""Thin wrappers/factories around scikit-learn and LightGBM regressors.

Hyperparameter defaults come from `configs/models.yaml`; this module provides the
Python-level construction logic only (no fitting).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def build_ridge(**params: Any) -> Ridge:
    """Construct a `sklearn.linear_model.Ridge` regressor.

    Args:
        **params: Hyperparameters, e.g. `alpha`.

    Returns:
        An unfitted `Ridge` estimator.
    """
    return Ridge(**params)


def build_random_forest(**params: Any) -> RandomForestRegressor:
    """Construct a `sklearn.ensemble.RandomForestRegressor`.

    Args:
        **params: Hyperparameters, e.g. `n_estimators`, `max_depth`. `n_jobs`
            defaults to -1 (use all cores) unless explicitly overridden.

    Returns:
        An unfitted `RandomForestRegressor` estimator.
    """
    params.setdefault("n_jobs", -1)
    return RandomForestRegressor(**params)


def build_lightgbm(**params: Any) -> lgb.LGBMRegressor:
    """Construct a `lightgbm.LGBMRegressor`.

    Args:
        **params: Hyperparameters, e.g. `n_estimators`, `learning_rate`, `num_leaves`.
            `n_jobs` defaults to -1 and `verbosity` to -1 (silent) unless overridden.

    Returns:
        An unfitted `LGBMRegressor` estimator.
    """
    params.setdefault("n_jobs", -1)
    params.setdefault("verbosity", -1)
    return lgb.LGBMRegressor(**params)


def build_mlp(**params: Any) -> MLPRegressor:
    """Construct a `sklearn.neural_network.MLPRegressor`.

    Args:
        **params: Hyperparameters, e.g. `hidden_layer_sizes`, `alpha`, `max_iter`.
            `early_stopping` defaults to True unless explicitly overridden.

    Returns:
        An unfitted `MLPRegressor` estimator.
    """
    params.setdefault("early_stopping", True)
    return MLPRegressor(**params)
