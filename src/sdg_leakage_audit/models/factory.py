"""Model factory: build a model instance by name using `configs/models.yaml` defaults.

Dispatches to `baselines.py`, `sklearn_models.py`, or `torch_models.py` depending on
the requested model name.
"""

from __future__ import annotations

from typing import Any

from sdg_leakage_audit.models import baselines, sklearn_models, torch_models

_RECURRENT_MODEL_NAMES = {"lstm", "gru"}

_MODEL_FAMILIES = {
    "seasonal_naive": "baseline",
    "persistence": "baseline",
    "ridge": "linear",
    "random_forest": "tree_ensemble",
    "lightgbm": "tree_ensemble",
    "mlp": "neural",
    "lstm": "neural",
    "gru": "neural",
}


def get_model_family(model_name: str) -> str:
    """Return the model family for a given model name, for family-level aggregation.

    Args:
        model_name: One of the keys in `configs/models.yaml`.

    Returns:
        One of "baseline", "linear", "tree_ensemble", "neural".

    Raises:
        ValueError: If `model_name` is not recognized.
    """
    try:
        return _MODEL_FAMILIES[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model_name: {model_name!r}") from exc


def build_model(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """Construct a model instance by name.

    Args:
        model_name: One of the keys in `configs/models.yaml` (e.g. "ridge",
            "random_forest", "lightgbm", "mlp", "lstm", "gru", "seasonal_naive",
            "persistence").
        params: Hyperparameter overrides; defaults come from `configs/models.yaml`.
            For "lstm"/"gru", an `"epochs"` key (if present) is accepted here but
            not passed to the constructor -- `evaluation.runner` reads it back out
            and passes it to `.fit(..., epochs=...)` instead, since epochs is a
            training-time argument, not a model hyperparameter.

    Returns:
        An unfitted model instance (or baseline forecaster) ready for `.fit`/`.predict`
        (`.predict(history, horizon)` for the two baselines; `.fit(X, y)`/`.predict(X)`
        for everything else).

    Raises:
        ValueError: If `model_name` is not recognized.
    """
    params = dict(params or {})

    if model_name == "seasonal_naive":
        return baselines.SeasonalNaiveForecaster(season_length=params["season_length"])
    if model_name == "persistence":
        return baselines.PersistenceForecaster()
    if model_name == "ridge":
        return sklearn_models.build_ridge(**params)
    if model_name == "random_forest":
        return sklearn_models.build_random_forest(**params)
    if model_name == "lightgbm":
        return sklearn_models.build_lightgbm(**params)
    if model_name == "mlp":
        return sklearn_models.build_mlp(**params)
    if model_name in _RECURRENT_MODEL_NAMES:
        constructor_params = {k: v for k, v in params.items() if k != "epochs"}
        cls = torch_models.SmallLSTMForecaster if model_name == "lstm" else torch_models.SmallGRUForecaster
        return cls(**constructor_params)

    raise ValueError(f"Unknown model_name: {model_name!r}")
