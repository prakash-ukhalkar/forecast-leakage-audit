"""Registry mapping protocol ids (P0..P6) to human descriptions and callables.

Notebooks and the evaluation runner look up protocols by id from
`configs/protocols.yaml`; this module provides the corresponding Python-level
implementation handles so a config-driven experiment grid can dispatch to the
correct splitter/injector functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sdg_leakage_audit.protocols import injectors, splitters


@dataclass(frozen=True)
class ProtocolSpec:
    """A single protocol's id, human description, and implementing callables."""

    protocol_id: str
    name: str
    description: str
    callables: dict[str, Callable[..., Any]]


PROTOCOL_REGISTRY: dict[str, ProtocolSpec] = {
    "P0": ProtocolSpec(
        protocol_id="P0",
        name="Corrected baseline",
        description=(
            "Chronological split with purge gap >= horizon; scaler/imputer fit on "
            "train only; tuning on a validation block preceding test."
        ),
        callables={
            "split": splitters.chronological_split_with_gap,
            "fit_scaler": injectors.fit_scaler_correct_train_only,
            "fit_imputer": injectors.fit_imputer_correct_train_only,
            "rolling_feature": injectors.rolling_feature_correct_causal,
            "lag_feature": injectors.lag_feature_correct_train_fold_only,
            "select_params": injectors.select_best_params_correct_on_validation,
            "cv": splitters.blocked_cv_splits,
        },
    ),
    "P1": ProtocolSpec(
        protocol_id="P1",
        name="Random/shuffled split",
        description="Rows/windows assigned to train/test by random shuffle, ignoring time order.",
        callables={"split": splitters.random_split},
    ),
    "P2": ProtocolSpec(
        protocol_id="P2",
        name="Scaling/imputation fit on full dataset before splitting",
        description="Scaler/imputer fit on train+test concatenated before any split.",
        callables={
            "fit_scaler": injectors.fit_scaler_leaky_full_data,
            "fit_imputer": injectors.fit_imputer_leaky_full_data,
        },
    ),
    "P3": ProtocolSpec(
        protocol_id="P3",
        name="Overlapping windows across train/test boundary",
        description="Sliding windows built across the boundary with no purge gap.",
        callables={"make_windows": injectors.make_windows_leaky_overlapping},
    ),
    "P4": ProtocolSpec(
        protocol_id="P4",
        name="Target-derived/future-informed features",
        description="Centered rolling features and/or lags computed on the full, unsorted series.",
        callables={
            "rolling_feature": injectors.rolling_feature_leaky_centered,
            "lag_feature": injectors.lag_feature_leaky_full_series,
        },
    ),
    "P5": ProtocolSpec(
        protocol_id="P5",
        name="Hyperparameter tuning/model selection on the test set",
        description="Best hyperparameters chosen by scoring directly on the test block.",
        callables={"select_params": injectors.select_best_params_leaky_on_test},
    ),
    "P6": ProtocolSpec(
        protocol_id="P6",
        name="Shuffled k-fold CV instead of blocked/expanding-window CV",
        description="Standard shuffled K-fold CV used in place of a time-series-aware CV scheme.",
        callables={"cv": splitters.shuffled_kfold_splits},
    ),
}


def get_protocol(protocol_id: str) -> ProtocolSpec:
    """Look up a protocol specification by id.

    Args:
        protocol_id: One of "P0".."P6".

    Returns:
        The corresponding `ProtocolSpec`.

    Raises:
        KeyError: If `protocol_id` is not a recognized protocol.
    """
    try:
        return PROTOCOL_REGISTRY[protocol_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown protocol id '{protocol_id}'; known ids: {sorted(PROTOCOL_REGISTRY)}"
        ) from exc
