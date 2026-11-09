"""Global random-seed control for reproducibility across runs.

Seeds the stdlib `random` module, `numpy`, and (if installed) `torch`, and sets
deterministic flags where available. Designed to degrade gracefully if optional
dependencies (torch) are not installed yet, which is expected in early Phase 1
environments.
"""

from __future__ import annotations

import os
import random

import numpy as np

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed all known sources of randomness used across this project.

    Seeds (in order): the `PYTHONHASHSEED` environment variable (best-effort, only
    affects future subprocesses since the current interpreter's hash seed is fixed
    at startup), the stdlib `random` module, `numpy.random`, and `torch` (CPU and
    CUDA generators) if `torch` is importable. If `torch` is not installed, that
    step is skipped with a debug-level log message rather than raising.

    Args:
        seed: Non-negative integer seed value.

    Raises:
        ValueError: If `seed` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Favor reproducibility over raw speed; acceptable for small CPU-friendly models.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("torch seeded with seed=%d", seed)
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding (seed=%d applied to random/numpy only)", seed)

    logger.info("Global random seed set to %d", seed)
