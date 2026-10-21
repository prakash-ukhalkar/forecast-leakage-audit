"""Static mapping from dataset id to SDG tag(s), sourced from `configs/datasets.yaml`.

Reads the config directly rather than hardcoding a duplicate mapping, so this
stays in sync if `configs/datasets.yaml` changes.
"""

from __future__ import annotations

from sdg_leakage_audit.utils.io import read_yaml
from sdg_leakage_audit.utils.paths import configs_dir


def get_sdg_tags(dataset_id: str) -> list[str]:
    """Return the SDG tag(s) associated with a dataset id.

    `configs/datasets.yaml` encodes multi-SDG datasets as an underscore-joined
    string (e.g. `"SDG3_SDG11"`); this splits it into a list.

    Args:
        dataset_id: Key into `configs/datasets.yaml`.

    Returns:
        List of SDG tags, e.g. `["SDG3", "SDG11"]`.

    Raises:
        KeyError: If `dataset_id` is not a key in `configs/datasets.yaml`.
    """
    datasets_cfg = read_yaml(configs_dir() / "datasets.yaml")["datasets"]
    if dataset_id not in datasets_cfg:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}; known: {sorted(datasets_cfg)}")
    return datasets_cfg[dataset_id]["sdg_tag"].split("_")
