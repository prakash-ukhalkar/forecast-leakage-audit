"""Small I/O helpers for reading/writing YAML and JSON configs via pathlib."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict.

    Args:
        path: Path to a `.yaml`/`.yml` file.

    Returns:
        Parsed YAML content as a dict (empty dict if the file is empty).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    return content or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to a YAML file, creating parent directories if needed.

    Args:
        path: Destination path.
        data: Data to serialize.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    logger.debug("Wrote YAML to %s", path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dict.

    Args:
        path: Path to a `.json` file.

    Returns:
        Parsed JSON content as a dict.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any], indent: int = 2) -> None:
    """Write a dict to a JSON file, creating parent directories if needed.

    Args:
        path: Destination path.
        data: Data to serialize.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, sort_keys=False)
    logger.debug("Wrote JSON to %s", path)
