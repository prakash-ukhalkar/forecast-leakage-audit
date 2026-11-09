"""Project-root discovery and canonical directory accessors.

All paths are returned as `pathlib.Path` objects, resolved relative to the project
root (the directory containing `pyproject.toml`), so this works identically on
Windows and Linux and regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file's location) to find the project root.

    The project root is identified as the first ancestor directory containing a
    `pyproject.toml` file.

    Args:
        start: Directory (or file) to start searching from. Defaults to this file.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If no ancestor directory contains `pyproject.toml`.
    """
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate project root (no pyproject.toml found above {current})"
    )


def _project_dir(*parts: str) -> Path:
    return find_project_root().joinpath(*parts)


def data_raw_dir() -> Path:
    """Return the path to `data/raw/`."""
    return _project_dir("data", "raw")


def data_interim_dir() -> Path:
    """Return the path to `data/interim/`."""
    return _project_dir("data", "interim")


def data_processed_dir() -> Path:
    """Return the path to `data/processed/`."""
    return _project_dir("data", "processed")


def results_tables_dir() -> Path:
    """Return the path to `results/tables/`."""
    return _project_dir("results", "tables")


def results_figures_dir() -> Path:
    """Return the path to `results/figures/`."""
    return _project_dir("results", "figures")


def results_metrics_dir() -> Path:
    """Return the path to `results/metrics/`."""
    return _project_dir("results", "metrics")


def configs_dir() -> Path:
    """Return the path to `configs/`."""
    return _project_dir("configs")


def manifest_path() -> Path:
    """Return the path to `data/MANIFEST.json`."""
    return _project_dir("data", "MANIFEST.json")


def to_project_relative(path: Path) -> Path:
    """Return `path` relative to the project root, for display/logging purposes.

    Absolute paths are machine-specific and must not leak into committed artifacts
    (notebook outputs, logs, result tables). Use this whenever a path derived from
    `find_project_root()` is about to be printed or logged.

    Args:
        path: An absolute (or already-relative) path, typically under the project root.

    Returns:
        `path` relative to the project root if possible, otherwise `path` unchanged.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(find_project_root())
    except ValueError:
        return path
