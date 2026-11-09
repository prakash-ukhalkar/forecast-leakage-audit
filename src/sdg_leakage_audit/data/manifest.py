"""Provenance manifest utilities: hashing, manifest-entry construction, I/O.

`data/MANIFEST.json` tracks, per dataset id, where each raw data file came from and
a content hash so re-downloads can be verified for integrity. This module provides
the building blocks; actual network downloads happen in `download.py` and are
invoked from `notebooks/01_data_acquisition.ipynb` in Phase 2.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sdg_leakage_audit.utils.io import read_json, write_json
from sdg_leakage_audit.utils.logging import get_logger
from sdg_leakage_audit.utils.paths import manifest_path, to_project_relative

logger = get_logger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in chunks.

    Args:
        path: Path to the file to hash.

    Returns:
        Hex-encoded SHA-256 digest string.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_entry(
    dataset_id: str,
    url: str,
    resolved_url: str,
    local_path: Path,
) -> dict[str, Any]:
    """Build a single manifest entry for a downloaded dataset file.

    Args:
        dataset_id: Short identifier matching a key in `configs/datasets.yaml`.
        url: The originally configured URL (or package call) for this dataset.
        resolved_url: The URL/route that actually succeeded (may differ from
            `url` if a fallback route was used).
        local_path: Path to the downloaded file on disk, used to compute size
            and SHA-256 hash.

    Returns:
        A dict with keys: dataset_id, url, resolved_url, date_downloaded (ISO 8601,
        UTC), size_bytes, sha256.
    """
    local_path = Path(local_path)
    return {
        "dataset_id": dataset_id,
        "url": url,
        "resolved_url": resolved_url,
        "date_downloaded": datetime.now(timezone.utc).isoformat(),
        "size_bytes": local_path.stat().st_size,
        "sha256": sha256_of_file(local_path),
    }


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load `data/MANIFEST.json` (or a given path), returning its parsed contents.

    Args:
        path: Optional override path; defaults to `utils.paths.manifest_path()`.

    Returns:
        Manifest dict with keys "datasets", "generated_at", "schema_version".
    """
    target = path or manifest_path()
    if not target.is_file():
        logger.warning("Manifest not found at %s; returning empty template", target)
        return {"datasets": {}, "generated_at": None, "schema_version": 1}
    return read_json(target)


def save_manifest(manifest: dict[str, Any], path: Path | None = None) -> None:
    """Write a manifest dict to `data/MANIFEST.json` (or a given path).

    Args:
        manifest: Manifest dict to persist.
        path: Optional override path; defaults to `utils.paths.manifest_path()`.
    """
    target = path or manifest_path()
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(target, manifest)
    logger.info(
        "Manifest written to %s (%d dataset entries)",
        to_project_relative(target),
        len(manifest.get("datasets", {})),
    )


def update_manifest(
    dataset_id: str,
    url: str,
    resolved_url: str,
    local_path: Path,
    path: Path | None = None,
) -> dict[str, Any]:
    """Add/update one dataset's entry in the manifest and persist it to disk.

    Args:
        dataset_id: Short identifier matching a key in `configs/datasets.yaml`.
        url: The originally configured URL (or package call) for this dataset.
        resolved_url: The URL/route that actually succeeded.
        local_path: Path to the downloaded file on disk.
        path: Optional override manifest path.

    Returns:
        The updated manifest dict.
    """
    manifest = load_manifest(path)
    entry = build_manifest_entry(dataset_id, url, resolved_url, local_path)
    manifest.setdefault("datasets", {})[dataset_id] = entry
    save_manifest(manifest, path)
    return manifest
