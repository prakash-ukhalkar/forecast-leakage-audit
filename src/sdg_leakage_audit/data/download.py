"""Generic, verified file downloading with a UCI-specific fallback strategy.

Implemented now for correctness (Phase 1); not invoked over the network until
`notebooks/01_data_acquisition.ipynb` in Phase 2. Downloads go to a temporary file
first, are verified (HTTP 200 + non-empty body), and only then moved to the final
destination path, so a partially-failed download never corrupts an existing file.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sdg_leakage_audit.utils.logging import get_logger
from sdg_leakage_audit.utils.paths import to_project_relative

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_RETRIES = 3
_CHUNK_SIZE = 1024 * 256  # 256 KiB


def _build_session(retries: int = _DEFAULT_RETRIES) -> requests.Session:
    """Build a `requests.Session` with retry/backoff on transient failures."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_file(
    url: str,
    dest_path: Path,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    retries: int = _DEFAULT_RETRIES,
) -> Path:
    """Download `url` to `dest_path`, verifying success before committing.

    The file is streamed to a temporary location in the destination's parent
    directory, the HTTP status is checked (must be 200), and the file is confirmed
    non-empty. Only then is it atomically moved to `dest_path`. On any failure, the
    temporary file is cleaned up and no partial file is left at `dest_path`.

    Args:
        url: Source URL to download.
        dest_path: Final destination path for the downloaded file.
        timeout: Per-request timeout in seconds.
        retries: Number of retries on transient HTTP errors.

    Returns:
        The `dest_path`, on success.

    Raises:
        requests.HTTPError: If the final response status is not 200.
        RuntimeError: If the downloaded file is empty.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    session = _build_session(retries=retries)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".part")
    tmp_path = Path(tmp_name)

    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"GET {url} returned status {response.status_code}, expected 200"
                )
            with open(tmp_fd, "wb") as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)

        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file from {url} is empty (0 bytes)")

        shutil.move(str(tmp_path), str(dest_path))
        logger.info(
            "Downloaded %s -> %s (%d bytes)",
            url,
            to_project_relative(dest_path),
            dest_path.stat().st_size,
        )
        return dest_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        logger.exception("Download failed for %s", url)
        raise
    finally:
        session.close()


def try_ucimlrepo_then_direct(
    uci_id: int,
    direct_url: str,
    dest_dir: Path,
) -> tuple[Path, str]:
    """Attempt to fetch a UCI dataset via the `ucimlrepo` package, falling back to a
    direct archive.ics.uci.edu static-mirror URL if the package route fails.

    The `ucimlrepo` package route (`fetch_ucirepo(id=uci_id)`) is preferred because it
    returns already-parsed feature/target dataframes with metadata. If it raises
    (e.g. network issue, API change, package not installed), this function falls back
    to downloading the raw zip archive directly via `download_file`.

    Args:
        uci_id: Numeric UCI dataset id (e.g. 235, 374, 360, 501).
        direct_url: Fallback direct-download URL for the dataset's zip archive.
        dest_dir: Directory to save the downloaded/exported file into, for either route.

    Returns:
        A tuple `(local_path, resolved_route)` where `resolved_route` is either
        `"ucimlrepo"` or `"direct_zip_url"`, and `local_path` is a Path to the
        persisted file on disk: a CSV export of the combined features/targets
        dataframe for the `ucimlrepo` route, or the downloaded zip archive for the
        direct-URL fallback.

    Raises:
        RuntimeError: If both the `ucimlrepo` route and the direct-download fallback
            fail.
    """
    dest_dir = Path(dest_dir)

    try:
        import pandas as pd
        from ucimlrepo import fetch_ucirepo

        logger.info("Attempting ucimlrepo route for UCI id=%d", uci_id)
        fetched = fetch_ucirepo(id=uci_id)

        features = fetched.data.features
        targets = fetched.data.targets
        if targets is not None and not targets.empty:
            combined = pd.concat([features, targets], axis=1)
        else:
            combined = features

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"uci_{uci_id}_ucimlrepo.csv"
        combined.to_csv(dest_path, index=False)
        if dest_path.stat().st_size == 0:
            raise RuntimeError(f"ucimlrepo route for id={uci_id} produced an empty CSV")
        logger.info(
            "ucimlrepo route succeeded for UCI id=%d -> %s (%d rows)",
            uci_id,
            to_project_relative(dest_path),
            len(combined),
        )
        return dest_path, "ucimlrepo"
    except Exception as exc:  # noqa: BLE001 - broad on purpose: any failure triggers fallback
        logger.warning(
            "ucimlrepo route failed for UCI id=%d (%s); falling back to direct URL", uci_id, exc
        )

    try:
        dest_path = dest_dir / f"uci_{uci_id}.zip"
        download_file(direct_url, dest_path)
        return dest_path, "direct_zip_url"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Both ucimlrepo and direct-download routes failed for UCI id={uci_id}: {exc}"
        ) from exc
