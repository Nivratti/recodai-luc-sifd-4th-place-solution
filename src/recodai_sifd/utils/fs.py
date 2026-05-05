from __future__ import annotations

from pathlib import Path
from typing import Union

from loguru import logger


def ensure_dir_exists(path: Union[str, Path], *, purpose: str = "input folder") -> Path:
    """
    Validate that a directory exists.

    This utility logs only errors and raises SystemExit on failure, so callers
    don't need to log the same error again.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path to validate.
    purpose : str, optional
        Human-readable label used in error messages (e.g., "input folder",
        "weights dir", "output dir"). Default is "input folder".

    Returns
    -------
    pathlib.Path
        Resolved directory path.

    Raises
    ------
    SystemExit
        If `path` is empty, does not exist, or is not a directory.
    """
    if path is None or (isinstance(path, str) and not path.strip()):
        logger.error(f"{purpose} path is empty: {path!r}")
        raise SystemExit(2)

    p = Path(path).expanduser()
    try:
        p = p.resolve(strict=False)
    except Exception:
        # Keep best-effort path; don't add extra logs here.
        pass

    if not p.exists():
        logger.error(f"{purpose} does not exist: {p}")
        raise SystemExit(2)

    if not p.is_dir():
        logger.error(f"{purpose} is not a directory: {p}")
        raise SystemExit(2)

    return p
