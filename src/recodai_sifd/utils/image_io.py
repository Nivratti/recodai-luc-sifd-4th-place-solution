from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from loguru import logger
import numpy as np
from PIL import Image

_DEFAULT_IMAGE_EXTS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)


def list_images(
    root: Union[str, Path],
    *,
    recursive: bool = True,
    exts: Optional[Sequence[str]] = None,
    sort: bool = True,
) -> List[Path]:
    """
    List image files under a directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory to search.
    recursive : bool, optional
        If True, search recursively (default). If False, only the top level.
    exts : sequence of str or None, optional
        Allowed extensions (case-insensitive). Examples: [".png", ".jpg"] or
        ["png", "jpg"]. If None, a default image extension list is used.
    sort : bool, optional
        If True (default), return paths sorted lexicographically.

    Returns
    -------
    list[pathlib.Path]
        List of matching image file paths.

    Notes
    -----
    - This function logs only warnings/errors (per your utils policy).
    - If `root` does not exist or is not a directory, it logs an error and exits.
    """
    p = Path(root).expanduser()
    try:
        p = p.resolve(strict=False)
    except Exception:
        pass

    if not p.exists():
        logger.error(f"Input dir does not exist: {p}")
        raise SystemExit(2)
    if not p.is_dir():
        logger.error(f"Input path is not a directory: {p}")
        raise SystemExit(2)

    allow = _normalize_exts(exts) if exts is not None else set(_DEFAULT_IMAGE_EXTS)

    it = p.rglob("*") if recursive else p.glob("*")
    out: List[Path] = []
    for fp in it:
        # skip non-files (dirs, symlinks to dirs, etc.)
        if not fp.is_file():
            continue
        if fp.suffix.lower() in allow:
            out.append(fp)

    if not out:
        logger.warning(f"No images found under: {p}")

    if sort:
        out.sort(key=lambda x: str(x))
    return out


def _normalize_exts(exts: Sequence[str]) -> set[str]:
    """
    Normalize extensions to a lowercase set with leading dots.

    Examples
    --------
    ["png", ".JPG"] -> {".png", ".jpg"}
    """
    norm: set[str] = set()
    for e in exts:
        if e is None:
            continue
        s = str(e).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        norm.add(s)
    return norm


def read_image_pil_rgb(path: Union[str, Path]) -> Image.Image:
    """
    Read an image as a PIL Image in RGB mode.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    PIL.Image.Image
        Loaded image converted to RGB.

    Raises
    ------
    SystemExit
        If the file does not exist, is not a file, or cannot be opened/decoded.
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve(strict=False)
    except Exception:
        pass

    if not p.exists():
        logger.error(f"Image file does not exist: {p}")
        raise SystemExit(2)
    if not p.is_file():
        logger.error(f"Image path is not a file: {p}")
        raise SystemExit(2)

    try:
        # Ensure file handle is closed; .copy() detaches from the file stream.
        with Image.open(p) as im:
            return im.convert("RGB").copy()
    except Exception as e:
        logger.error(f"Failed to read image: {p} | {type(e).__name__}: {e}")
        raise SystemExit(2)

def resolve_input_images(input_path: str) -> list[str]:
    """
    If input_path is a file -> return [file]
    If directory -> return list_images(directory)
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if p.is_file():
        return [str(p)]

    # directory
    return list_images(str(p))

