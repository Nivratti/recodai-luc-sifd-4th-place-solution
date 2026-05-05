from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_images(source: Path, recursive: bool = True, exts: Sequence[str] | None = None) -> List[Path]:
    """Collect image files from a directory or a single file path."""
    exts_set = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in (exts or IMG_EXTS)}
    source = Path(source)

    if source.is_file():
        return [source] if source.suffix.lower() in exts_set else []

    if not source.exists():
        return []

    pattern = "**/*" if recursive else "*"
    files = [p for p in source.glob(pattern) if p.is_file() and p.suffix.lower() in exts_set]
    files.sort()
    return files


def rel_path_under(p: Path, root: Path) -> Path:
    """Return path relative to root if possible; else just filename."""
    try:
        return p.resolve().relative_to(root.resolve())
    except Exception:
        return Path(p.name)
