from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import json

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    root = Path(root)
    exts_l = {e.lower() for e in exts}

    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l]
    else:
        paths = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in exts_l]

    paths.sort()
    return paths


def read_image_rgb(path: Path) -> Optional[np.ndarray]:
    path = Path(path)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_image(path: Path, img: np.ndarray, fmt: str = "png", jpg_quality: int = 95) -> None:
    """
    Writes:
      - RGB images: uint8 HxWx3
      - masks: uint8 HxW (0..255)
    """
    path = Path(path)
    fmt = fmt.lower()

    # Convert to OpenCV expected
    if img.ndim == 2:
        out = img
    elif img.ndim == 3 and img.shape[2] == 3:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image shape for write: {img.shape}")

    if fmt == "png":
        ok, buf = cv2.imencode(".png", out)
    elif fmt in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        raise ValueError(f"Unsupported image format: {fmt}")

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def make_relpath(p: Path, root: Path) -> str:
    p = Path(p)
    root = Path(root)
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def resize_to_wh(img: np.ndarray, w: int, h: int, is_mask: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if img.ndim == 2:
        return cv2.resize(img, (w, h), interpolation=interp)
    return cv2.resize(img, (w, h), interpolation=interp)
