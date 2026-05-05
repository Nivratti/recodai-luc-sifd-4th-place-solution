from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image

from .types import ImageInput


IMG_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))

def default_item_id(x: ImageInput, idx: int) -> str:
    """Create a stable-ish id for an item when user did not provide one."""
    if is_pathlike(x):
        try:
            return str(Path(x))
        except Exception:
            return str(x)
    return f"mem:{idx}"

def pil_from_numpy_rgb(arr: np.ndarray) -> Image.Image:
    """Convert a numpy array into a RGB PIL image.

    Accepts:
      - (H,W) grayscale
      - (H,W,3) RGB
      - (H,W,4) RGBA (alpha dropped)

    Dtypes:
      - uint8 expected
      - float arrays are interpreted as 0..1 if max<=1.01 else 0..255
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.ndim == 3 and a.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported numpy image shape: {a.shape}")

    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            mx = float(np.nanmax(a)) if a.size else 0.0
            if mx <= 1.01:
                a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)

    return Image.fromarray(a, mode="RGB")

def safe_load_image_pil(x: ImageInput) -> Optional[Image.Image]:
    """Load an image input into a PIL.Image safely.

    Supports:
      - filesystem path (str/Path)
      - numpy array
      - PIL.Image

    Returns None on failure (logs a warning).
    """
    try:
        if isinstance(x, Image.Image):
            return x.copy()
        if isinstance(x, np.ndarray):
            return pil_from_numpy_rgb(x)
        # Path-like
        p = Path(x)  # type: ignore[arg-type]
        with Image.open(p) as im:
            return im.copy()
    except Exception as e:
        logger.warning(f"Failed to read image: {x} | {type(e).__name__}: {e}")
        return None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def discover_images(folder: Path, recursive: bool = True, exts: Optional[Sequence[str]] = None) -> List[Path]:
    folder = Path(folder)
    if exts is None:
        exts_set = IMG_EXTS_DEFAULT
    else:
        exts_set = {e.lower().strip() if e.startswith(".") else f".{e.lower().strip()}" for e in exts}

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    it = folder.rglob("*") if recursive else folder.glob("*")
    paths: List[Path] = []
    for p in it:
        if p.is_file() and p.suffix.lower() in exts_set:
            paths.append(p)
    paths.sort()
    return paths


def to_rgb_pil(img: Image.Image, grayscale: bool) -> Image.Image:
    if grayscale:
        return img.convert("L").convert("RGB")
    return img.convert("RGB")


def letterpad_to_square(img: Image.Image, size: int, fill: int = 0) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), (fill, fill, fill))

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def direct_resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BILINEAR)


def pil_to_tensor(img: Image.Image, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def set_determinism(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def topk_indices_per_row(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices per row for a 2D score matrix."""

    if scores.ndim != 2:
        raise ValueError("scores must be 2D")
    Q, N = scores.shape
    if k <= 0:
        return np.empty((Q, 0), dtype=np.int64)
    k = min(int(k), N)

    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(Q)[:, None]
    idx = idx[row, np.argsort(-scores[row, idx], axis=1)]
    return idx.astype(np.int64)


def estimate_scores_bytes(q: int, n: int, dtype_bytes: int = 4) -> int:
    return int(q) * int(n) * int(dtype_bytes)
