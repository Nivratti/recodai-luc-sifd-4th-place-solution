from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path, recursive: bool = True) -> List[Path]:
    root = Path(root)
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return [p for p in root.glob("*") if p.suffix.lower() in IMAGE_EXTS]

def save_bar_counts(values: pd.Series, out_path: Path, title: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = values.value_counts()
    fig = plt.figure(figsize=(max(6, 0.6 * len(counts)), 4))
    ax = fig.add_subplot(111)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _safe_open_rgb(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def _fmt_val(v, decimals: int = 2) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)

def save_samples_grid(
    df: pd.DataFrame,
    category_col: str,
    out_path: Path,
    n_per_cat: int = 8,
    seed: Optional[int] = None,
    path_col: str = "path",
    extra_title_cols: Optional[Sequence[str]] = None,
    decimals: int = 2,
    show_filename: bool = False,
    ncols: int = 4,
    thumb_size: int = 320,
):
    """Save a montage of random samples per category."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df is None or len(df) == 0:
        return

    work = df.copy()
    if seed is not None:
        work = work.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        work = work.sample(frac=1.0).reset_index(drop=True)

    cats = sorted([c for c in work[category_col].dropna().unique().tolist()])
    if not cats:
        return

    images: List[Image.Image] = []
    titles: List[str] = []

    for cat in cats:
        chunk = work[work[category_col] == cat]
        if len(chunk) == 0:
            continue
        take = chunk.head(n_per_cat)
        for _, row in take.iterrows():
            im = _safe_open_rgb(str(row[path_col]))
            if im is None:
                continue
            images.append(im)

            t_parts = [str(cat)]
            if extra_title_cols:
                for c in extra_title_cols:
                    if c in row:
                        t_parts.append(f"{c}={_fmt_val(row[c], decimals)}")
            if show_filename and "filename" in row:
                t_parts.append(str(row["filename"]))
            titles.append("\n".join(t_parts))

    if not images:
        return

    nrows = math.ceil(len(images) / ncols)
    fig = plt.figure(figsize=(ncols * 3.0, nrows * 3.0))
    for i, im in enumerate(images):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.axis("off")
        ax.imshow(im.resize((thumb_size, thumb_size)))
        if i < len(titles):
            ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
