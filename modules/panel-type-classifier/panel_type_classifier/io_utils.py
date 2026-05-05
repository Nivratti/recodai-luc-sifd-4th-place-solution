from __future__ import annotations

from typing import Any, Union, Optional
from pathlib import Path

import numpy as np
from PIL import Image

ImageLike = Union[str, Path, Image.Image, np.ndarray]

def to_pil_rgb(x: ImageLike) -> Image.Image:
    """Convert image input (path / PIL / numpy) to PIL RGB."""
    if isinstance(x, (str, Path)):
        im = Image.open(str(x)).convert("RGB")
        return im

    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported numpy shape {arr.shape}. Expected HxW, HxWx1, HxWx3, or HxWx4.")

        if arr.shape[2] == 4:
            # RGBA -> RGB by dropping alpha (common in some pipelines)
            arr = arr[:, :, :3]

        # If float, assume 0..1 or 0..255; clip and convert.
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            mx = float(arr.max()) if arr.size else 1.0
            if mx <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)

        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported input type: {type(x)}")
