from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PanelCropItem:
    uid: str
    xyxy: Tuple[float, float, float, float]  # figure coords
    image: Any  # PIL or np.ndarray or similar


def _to_rgb_numpy(img: Any) -> np.ndarray:
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # If float image, you can optionally scale; for now just uint8-cast safely.
        return arr.astype(np.uint8, copy=False)

    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _first_not_none_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def _first_not_none_key(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            v = d[k]
            if v is not None:
                return v
    return None


def iter_panel_crops(panel_crops: Any) -> Iterator[PanelCropItem]:
    """
    Robust adapter over different panel_crops structures.

    Supports:
      - dict[uid] -> item (obj or dict) with xyxy/bbox_xyxy and image/crop/pil/rgb
      - object with .by_uid dict
      - object with .crops list
      - list/tuple of items (obj or dict)
    """
    if panel_crops is None:
        return

    # mapping forms
    by_uid = None
    if isinstance(panel_crops, dict):
        by_uid = panel_crops
    elif hasattr(panel_crops, "by_uid"):
        by_uid = getattr(panel_crops, "by_uid")

    if by_uid is not None:
        for uid, it in by_uid.items():
            if isinstance(it, dict):
                xyxy = _first_not_none_key(it, ("xyxy", "bbox_xyxy"))
                img = _first_not_none_key(it, ("image", "crop", "pil", "rgb"))
            else:
                xyxy = _first_not_none_attr(it, ("xyxy", "bbox_xyxy"))
                img = _first_not_none_attr(it, ("image", "crop", "pil", "rgb"))

            if xyxy is None or img is None:
                continue

            x1, y1, x2, y2 = xyxy
            yield PanelCropItem(uid=str(uid), xyxy=(float(x1), float(y1), float(x2), float(y2)), image=img)
        return

    # list form
    crops = getattr(panel_crops, "crops", None)
    if crops is None and isinstance(panel_crops, (list, tuple)):
        crops = panel_crops

    if crops is not None:
        for i, it in enumerate(crops):
            if isinstance(it, dict):
                uid = it.get("uid") or it.get("panel_uid") or str(i)
                xyxy = _first_not_none_key(it, ("xyxy", "bbox_xyxy"))
                img = _first_not_none_key(it, ("image", "crop", "pil", "rgb"))
            else:
                uid = getattr(it, "uid", None) or getattr(it, "panel_uid", None) or str(i)
                xyxy = _first_not_none_attr(it, ("xyxy", "bbox_xyxy"))
                img = _first_not_none_attr(it, ("image", "crop", "pil", "rgb"))

            if xyxy is None or img is None:
                continue

            x1, y1, x2, y2 = xyxy
            yield PanelCropItem(uid=str(uid), xyxy=(float(x1), float(y1), float(x2), float(y2)), image=img)
        return


def panel_item_to_rgb_numpy(item: PanelCropItem) -> np.ndarray:
    return _to_rgb_numpy(item.image)
