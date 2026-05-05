from __future__ import annotations

from typing import Iterable, Tuple, Union, overload

import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float], Iterable[int]]


def xyxy_to_xywh(xyxy: ArrayLike) -> np.ndarray:
    """
    Convert [x1,y1,x2,y2] -> [x,y,w,h] where (x,y) is top-left.
    Supports shape (...,4). Returns float32 array.
    """
    a = np.asarray(xyxy, dtype=np.float32)
    if a.shape[-1] != 4:
        raise ValueError(f"xyxy_to_xywh expects (...,4), got {a.shape}")
    out = a.copy()
    out[..., 2] = np.clip(a[..., 2] - a[..., 0], 0.0, None)  # w
    out[..., 3] = np.clip(a[..., 3] - a[..., 1], 0.0, None)  # h
    out[..., 0] = a[..., 0]
    out[..., 1] = a[..., 1]
    return out


def xywh_to_xyxy(xywh: ArrayLike) -> np.ndarray:
    """
    Convert [x,y,w,h] -> [x1,y1,x2,y2]. Supports shape (...,4).
    Returns float32 array.
    """
    a = np.asarray(xywh, dtype=np.float32)
    if a.shape[-1] != 4:
        raise ValueError(f"xywh_to_xyxy expects (...,4), got {a.shape}")
    out = a.copy()
    out[..., 2] = a[..., 0] + np.clip(a[..., 2], 0.0, None)
    out[..., 3] = a[..., 1] + np.clip(a[..., 3], 0.0, None)
    out[..., 0] = a[..., 0]
    out[..., 1] = a[..., 1]
    return out


def clip_xyxy(xyxy: ArrayLike, *, w: int, h: int) -> np.ndarray:
    """
    Clip [x1,y1,x2,y2] into image bounds [0,w],[0,h].
    Returns float32 array.
    """
    a = np.asarray(xyxy, dtype=np.float32)
    if a.shape[-1] != 4:
        raise ValueError(f"clip_xyxy expects (...,4), got {a.shape}")
    out = a.copy()
    out[..., 0] = np.clip(out[..., 0], 0, w)
    out[..., 2] = np.clip(out[..., 2], 0, w)
    out[..., 1] = np.clip(out[..., 1], 0, h)
    out[..., 3] = np.clip(out[..., 3], 0, h)
    return out


def normalize_xyxy(xyxy: ArrayLike, *, w: int, h: int) -> np.ndarray:
    """
    Convert pixel coords [x1,y1,x2,y2] -> normalized [0..1].
    Returns float32 array.
    """
    a = np.asarray(xyxy, dtype=np.float32)
    if a.shape[-1] != 4:
        raise ValueError(f"normalize_xyxy expects (...,4), got {a.shape}")
    out = a.copy()
    out[..., 0] = out[..., 0] / float(w)
    out[..., 2] = out[..., 2] / float(w)
    out[..., 1] = out[..., 1] / float(h)
    out[..., 3] = out[..., 3] / float(h)
    return out


def denormalize_xyxy(nxyxy: ArrayLike, *, w: int, h: int) -> np.ndarray:
    """
    Convert normalized [0..1] coords -> pixel coords.
    Returns float32 array.
    """
    a = np.asarray(nxyxy, dtype=np.float32)
    if a.shape[-1] != 4:
        raise ValueError(f"denormalize_xyxy expects (...,4), got {a.shape}")
    out = a.copy()
    out[..., 0] = out[..., 0] * float(w)
    out[..., 2] = out[..., 2] * float(w)
    out[..., 1] = out[..., 1] * float(h)
    out[..., 3] = out[..., 3] * float(h)
    return out


def crop_to_image_xyxy(crop_xyxy: ArrayLike, crop_offset_xyxy: ArrayLike) -> np.ndarray:
    """
    Map crop-local box -> image-global box by adding crop offset.

    crop_xyxy: box in crop coordinates (0..crop_w/h)
    crop_offset_xyxy: the crop box in image coordinates [x1,y1,x2,y2]
    """
    b = np.asarray(crop_xyxy, dtype=np.float32)
    off = np.asarray(crop_offset_xyxy, dtype=np.float32)
    if b.shape[-1] != 4 or off.shape[-1] != 4:
        raise ValueError(f"crop_to_image_xyxy expects (...,4) inputs, got {b.shape}, {off.shape}")
    out = b.copy()
    out[..., 0] = b[..., 0] + off[..., 0]
    out[..., 2] = b[..., 2] + off[..., 0]
    out[..., 1] = b[..., 1] + off[..., 1]
    out[..., 3] = b[..., 3] + off[..., 1]
    return out


def image_to_crop_xyxy(image_xyxy: ArrayLike, crop_offset_xyxy: ArrayLike) -> np.ndarray:
    """
    Map image-global box -> crop-local box by subtracting crop offset.
    (Does not auto-clip; call clip_xyxy(...) with crop dims if needed.)
    """
    b = np.asarray(image_xyxy, dtype=np.float32)
    off = np.asarray(crop_offset_xyxy, dtype=np.float32)
    if b.shape[-1] != 4 or off.shape[-1] != 4:
        raise ValueError(f"image_to_crop_xyxy expects (...,4) inputs, got {b.shape}, {off.shape}")
    out = b.copy()
    out[..., 0] = b[..., 0] - off[..., 0]
    out[..., 2] = b[..., 2] - off[..., 0]
    out[..., 1] = b[..., 1] - off[..., 1]
    out[..., 3] = b[..., 3] - off[..., 1]
    return out
