from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def paste_crop_mask_to_figure(
    mask_crop: np.ndarray,
    xyxy: Tuple[float, float, float, float],
    figure_size_wh: Tuple[int, int],
) -> np.ndarray:
    """
    Convert a crop-space mask to a full-figure mask by pasting into xyxy.
    - mask_crop: (h,w) bool/uint8
    - xyxy: crop box in figure coords
    - figure_size_wh: (W,H) like PIL
    """
    W, H = int(figure_size_wh[0]), int(figure_size_wh[1])
    out = np.zeros((H, W), dtype=bool)

    x1, y1, x2, y2 = map(int, map(round, xyxy))
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return out

    crop_h = y2 - y1
    crop_w = x2 - x1

    m = mask_crop
    if m.dtype != bool:
        m = m.astype(bool)

    # If model output size differs from crop box size, resize.
    if m.shape[0] != crop_h or m.shape[1] != crop_w:
        if cv2 is None:
            # nearest neighbor resize using numpy indexing fallback (rough)
            m = m.astype(np.uint8)
            yy = (np.linspace(0, m.shape[0] - 1, crop_h)).astype(int)
            xx = (np.linspace(0, m.shape[1] - 1, crop_w)).astype(int)
            m = m[yy][:, xx].astype(bool)
        else:
            m = cv2.resize(m.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    out[y1:y2, x1:x2] = m
    return out


def dilate_bool_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    if cv2 is None:
        return mask  # keep dependency-free fallback
    k = 2 * radius_px + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
