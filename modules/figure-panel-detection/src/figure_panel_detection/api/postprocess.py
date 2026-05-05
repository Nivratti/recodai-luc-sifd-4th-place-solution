from __future__ import annotations

from typing import Optional

import numpy as np


def postprocess_det_xyxy_conf_cls(
    det: np.ndarray,
    *,
    image_shape: tuple[int, int],  # (H, W)
    # area filters (either px or fraction of image area)
    min_area_px: Optional[float] = None,
    min_area_frac: Optional[float] = None,
    max_area_px: Optional[float] = None,
    max_area_frac: Optional[float] = None,
    # aspect ratio filters
    min_aspect: Optional[float] = None,
    max_aspect: Optional[float] = None,
    # top-k
    topk: Optional[int] = None,
    topk_per_class: Optional[int] = None,
    # sorting
    sort: str = "conf_desc",  # conf_desc|area_desc|yx
) -> np.ndarray:
    """
    Post-filter and sort detections in Nx6 format: [x1,y1,x2,y2,conf,cls].

    - Filters by area and aspect ratio
    - Optionally keeps top-k overall and/or per class
    - Returns a new array (float32), potentially empty (0x6)
    """
    if det is None or len(det) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det = np.asarray(det, dtype=np.float32)
    if det.ndim != 2 or det.shape[1] != 6:
        raise ValueError(f"det must be Nx6, got {det.shape}")

    H, W = int(image_shape[0]), int(image_shape[1])
    img_area = float(H * W)
    eps = 1e-9

    x1 = det[:, 0]
    y1 = det[:, 1]
    x2 = det[:, 2]
    y2 = det[:, 3]
    w = np.clip(x2 - x1, 0.0, None)
    h = np.clip(y2 - y1, 0.0, None)
    area = w * h
    aspect = w / (h + eps)

    # Resolve frac -> px thresholds
    if min_area_frac is not None:
        thr = float(min_area_frac) * img_area
        min_area_px = thr if min_area_px is None else max(float(min_area_px), thr)
    if max_area_frac is not None:
        thr = float(max_area_frac) * img_area
        max_area_px = thr if max_area_px is None else min(float(max_area_px), thr)

    keep = np.ones((len(det),), dtype=bool)

    if min_area_px is not None:
        keep &= area >= float(min_area_px)
    if max_area_px is not None:
        keep &= area <= float(max_area_px)

    if min_aspect is not None:
        keep &= aspect >= float(min_aspect)
    if max_aspect is not None:
        keep &= aspect <= float(max_aspect)

    det = det[keep]
    if len(det) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Apply per-class topk first (usually more useful)
    if topk_per_class is not None:
        k = int(topk_per_class)
        if k <= 0:
            return np.zeros((0, 6), dtype=np.float32)
        det = _topk_per_class(det, k=k)

    if len(det) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Sort
    det = _sort_det(det, sort=sort)

    # Overall topk
    if topk is not None:
        k = int(topk)
        if k <= 0:
            return np.zeros((0, 6), dtype=np.float32)
        det = det[:k]

    return det.astype(np.float32, copy=False)


def _topk_per_class(det: np.ndarray, *, k: int) -> np.ndarray:
    # det is Nx6 float32
    cls = det[:, 5].astype(np.int32, copy=False)
    conf = det[:, 4]
    keep_idx = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        if len(idx) <= k:
            keep_idx.append(idx)
            continue
        # highest conf first
        order = np.argsort(-conf[idx], kind="mergesort")
        keep_idx.append(idx[order[:k]])
    out_idx = np.concatenate(keep_idx) if keep_idx else np.zeros((0,), dtype=np.int64)
    return det[out_idx]


def _sort_det(det: np.ndarray, *, sort: str) -> np.ndarray:
    sort = (sort or "conf_desc").lower()
    if sort == "conf_desc":
        order = np.argsort(-det[:, 4], kind="mergesort")
        return det[order]
    if sort == "area_desc":
        w = np.clip(det[:, 2] - det[:, 0], 0.0, None)
        h = np.clip(det[:, 3] - det[:, 1], 0.0, None)
        area = w * h
        order = np.argsort(-area, kind="mergesort")
        return det[order]
    if sort == "yx":
        # top-to-bottom then left-to-right based on box top-left (y1, x1)
        y1 = det[:, 1]
        x1 = det[:, 0]
        order = np.lexsort((x1, y1))  # primary y1, secondary x1
        return det[order]
    raise ValueError(f"Unsupported sort={sort!r}. Use: conf_desc|area_desc|yx")
