from __future__ import annotations

from typing import Optional

import numpy as np


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    box: (4,) xyxy
    boxes: (N,4) xyxy
    returns: (N,) IoU
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) * np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
    union = area_a + area_b - inter + 1e-9

    return (inter / union).astype(np.float32)


def dedup_detections(
    det: Optional[np.ndarray],
    iou_thres: float = 0.90,
    merge: bool = False,
    class_agnostic: bool = False,
) -> Optional[np.ndarray]:
    """
    Deduplicate detections by IoU threshold (after NMS).
    Input det: Nx6 [x1,y1,x2,y2,conf,cls] (float)
    Returns deduplicated det (same format).

    - If merge=False: keep highest-conf box, suppress duplicates (NMS-like).
    - If merge=True: merge duplicates into highest-conf box using conf-weighted average of boxes.
    - If class_agnostic=False: dedup only within same class (recommended).
    """
    if det is None:
        return None
    det = np.asarray(det, dtype=np.float32)
    if det.shape[0] == 0:
        return det

    iou_thres = float(iou_thres)
    if iou_thres <= 0:
        return det

    boxes = det[:, 0:4].copy()
    conf = det[:, 4].copy()
    cls = det[:, 5].astype(np.int32)

    order = np.argsort(-conf)  # high -> low
    keep_idx = []

    while order.size > 0:
        i = int(order[0])
        keep_idx.append(i)

        rest = order[1:]
        if rest.size == 0:
            break

        if class_agnostic:
            cand = rest
        else:
            cand = rest[cls[rest] == cls[i]]

        if cand.size == 0:
            order = rest
            continue

        ious = _iou_one_to_many(boxes[i], boxes[cand])
        dup = cand[ious > iou_thres]

        if merge and dup.size > 0:
            group = np.concatenate(([i], dup))
            w = conf[group]
            wsum = float(np.sum(w)) + 1e-9
            boxes[i] = (boxes[group] * w[:, None]).sum(axis=0) / wsum
            # keep confidence as max (already highest by ordering)
            conf[i] = float(np.max(conf[group]))

        if dup.size > 0:
            # remove suppressed duplicates from rest
            mask = ~np.isin(rest, dup)
            order = rest[mask]
        else:
            order = rest

    out = np.concatenate(
        [boxes[keep_idx], conf[keep_idx, None], cls[keep_idx, None].astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    # keep output sorted by confidence descending (stable)
    out = out[np.argsort(-out[:, 4])]
    return out
