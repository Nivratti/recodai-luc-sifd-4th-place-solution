from __future__ import annotations

from typing import List

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU between one box a (4,) and many boxes b (N,4) in xyxy."""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[2] - a[0]) * np.maximum(0.0, a[3] - a[1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Pure numpy NMS. Returns indices of kept boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    idxs = scores.argsort()[::-1]
    keep: List[int] = []

    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        iou = box_iou_xyxy(boxes[i], boxes[rest])
        idxs = rest[iou <= iou_thres]

    return np.array(keep, dtype=np.int64)


def nms_opencv(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, conf_thres: float = 0.0) -> np.ndarray:
    """OpenCV NMSBoxes (if available). Returns kept indices."""
    if cv2 is None or boxes.size == 0:
        return nms_numpy(boxes, scores, iou_thres)

    # cv2.dnn.NMSBoxes expects xywh
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    scores_list = scores.astype(float).tolist()
    kept = cv2.dnn.NMSBoxes(xywh, scores_list, conf_thres, iou_thres)
    if kept is None or len(kept) == 0:
        return np.zeros((0,), dtype=np.int64)
    kept = np.array(kept).reshape(-1).astype(np.int64)
    return kept
