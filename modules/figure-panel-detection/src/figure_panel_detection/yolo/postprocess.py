from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .nms import nms_numpy, nms_opencv


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def scale_boxes_yolov5(
    boxes_xyxy: np.ndarray,
    im0_shape: Tuple[int, int],
    pad: Tuple[int, int],
    gain: float,
) -> np.ndarray:
    """
    Map boxes from letterboxed img to original image.
    im0_shape: (h0,w0) original
    pad: (pad_x_left, pad_y_top)
    gain: scaling gain used in letterbox
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy

    boxes = boxes_xyxy.astype(np.float32).copy()
    px, py = pad
    boxes[:, [0, 2]] -= px
    boxes[:, [1, 3]] -= py
    boxes[:, :4] /= gain

    h0, w0 = im0_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)
    return boxes


def postprocess_yolov5(
    pred: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    classes: Optional[List[int]] = None,
    agnostic_nms: bool = False,
    nms_impl: str = "opencv",  # opencv | numpy
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accepts YOLOv5 ONNX output.

    Supported:
      - raw output: (1, N, 5+nc) or (N, 5+nc) with xywh center
      - already-NMS output: (1, M, 6) or (M, 6) where columns are xyxy, conf, cls

    Returns:
      boxes_xyxy (K,4) in letterbox coords
      scores (K,)
      classes (K,)
    """
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim != 2:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    # Case A: already NMS-ed output [x1 y1 x2 y2 conf cls]
    if pred.shape[1] == 6:
        boxes = pred[:, 0:4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        cls_id = pred[:, 5].astype(np.int64)

        m = scores >= float(conf_thres)
        boxes, scores, cls_id = boxes[m], scores[m], cls_id[m]

        if classes is not None:
            m = np.isin(cls_id, np.array(classes, dtype=np.int64))
            boxes, scores, cls_id = boxes[m], scores[m], cls_id[m]

        order = scores.argsort()[::-1]
        boxes, scores, cls_id = boxes[order], scores[order], cls_id[order]
        if boxes.shape[0] > max_det:
            boxes, scores, cls_id = boxes[:max_det], scores[:max_det], cls_id[:max_det]
        return boxes, scores, cls_id

    # Case B: raw output [x y w h obj cls...]
    nc = pred.shape[1] - 5
    if nc <= 0:
        raise ValueError(f"Output does not look like raw YOLOv5 (need 5+nc). Got: {pred.shape}")

    xywh = pred[:, 0:4].astype(np.float32)
    obj = pred[:, 4:5].astype(np.float32)
    clsP = pred[:, 5:5+nc].astype(np.float32)

    # If logits: apply sigmoid. (Some exports already in [0,1], keep as-is.)
    if obj.min() < 0 or obj.max() > 1 or clsP.min() < 0 or clsP.max() > 1:
        obj = sigmoid(obj)
        clsP = sigmoid(clsP)

    scores_all = obj * clsP  # (N,nc)
    cls_id = scores_all.argmax(axis=1).astype(np.int64)
    scores = scores_all.max(axis=1).astype(np.float32)

    m = scores >= float(conf_thres)
    xywh, scores, cls_id = xywh[m], scores[m], cls_id[m]
    if scores.size == 0:
        return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)

    if classes is not None:
        keep = np.isin(cls_id, np.array(classes, dtype=np.int64))
        xywh, scores, cls_id = xywh[keep], scores[keep], cls_id[keep]
        if scores.size == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)

    # xywh -> xyxy (still in letterbox space)
    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

    # NMS
    if agnostic_nms:
        keep_idx = nms_opencv(boxes, scores, iou_thres, conf_thres) if nms_impl == "opencv" else nms_numpy(boxes, scores, iou_thres)
    else:
        keep_all = []
        for c in np.unique(cls_id):
            idx = np.where(cls_id == c)[0]
            if idx.size == 0:
                continue
            if nms_impl == "opencv":
                kept = nms_opencv(boxes[idx], scores[idx], iou_thres, conf_thres)
            else:
                kept = nms_numpy(boxes[idx], scores[idx], iou_thres)
            keep_all.extend(idx[kept].tolist())
        keep_idx = np.array(sorted(set(keep_all), key=lambda i: float(scores[i]), reverse=True), dtype=np.int64)

    boxes, scores, cls_id = boxes[keep_idx], scores[keep_idx], cls_id[keep_idx]

    order = scores.argsort()[::-1]
    boxes, scores, cls_id = boxes[order], scores[order], cls_id[order]
    if boxes.shape[0] > max_det:
        boxes, scores, cls_id = boxes[:max_det], scores[:max_det], cls_id[:max_det]

    return boxes, scores, cls_id
