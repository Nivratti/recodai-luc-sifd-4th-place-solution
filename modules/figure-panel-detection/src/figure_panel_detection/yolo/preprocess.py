from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    im: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize + pad to square (new_shape,new_shape), like YOLOv5 letterbox.
    Returns: im_lb, gain, (pad_x_left, pad_y_top)
    """
    if im is None or im.size == 0:
        raise ValueError("Empty image passed to letterbox().")

    shape = im.shape[:2]  # (h,w)
    h, w = shape
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (w',h')
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, float(r), (left, top)


def bgr_to_tensor(im_bgr: np.ndarray, fp16: bool = False) -> np.ndarray:
    """BGR HWC uint8 -> NCHW float32/float16 RGB normalized [0,1]."""
    im = im_bgr[:, :, ::-1]  # BGR->RGB
    im = np.ascontiguousarray(im.transpose(2, 0, 1))  # CHW
    im = im.astype(np.float16 if fp16 else np.float32) / 255.0
    return im[None, ...]  # NCHW
