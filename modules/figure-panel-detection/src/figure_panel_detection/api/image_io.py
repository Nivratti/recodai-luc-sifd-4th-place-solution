from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None


InputColor = Literal["bgr", "rgb"]


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for figure-panel-detection image IO, but could not be imported. "
            "Install opencv-python. Original error: "
            f"{_CV2_IMPORT_ERROR}"
        )


def load_image_bgr(
    x: Any,
    *,
    input_color: InputColor = "bgr",
) -> np.ndarray:
    """
    Normalize common image inputs into a uint8 BGR numpy array (H,W,3).

    Supports:
    - str / Path: read via cv2.imread
    - numpy array: (H,W), (H,W,3), (H,W,4)
    - PIL.Image.Image: converted to RGB then to BGR (requires Pillow installed)

    Notes:
    - If your numpy array is RGB (e.g. from PIL/matplotlib), pass input_color="rgb".
    """
    _require_cv2()

    if x is None:
        raise ValueError("image input is None")

    if isinstance(x, (str, Path)):
        p = Path(x)
        if not p.exists():
            raise FileNotFoundError(f"Image path not found: {p}")
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"cv2.imread failed to read image: {p}")
        return im

    if _is_pil_image(x):
        rgb = np.array(x.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported numpy image shape: {arr.shape} (expected HxW or HxWxC)")
        _, _, c = arr.shape
        if c == 1:
            return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        if c == 4:
            arr = arr[:, :, :3]
            c = 3
        if c != 3:
            raise ValueError(f"Unsupported numpy channel count: {c} (expected 3 or 4)")
        if input_color == "bgr":
            return arr.astype(np.uint8, copy=False)
        if input_color == "rgb":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Invalid input_color={input_color!r}")

    raise TypeError(
        "Unsupported image input type. Expected path/str, numpy array, or PIL.Image.Image. "
        f"Got: {type(x)}"
    )


def to_rgb(im_bgr: np.ndarray) -> np.ndarray:
    _require_cv2()
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)


def to_pil(im_bgr: np.ndarray):
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required to return PIL images. Install with `pip install pillow`.") from e
    return Image.fromarray(to_rgb(im_bgr))


def _is_pil_image(x: Any) -> bool:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return False
    return isinstance(x, Image.Image)
