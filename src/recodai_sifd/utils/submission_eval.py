from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

# IMPORTANT:
# These functions are copied verbatim from the official RecodAI metric notebook.
# Do not modify them here; just import and call them.
from recodai_sifd.eval.recodai_f1_official import (
    oF1_score,
    rle_encode,
)


Array = np.ndarray


def multichannel_to_mask_list(
    mc: Optional[np.ndarray],
    *,
    channel_axis: int = -1,
    min_pixels: int = 0,
) -> List[np.ndarray]:
    """
    Convert a multi-channel instance tensor into list of 2D binary masks.

    Accepts:
      - None -> []
      - (H,W) -> [mask] if non-empty
      - (H,W,K) if channel_axis=-1
      - (K,H,W) if channel_axis=0

    Returned masks are uint8 with values {0,1}.
    """
    if mc is None:
        return []

    if mc.ndim == 2:
        m = (mc > 0).astype(np.uint8)
        return [m] if int(m.sum()) >= min_pixels and int(m.sum()) > 0 else []

    if mc.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={mc.shape}")

    masks: List[np.ndarray] = []
    if channel_axis == -1:
        for i in range(mc.shape[-1]):
            m = (mc[..., i] > 0).astype(np.uint8)
            if int(m.sum()) >= min_pixels and int(m.sum()) > 0:
                masks.append(m)
    elif channel_axis == 0:
        for i in range(mc.shape[0]):
            m = (mc[i, ...] > 0).astype(np.uint8)
            if int(m.sum()) >= min_pixels and int(m.sum()) > 0:
                masks.append(m)
    else:
        raise ValueError("channel_axis must be -1 or 0")

    return masks


def instance_dict_to_multichannel(
    instance_masks_by_id: Dict[int, np.ndarray],
    *,
    channel_axis: int = -1,
) -> Optional[np.ndarray]:
    """
    Convert {instance_id -> HxW mask} into a multichannel tensor.

    Returns:
      - None if dict is empty
      - (H,W,K) if channel_axis=-1
      - (K,H,W) if channel_axis=0
    """
    if not instance_masks_by_id:
        return None

    masks = [(instance_masks_by_id[k] > 0).astype(np.uint8) for k in sorted(instance_masks_by_id.keys())]
    if channel_axis == -1:
        return np.stack(masks, axis=-1)
    if channel_axis == 0:
        return np.stack(masks, axis=0)
    raise ValueError("channel_axis must be -1 or 0")


def _mask_list_shape(lst: List[np.ndarray]) -> Optional[Tuple[int, int]]:
    """Return (H,W) of the first mask in list, or None if empty."""
    if not lst:
        return None
    return tuple(map(int, lst[0].shape[:2]))  # type: ignore


def _all_masks_same_shape(lst: List[np.ndarray], shape_hw: Tuple[int, int]) -> bool:
    H, W = int(shape_hw[0]), int(shape_hw[1])
    for m in lst:
        if m.shape[:2] != (H, W):
            return False
    return True


def recodai_image_score_from_multichannel(
    pred_mc: Optional[np.ndarray],
    gt_mc: Optional[np.ndarray],
    *,
    pred_channel_axis: int = -1,
    gt_channel_axis: int = -1,
    min_pixels_pred: int = 10,
) -> float:
    """
    Per-image score using official RecodAI logic, but operating DIRECTLY on multichannel arrays.

    IMPORTANT (robustness):
    - If gt_channel_axis is wrong, you can end up slicing GT masks with shape like (1,H) instead of (H,W),
      which later crashes inside the official metric with broadcasting errors.
    - To avoid that, this wrapper:
        1) converts pred/gt to list of instance masks
        2) infers the expected (H,W) from the prediction list (when available)
        3) if GT mask shapes don't match expected, it automatically tries the other axis (0 <-> -1)
        4) if still mismatched, raises a clear ValueError showing shapes.

    Logic (same high-level behavior as RecodAI `score()`):
    - Convert pred/gt to list of instance masks
    - If either side has zero instances -> score=1 only if both sides have zero instances else 0
    - Else score = oF1_score(pred_instances, gt_instances)
    """
    pred_list = multichannel_to_mask_list(pred_mc, channel_axis=pred_channel_axis, min_pixels=min_pixels_pred)

    # GT list with user-selected axis
    gt_list = multichannel_to_mask_list(gt_mc, channel_axis=gt_channel_axis, min_pixels=0)

    pred_empty = (len(pred_list) == 0)
    gt_empty = (len(gt_list) == 0)
    if pred_empty or gt_empty:
        return 1.0 if (pred_empty and gt_empty) else 0.0

    expected = _mask_list_shape(pred_list)
    if expected is None:
        # shouldn't happen because pred_empty handled above
        expected = _mask_list_shape(gt_list)

    if expected is None:
        return 1.0

    # Validate GT shapes; auto-fallback axis if mismatch.
    if not _all_masks_same_shape(gt_list, expected):
        alt_axis = 0 if gt_channel_axis == -1 else -1
        alt_gt_list = multichannel_to_mask_list(gt_mc, channel_axis=alt_axis, min_pixels=0)

        if _all_masks_same_shape(alt_gt_list, expected):
            logger.warning(
                f"[eval] gt_channel_axis={gt_channel_axis} produced mask shape mismatch; "
                f"auto-switching to gt_channel_axis={alt_axis}. "
                f"pred_shape={expected}, gt_mc_shape={None if gt_mc is None else tuple(gt_mc.shape)}"
            )
            gt_list = alt_gt_list
        else:
            # Provide a clear error instead of the cryptic numpy broadcasting error.
            gt_shapes = sorted({tuple(m.shape) for m in gt_list})
            alt_shapes = sorted({tuple(m.shape) for m in alt_gt_list})
            raise ValueError(
                "GT instance mask shapes do not match prediction mask shape.\n"
                f"  pred_mask_shape={expected}\n"
                f"  gt_mc_shape={None if gt_mc is None else tuple(gt_mc.shape)}\n"
                f"  gt_channel_axis={gt_channel_axis} -> unique_mask_shapes={gt_shapes}\n"
                f"  alt_gt_channel_axis={alt_axis} -> unique_mask_shapes={alt_shapes}\n"
                "Fix: pass the correct --gt-channel-axis (0 for (K,H,W), -1 for (H,W,K)), "
                "or store GT in the expected layout."
            )

    return float(oF1_score(pred_list, gt_list))


def recodai_annotation_from_multichannel(
    pred_mc: Optional[np.ndarray],
    *,
    channel_axis: int = -1,
    min_pixels_pred: int = 10,
) -> str:
    """
    Build submission annotation using official rle_encode().

    Returns:
      - "authentic" if no instances after filtering
      - else "<json_rle>;<json_rle>;..."
    """
    pred_list = multichannel_to_mask_list(pred_mc, channel_axis=channel_axis, min_pixels=min_pixels_pred)
    if not pred_list:
        return "authentic"
    return rle_encode(pred_list, fg_val=1)


def load_gt_multichannel(
    gt_mask_dir: Union[str, Path],
    case_id: str,
    *,
    prefer_ext: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Load GT multichannel mask for a given case_id.

    Supported:
      - <case_id>.npy  (recommended; shape (H,W,K) or (K,H,W))
      - <case_id>.npz  (expects key 'mask' OR first array)
    If prefer_ext is provided ('.npy' or '.npz'), tries that first.

    Returns None if not found.
    """
    d = Path(gt_mask_dir)

    candidates: List[Path] = []
    if prefer_ext:
        candidates.append(d / f"{case_id}{prefer_ext}")

    candidates.extend([
        d / f"{case_id}.npy",
        d / f"{case_id}.npz",
    ])

    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".npy":
            return np.load(p)
        if p.suffix == ".npz":
            z = np.load(p)
            if "mask" in z.files:
                return z["mask"]
            # fallback: first key
            return z[z.files[0]]
    return None


@dataclass
class SubmissionWriter:
    """
    Collects (case_id, annotation) rows and writes submission.csv at the end.
    """
    rows: List[Dict[str, str]]

    def __init__(self) -> None:
        self.rows = []

    def add(self, case_id: str, annotation: str) -> None:
        self.rows.append({"case_id": str(case_id), "annotation": str(annotation)})

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def write_csv(self, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df_pred = self.to_dataframe()
        df_pred.to_csv(out_path, index=False)
        logger.info(f"[submission] wrote {out_path} (rows={len(df_pred)})")
        return out_path
