from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from recodai_sifd.pipeline.mask_geometry import dilate_bool_mask


@dataclass(frozen=True)
class MaskFusionConfig:
    # merge condition (robust defaults)
    overlap_intra_thresh: float = 0.30  # |A∩B| / |A|
    iou_thresh: float = 0.15           # |A∩B| / |A∪B|
    dilate_radius_px: int = 2          # helps tiny misalignments
    min_pixels: int = 10               # drop tiny masks early


def _area(m: np.ndarray) -> int:
    return int(np.count_nonzero(m))


def _overlap_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0, 0.0
    a_area = np.count_nonzero(a)
    union = np.count_nonzero(a | b)
    overlap_intra = inter / max(1, a_area)
    iou = inter / max(1, union)
    return float(overlap_intra), float(iou)


def fuse_inter_intra_instances(
    inter_instances: Dict[int, np.ndarray],
    intra_instances: List[np.ndarray],
    cfg: MaskFusionConfig,
) -> Dict[int, np.ndarray]:
    """
    inter_instances: dict[id] -> (H,W) bool
    intra_instances: list of (H,W) bool (already in figure coords)
    returns: merged dict with new ids appended after max existing
    """
    out: Dict[int, np.ndarray] = {k: v.astype(bool, copy=False) for k, v in inter_instances.items()}
    next_id = (max(out.keys()) + 1) if out else 1

    # Pre-dilate inter masks (copy) for robust overlap checks
    inter_for_match = {k: dilate_bool_mask(v, cfg.dilate_radius_px) for k, v in out.items()}

    for m_intra in intra_instances:
        m_intra = m_intra.astype(bool, copy=False)

        if _area(m_intra) < cfg.min_pixels:
            continue

        best_id: Optional[int] = None
        best_score: float = 0.0

        m_intra_match = dilate_bool_mask(m_intra, cfg.dilate_radius_px)

        for iid, m_inter_match in inter_for_match.items():
            overlap_intra, iou = _overlap_stats(m_intra_match, m_inter_match)
            if overlap_intra >= cfg.overlap_intra_thresh or iou >= cfg.iou_thresh:
                score = max(overlap_intra, iou)
                if score > best_score:
                    best_score = score
                    best_id = iid

        if best_id is not None:
            # combine into that inter instance (union in original space)
            out[best_id] = out[best_id] | m_intra
            # refresh match mask for subsequent merges
            inter_for_match[best_id] = dilate_bool_mask(out[best_id], cfg.dilate_radius_px)
        else:
            out[next_id] = m_intra
            inter_for_match[next_id] = dilate_bool_mask(m_intra, cfg.dilate_radius_px)
            next_id += 1

    return out
