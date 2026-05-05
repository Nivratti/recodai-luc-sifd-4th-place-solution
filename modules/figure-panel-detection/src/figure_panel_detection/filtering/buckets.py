from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def bucket_for_image(has_any_det: bool, has_kept_det: bool, using_keep: bool) -> str:
    """without keep: has_objects / no_objects ; with keep: kept / ignored / no_objects"""
    if using_keep:
        if not has_any_det:
            return "no_objects"
        return "kept" if has_kept_det else "ignored"
    return "has_objects" if has_any_det else "no_objects"


def choose_det_sources(
    bucket: str,
    using_keep: bool,
    keep_apply: str,
    det_all: Optional[np.ndarray],
    det_kept: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns: (det_for_txt, det_for_vis) in ORIGINAL coords.

    Rules when using_keep:
      - kept:
          vis = kept only
          txt = kept only if keep_apply=both else all
      - ignored:
          vis = all
          txt = all
      - no_objects:
          None, None

    When not using_keep: txt=all, vis=all
    """
    if not using_keep:
        return det_all, det_all

    if bucket == "kept":
        det_vis = det_kept
        det_txt = det_kept if keep_apply == "both" else det_all
        return det_txt, det_vis

    if bucket == "ignored":
        return det_all, det_all

    return None, None
