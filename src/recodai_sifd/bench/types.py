from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PairExample:
    """One inter-panel pair example."""
    pair_id: str
    label: int  # 1=match, 0=no_match

    a_path: Path
    b_path: Path

    a_mask_path: Optional[Path] = None
    b_mask_path: Optional[Path] = None
    meta_path: Optional[Path] = None

    # optional extra metadata loaded from meta.json (or csv)
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MatchPrediction:
    """Standardized prediction returned by a backend."""
    is_match: bool
    score: float  # larger => more confident match
    matched_keypoints: int = 0
    shared_area_a: float = 0.0
    shared_area_b: float = 0.0
    is_flipped: bool = False

    mask_a: Optional[np.ndarray] = None  # uint8 HxW (0/255) or 0/1 ok
    mask_b: Optional[np.ndarray] = None

    # for debugging / additional analysis
    extras: Optional[Dict[str, Any]] = None
