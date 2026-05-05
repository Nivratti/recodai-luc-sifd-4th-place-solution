from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from panel_reuse_synth.config import Config
from panel_reuse_synth.crop_utils import sample_crop_box, crop_img
from panel_reuse_synth.augments import augment_panel


def _rect_mask(h: int, w: int, box_xyxy: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if box_xyxy is None:
        return m
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 255
    return m


def generate_full_duplicate(cfg: Config, src_path: Path, src_img: np.ndarray, sample_key: str) -> Optional[Dict]:
    seed = cfg.seed + (abs(hash(sample_key)) % (2**31))
    rng = np.random.default_rng(seed)

    H, W = src_img.shape[:2]
    use_full = rng.random() < float(cfg.sampling.use_full_image_prob)

    if use_full:
        A = src_img.copy()
        crop_box = [0, 0, W, H]
        crop_attempts = 1
    else:
        crop_attempts = 0
        box = None
        for _ in range(int(cfg.sampling.crop.max_tries)):
            crop_attempts += 1
            cand = sample_crop_box(
                W=W, H=H, rng=rng,
                area_range=cfg.sampling.crop.area_range,
                aspect_ratio_range=cfg.sampling.crop.aspect_ratio_range,
                min_side_px=cfg.sampling.crop.min_side_px,
                max_tries=1,
            )
            if cand is None:
                continue
            box = cand
            break
        if box is None:
            return None
        A = crop_img(src_img, box)
        crop_box = box.to_list()

    B = A.copy()

    roll_aug = float(rng.random())
    did_augment = roll_aug < float(cfg.full_duplicate.augment_prob)
    decisions = []
    if did_augment:
        B, decisions = augment_panel(B, rng=rng, cfg=cfg.full_duplicate.augments)

    matchA = (0, 0, A.shape[1], A.shape[0])
    matchB = (0, 0, B.shape[1], B.shape[0])

    A_mask = _rect_mask(A.shape[0], A.shape[1], matchA)
    B_mask = _rect_mask(B.shape[0], B.shape[1], matchB)

    meta = {
        "type": "FULL_DUPLICATE",
        "source_path": str(src_path),
        "sample_key": sample_key,
        "rng_seed": int(seed),

        "panel_source": {
            "use_full_image": bool(use_full),
            "crop_box_xyxy": crop_box,
            "crop_attempts": int(crop_attempts),
        },

        "match_region_A_xyxy": [int(v) for v in matchA],
        "match_region_B_xyxy": [int(v) for v in matchB],

        "augments": {
            "augment_prob": float(cfg.full_duplicate.augment_prob),
            "augment_roll": roll_aug,
            "applied": bool(did_augment),
            "max_ops": int(cfg.full_duplicate.augments.max_ops),
            "allow_geometric": bool(cfg.full_duplicate.augments.allow_geometric),
            "trace": decisions,
        },

        "shapes": {
            "A_hw": [int(A.shape[0]), int(A.shape[1])],
            "B_hw": [int(B.shape[0]), int(B.shape[1])],
        },

        "mask_rule": "FULL_DUPLICATE => full-white masks for A and B",
        "mask_values": {"match": 255, "non_match": 0},
    }

    return {"A_img": A, "B_img": B, "A_mask": A_mask, "B_mask": B_mask, "meta": meta}
