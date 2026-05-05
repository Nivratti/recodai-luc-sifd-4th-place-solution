from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from panel_reuse_synth.config import Config
from panel_reuse_synth.crop_utils import Box, sample_crop_box, crop_img
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


def _normalize_probs(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in d.values())
    if s <= 0:
        return {k: 1.0 / len(d) for k in d}
    return {k: float(v) / s for k, v in d.items()}


def _sample_inner_box(
    big_w: int,
    big_h: int,
    rng: np.random.Generator,
    area_ratio_of_big: Tuple[float, float],
    aspect_range: Tuple[float, float],
    min_side_px: int,
    max_tries: int,
    gap_px: int,
) -> Optional[Box]:
    # sample a crop box in "big image coordinates"
    # target area fraction relative to big area
    bw, bh = big_w, big_h
    if bw < min_side_px or bh < min_side_px:
        return None

    for _ in range(int(max_tries)):
        frac = float(rng.uniform(area_ratio_of_big[0], area_ratio_of_big[1]))
        target_area = frac * (bw * bh)
        aspect = float(rng.uniform(aspect_range[0], aspect_range[1]))

        w = int(round(np.sqrt(target_area * aspect)))
        h = int(round(np.sqrt(target_area / aspect)))

        if w < min_side_px or h < min_side_px:
            continue
        if w > (bw - 2 * gap_px) or h > (bh - 2 * gap_px):
            continue

        x1 = int(rng.integers(gap_px, (bw - gap_px) - w + 1))
        y1 = int(rng.integers(gap_px, (bh - gap_px) - h + 1))
        return Box(x1, y1, x1 + w, y1 + h)

    return None


def generate_full_overlap_crop(cfg: Config, src_path: Path, src_img: np.ndarray, sample_key: str) -> Optional[Dict]:
    seed = cfg.seed + (abs(hash(sample_key)) % (2**31))
    rng = np.random.default_rng(seed)

    H, W = src_img.shape[:2]

    # pick direction
    probs = _normalize_probs(cfg.full_overlap_crop.direction_probs)
    direction = rng.choice(list(probs.keys()), p=[probs[k] for k in probs])

    # pick big crop from source (or full)
    use_full = rng.random() < float(cfg.full_overlap_crop.big_crop.use_full_image_prob)
    big_box_src = None

    if use_full:
        big_box_src = Box(0, 0, W, H)
        big_attempts = 1
    else:
        big_attempts = 0
        for _ in range(int(cfg.full_overlap_crop.max_tries)):
            big_attempts += 1
            cand = sample_crop_box(
                W=W, H=H, rng=rng,
                area_range=cfg.full_overlap_crop.big_crop.area_range,
                aspect_ratio_range=cfg.full_overlap_crop.big_crop.aspect_ratio_range,
                min_side_px=cfg.sampling.crop.min_side_px,
                max_tries=1,
            )
            if cand is None:
                continue
            big_box_src = cand
            break
        if big_box_src is None:
            return None

    big_img = crop_img(src_img, big_box_src)
    big_h, big_w = big_img.shape[:2]

    # sample small box inside big
    inner = _sample_inner_box(
        big_w=big_w, big_h=big_h, rng=rng,
        area_ratio_of_big=cfg.full_overlap_crop.small_in_big.area_ratio_of_big,
        aspect_range=cfg.sampling.crop.aspect_ratio_range,
        min_side_px=cfg.full_overlap_crop.small_in_big.min_side_px,
        max_tries=cfg.full_overlap_crop.max_tries,
        gap_px=cfg.full_overlap_crop.small_in_big.gap_px,
    )
    if inner is None:
        return None

    small_img = crop_img(big_img, inner)

    # optional photometric aug (no geometry if allow_geometric=false)
    aug_roll = float(rng.random())
    aug_applied = aug_roll < float(cfg.full_overlap_crop.augments.apply_prob)
    aug_trace = []
    if aug_applied:
        # apply to small (default) to simulate “inset edited”
        small_img, aug_trace = augment_panel(small_img, rng=rng, cfg=cfg.full_overlap_crop.augments)

    # Build A/B according to direction, and match regions (ALWAYS store per-panel boxes)
    if direction == "A_IN_B":
        A = small_img
        B = big_img
        match_region_A = (0, 0, A.shape[1], A.shape[0])  # full
        match_region_B = (inner.x1, inner.y1, inner.x2, inner.y2)
    else:  # B_IN_A
        A = big_img
        B = small_img
        match_region_A = (inner.x1, inner.y1, inner.x2, inner.y2)
        match_region_B = (0, 0, B.shape[1], B.shape[0])

    A_mask = _rect_mask(A.shape[0], A.shape[1], match_region_A)
    B_mask = _rect_mask(B.shape[0], B.shape[1], match_region_B)

    meta = {
        "type": "FULL_OVERLAP_CROP",
        "source_path": str(src_path),
        "sample_key": sample_key,
        "rng_seed": int(seed),

        "direction": direction,
        "use_full_image_for_big": bool(use_full),
        "big_box_in_source_xyxy": big_box_src.to_list(),
        "inner_box_in_big_xyxy": inner.to_list(),  # always in big coords (before swap)

        # uniform fields for resizing/scaling later:
        "match_region_A_xyxy": [int(v) for v in match_region_A],
        "match_region_B_xyxy": [int(v) for v in match_region_B],

        "augments": {
            "apply_prob": float(cfg.full_overlap_crop.augments.apply_prob),
            "apply_roll": aug_roll,
            "applied": bool(aug_applied),
            "max_ops": int(cfg.full_overlap_crop.augments.max_ops),
            "allow_geometric": bool(cfg.full_overlap_crop.augments.allow_geometric),
            "trace": aug_trace,
        },

        "shapes": {
            "A_hw": [int(A.shape[0]), int(A.shape[1])],
            "B_hw": [int(B.shape[0]), int(B.shape[1])],
        },

        "mask_rule": "FULL_OVERLAP_CROP => small panel full-white; big panel only contained box white",
        "mask_values": {"match": 255, "non_match": 0},
    }

    return {"A_img": A, "B_img": B, "A_mask": A_mask, "B_mask": B_mask, "meta": meta}
