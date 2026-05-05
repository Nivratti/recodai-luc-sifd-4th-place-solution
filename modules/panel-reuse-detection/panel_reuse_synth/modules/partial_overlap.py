from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from panel_reuse_synth.config import Config
from panel_reuse_synth.crop_utils import Box, sample_crop_box, crop_img, intersection_box
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


def _sample_overlap_fracs(rng: np.random.Generator, pattern: str) -> Tuple[float, float]:
    """
    Returns (overlap_w_frac, overlap_h_frac).
    overlap_area_frac = overlap_w_frac * overlap_h_frac.
    """
    pattern = pattern.upper().strip()
    if pattern == "STRIP_OVERLAP":
        if rng.random() < 0.5:
            return float(rng.uniform(0.05, 0.25)), float(rng.uniform(0.60, 0.95))
        else:
            return float(rng.uniform(0.60, 0.95)), float(rng.uniform(0.05, 0.25))
    if pattern == "CORNER_OVERLAP":
        return float(rng.uniform(0.05, 0.25)), float(rng.uniform(0.05, 0.25))
    if pattern == "SHIFTED_VIEW":
        return float(rng.uniform(0.45, 0.90)), float(rng.uniform(0.45, 0.90))
    # OFFSET_CROPS default
    return float(rng.uniform(0.25, 0.85)), float(rng.uniform(0.25, 0.85))


def generate_partial_overlap(cfg: Config, src_path: Path, src_img: np.ndarray, sample_key: str) -> Optional[Dict]:
    seed = cfg.seed + 777 + (abs(hash(sample_key)) % (2**31))
    rng = np.random.default_rng(seed)

    H, W = src_img.shape[:2]

    # Base crop (or full image)
    use_full = rng.random() < float(cfg.partial_overlap.base_crop.use_full_image_prob)
    if use_full:
        base_box_src = Box(0, 0, W, H)
    else:
        base_box_src = None
        for _ in range(int(cfg.partial_overlap.max_tries)):
            cand = sample_crop_box(
                W=W, H=H, rng=rng,
                area_range=cfg.partial_overlap.base_crop.area_range,
                aspect_ratio_range=cfg.partial_overlap.base_crop.aspect_ratio_range,
                min_side_px=cfg.sampling.crop.min_side_px,
                max_tries=1,
            )
            if cand is None:
                continue
            base_box_src = cand
            break
        if base_box_src is None:
            return None

    base_img = crop_img(src_img, base_box_src)
    BH, BW = base_img.shape[:2]
    min_side = int(cfg.sampling.crop.min_side_px)

    probs = _normalize_probs(cfg.partial_overlap.patterns)
    pattern = rng.choice(list(probs.keys()), p=[probs[k] for k in probs])

    # We create two same-size crops with a shift -> overlap but not full containment
    for attempt in range(int(cfg.partial_overlap.max_tries)):
        a = sample_crop_box(
            W=BW, H=BH, rng=rng,
            area_range=cfg.sampling.crop.area_range,
            aspect_ratio_range=cfg.sampling.crop.aspect_ratio_range,
            min_side_px=min_side,
            max_tries=1,
        )
        if a is None:
            continue

        aw, ah = a.w(), a.h()
        if aw < min_side or ah < min_side:
            continue

        owf, ohf = _sample_overlap_fracs(rng, pattern)

        dx = int(round((1.0 - owf) * aw))
        dy = int(round((1.0 - ohf) * ah))
        if dx >= aw or dy >= ah:
            continue

        dx *= -1 if rng.random() < 0.5 else 1
        dy *= -1 if rng.random() < 0.5 else 1

        x1_min = max(0, -dx)
        x1_max = min(BW - aw, BW - aw - dx)
        y1_min = max(0, -dy)
        y1_max = min(BH - ah, BH - ah - dy)
        if x1_max < x1_min or y1_max < y1_min:
            continue

        x1 = int(rng.integers(x1_min, x1_max + 1))
        y1 = int(rng.integers(y1_min, y1_max + 1))

        A_box = Box(x1, y1, x1 + aw, y1 + ah)
        B_box = Box(x1 + dx, y1 + dy, x1 + dx + aw, y1 + dy + ah)

        ib = intersection_box(A_box, B_box)
        if ib is None:
            continue

        overlap_area = ib.area()
        ratioA = overlap_area / max(1.0, A_box.area())
        ratioB = overlap_area / max(1.0, B_box.area())

        if not (cfg.partial_overlap.overlap_ratio_A[0] <= ratioA <= cfg.partial_overlap.overlap_ratio_A[1]):
            continue
        if not (cfg.partial_overlap.overlap_ratio_B[0] <= ratioB <= cfg.partial_overlap.overlap_ratio_B[1]):
            continue

        if cfg.partial_overlap.forbid_containment and (dx == 0 and dy == 0):
            continue

        overlap_A = (ib.x1 - A_box.x1, ib.y1 - A_box.y1, ib.x2 - A_box.x1, ib.y2 - A_box.y1)
        overlap_B = (ib.x1 - B_box.x1, ib.y1 - B_box.y1, ib.x2 - B_box.x1, ib.y2 - B_box.y1)

        A = crop_img(base_img, A_box)
        B = crop_img(base_img, B_box)

        aug_roll = float(rng.random())
        aug_applied = aug_roll < float(cfg.partial_overlap.augments.apply_prob)
        aug_trace = []
        if aug_applied:
            B, aug_trace = augment_panel(B, rng=rng, cfg=cfg.partial_overlap.augments)

        A_mask = _rect_mask(A.shape[0], A.shape[1], overlap_A)
        B_mask = _rect_mask(B.shape[0], B.shape[1], overlap_B)

        meta = {
            "type": "PARTIAL_OVERLAP",
            "source_path": str(src_path),
            "sample_key": sample_key,
            "rng_seed": int(seed),

            "pattern": str(pattern),
            "use_full_image_for_base": bool(use_full),
            "base_box_in_source_xyxy": base_box_src.to_list(),
            "attempt": int(attempt),

            "A_box_in_base_xyxy": A_box.to_list(),
            "B_box_in_base_xyxy": B_box.to_list(),
            "intersection_in_base_xyxy": ib.to_list(),

            "match_region_A_xyxy": [int(v) for v in overlap_A],
            "match_region_B_xyxy": [int(v) for v in overlap_B],

            "overlap": {
                "dx": int(dx), "dy": int(dy),
                "overlap_area": int(overlap_area),
                "ratioA": float(ratioA),
                "ratioB": float(ratioB),
                "overlap_w_frac": float(owf),
                "overlap_h_frac": float(ohf),
            },

            "augments": {
                "apply_prob": float(cfg.partial_overlap.augments.apply_prob),
                "apply_roll": aug_roll,
                "applied": bool(aug_applied),
                "max_ops": int(cfg.partial_overlap.augments.max_ops),
                "allow_geometric": bool(cfg.partial_overlap.augments.allow_geometric),
                "trace": aug_trace,
            },

            "shapes": {
                "A_hw": [int(A.shape[0]), int(A.shape[1])],
                "B_hw": [int(B.shape[0]), int(B.shape[1])],
            },

            "mask_rule": "PARTIAL_OVERLAP => overlap region white in both A and B masks",
            "mask_values": {"match": 255, "non_match": 0},
        }

        return {"A_img": A, "B_img": B, "A_mask": A_mask, "B_mask": B_mask, "meta": meta}

    return None
