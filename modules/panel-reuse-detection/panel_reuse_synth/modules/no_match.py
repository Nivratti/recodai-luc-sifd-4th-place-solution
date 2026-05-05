from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np

from panel_reuse_synth.config import Config
from panel_reuse_synth.crop_utils import sample_crop_box, crop_img, intersection_area, Box


def _mask_full_black(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _valid_min_side(box: Box, min_side: int) -> bool:
    return box.w() >= min_side and box.h() >= min_side


def _scheme_boxes(W: int, H: int, scheme: str, gap: int) -> List[Box]:
    scheme = scheme.upper().strip()
    boxes: List[Box] = []

    def grid(n: int) -> List[Box]:
        xs = [int(round(i * W / n)) for i in range(n + 1)]
        ys = [int(round(i * H / n)) for i in range(n + 1)]
        out: List[Box] = []
        for r in range(n):
            for c in range(n):
                x1, x2 = xs[c], xs[c + 1]
                y1, y2 = ys[r], ys[r + 1]
                if c > 0: x1 += gap
                if c < n - 1: x2 -= gap
                if r > 0: y1 += gap
                if r < n - 1: y2 -= gap
                out.append(Box(x1, y1, x2, y2))
        return out

    if scheme == "HALVES_LR":
        mid = W // 2
        boxes = [Box(0, 0, max(0, mid - gap), H), Box(min(W, mid + gap), 0, W, H)]
    elif scheme == "HALVES_TB":
        mid = H // 2
        boxes = [Box(0, 0, W, max(0, mid - gap)), Box(0, min(H, mid + gap), W, H)]
    elif scheme == "QUADRANTS_2X2":
        boxes = grid(2)
    elif scheme.startswith("GRID_"):
        try:
            n = int(scheme.split("_")[1].split("X")[0])
        except Exception:
            n = 3
        boxes = grid(max(2, n))

    return boxes


def _pick_pair_from_boxes(boxes: List[Box], scheme: str) -> Optional[Tuple[Box, Box, str]]:
    if len(boxes) < 2:
        return None
    scheme = scheme.upper().strip()

    if scheme == "QUADRANTS_2X2" and len(boxes) >= 4:
        order = [(0, 3), (1, 2), (0, 1), (2, 3), (0, 2), (1, 3)]
        for i, j in order:
            if i < len(boxes) and j < len(boxes):
                return boxes[i], boxes[j], f"{scheme}:pair({i},{j})"
        return None

    candidates = [(0, len(boxes) - 1)]
    if len(boxes) > 2:
        candidates += [(0, 1), (1, len(boxes) - 1), (0, len(boxes) // 2)]
    for i, j in candidates:
        if 0 <= i < len(boxes) and 0 <= j < len(boxes) and i != j:
            return boxes[i], boxes[j], f"{scheme}:pair({i},{j})"

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            return boxes[i], boxes[j], f"{scheme}:pair({i},{j})"
    return None


def generate_no_match(cfg: Config, src_path: Path, src_img: np.ndarray, sample_key: str) -> Optional[Dict]:
    seed = cfg.seed * 31 + (abs(hash(sample_key)) % (2**31))
    rng = np.random.default_rng(seed)

    H, W = src_img.shape[:2]
    min_side = int(cfg.sampling.crop.min_side_px)
    content_filter = cfg.sampling.crop.content_filter  # future hook

    # 1) random disjoint
    boxA = None
    a_attempts = 0
    for _ in range(int(cfg.sampling.crop.max_tries)):
        a_attempts += 1
        cand = sample_crop_box(
            W=W, H=H, rng=rng,
            area_range=cfg.sampling.crop.area_range,
            aspect_ratio_range=cfg.sampling.crop.aspect_ratio_range,
            min_side_px=min_side,
            max_tries=1,
        )
        if cand is None:
            continue
        boxA = cand
        break
    if boxA is None:
        return None

    boxB = None
    b_attempts = 0
    for _ in range(int(cfg.no_match.max_tries)):
        b_attempts += 1
        cand = sample_crop_box(
            W=W, H=H, rng=rng,
            area_range=cfg.sampling.crop.area_range,
            aspect_ratio_range=cfg.sampling.crop.aspect_ratio_range,
            min_side_px=min_side,
            max_tries=1,
        )
        if cand is None:
            continue
        if cfg.no_match.require_disjoint:
            if intersection_area(boxA, cand) == 0:
                boxB = cand
                break
        else:
            boxB = cand
            break

    method = "random_disjoint"
    fallback_trace = {"used": False}

    # 2) fallback schemes
    if boxB is None and cfg.no_match.fallback.enabled:
        gap = int(cfg.no_match.fallback.gap_px)
        for scheme in cfg.no_match.fallback.schemes:
            boxes = _scheme_boxes(W, H, scheme, gap=gap)
            picked = _pick_pair_from_boxes(boxes, scheme)
            if picked is None:
                continue
            bA, bB, desc = picked
            if not (_valid_min_side(bA, min_side) and _valid_min_side(bB, min_side)):
                continue
            if cfg.no_match.require_disjoint and intersection_area(bA, bB) != 0:
                continue
            boxA, boxB = bA, bB
            method = "fallback_scheme"
            fallback_trace = {"used": True, "scheme": scheme, "picked": desc, "gap_px": gap}
            break

    if boxB is None:
        return None

    A = crop_img(src_img, boxA)
    B = crop_img(src_img, boxB)

    A_mask = _mask_full_black(A.shape[0], A.shape[1])
    B_mask = _mask_full_black(B.shape[0], B.shape[1])

    meta = {
        "type": "NO_MATCH",
        "source_path": str(src_path),
        "sample_key": sample_key,
        "rng_seed": int(seed),

        "selection_method": method,
        "fallback": fallback_trace,

        "panel_source": {
            "cropA_xyxy": boxA.to_list(),
            "cropB_xyxy": boxB.to_list(),
            "cropA_attempts": int(a_attempts),
            "cropB_attempts": int(b_attempts),
            "require_disjoint": bool(cfg.no_match.require_disjoint),
            "intersection_area": int(intersection_area(boxA, boxB)),
            "content_filter_future": {
                "enabled": bool(content_filter.enabled),
                "method": str(content_filter.method),
                "min_score": float(content_filter.min_score),
            },
        },

        "match_region_A_xyxy": None,
        "match_region_B_xyxy": None,

        "shapes": {
            "A_hw": [int(A.shape[0]), int(A.shape[1])],
            "B_hw": [int(B.shape[0]), int(B.shape[1])],
        },

        "mask_rule": "NO_MATCH => full-black masks for A and B",
        "mask_values": {"match": 255, "non_match": 0},
    }

    return {"A_img": A, "B_img": B, "A_mask": A_mask, "B_mask": B_mask, "meta": meta}
