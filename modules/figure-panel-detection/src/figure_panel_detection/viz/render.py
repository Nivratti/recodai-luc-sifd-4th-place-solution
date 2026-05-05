from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# -----------------------------
# Default class palettes (BGR)
# -----------------------------
DEFAULT_COLOR_MAP = {
    "Blots": [[0, 0, 255], [127, 0, 255], [255, 0, 255]],
    "Microscopy": [[255, 0, 0], [255, 127, 0], [255, 255, 0]],
    "Graphs": [0, 255, 0],
    "Flow Cytometry": [57, 114, 3],
    "Body Imaging": [[0, 127, 255], [0, 255, 255], [0, 127, 127]],
}


def norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())


# -----------------------------
# Color palette utilities
# -----------------------------
def _blend_to_white(bgr: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    a = float(max(0.0, min(1.0, alpha)))
    b, g, r = bgr
    b2 = int(b * (1 - a) + 255 * a)
    g2 = int(g * (1 - a) + 255 * a)
    r2 = int(r * (1 - a) + 255 * a)
    return (b2, g2, r2)


def _blend_to_black(bgr: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    a = float(max(0.0, min(1.0, alpha)))
    b, g, r = bgr
    b2 = int(b * (1 - a))
    g2 = int(g * (1 - a))
    r2 = int(r * (1 - a))
    return (b2, g2, r2)


def _extend_palette(pal: List[Tuple[int, int, int]], needed: int) -> List[Tuple[int, int, int]]:
    """
    Ensure palette has at least `needed` colors.
    Extra colors are derived from base using blend-to-white/black,
    with gradually increasing alpha.
    """
    if needed <= 0:
        return pal
    if len(pal) >= needed:
        return pal

    base = pal[0]
    out = list(pal)
    k = 0
    while len(out) < needed:
        alpha = min(0.85, 0.35 + 0.12 * (k // 2))
        cand = _blend_to_white(base, alpha) if (k % 2 == 0) else _blend_to_black(base, alpha)
        if cand in out:
            alpha = min(0.95, alpha + 0.10)
            cand = _blend_to_white(base, alpha) if (k % 2 == 0) else _blend_to_black(base, alpha)
        out.append(cand)
        k += 1
    return out


def load_color_map(color_map_arg: str) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Returns: norm_key(class_name) -> [base_bgr, alt1_bgr, alt2_bgr]

    Accepts JSON string or path to JSON file.
      - Old format: {"Blots":[B,G,R], ...}  -> auto-derive alt1/alt2
      - New format: {"Blots":[[B,G,R],[B,G,R],[B,G,R]], ...}
    """
    if not color_map_arg:
        data = DEFAULT_COLOR_MAP
    else:
        p = Path(color_map_arg)
        if p.exists() and p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(color_map_arg)

    out: Dict[str, List[Tuple[int, int, int]]] = {}

    def to_bgr(v) -> Optional[Tuple[int, int, int]]:
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return (int(v[0]), int(v[1]), int(v[2]))
        return None

    for k, v in (data or {}).items():
        kk = norm_key(str(k))

        # v can be [B,G,R] or [[B,G,R],[B,G,R],[B,G,R]]
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            base = to_bgr(v)
            if base is None:
                continue
            out[kk] = [base, _blend_to_white(base, 0.35), _blend_to_black(base, 0.35)]
            continue

        if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (list, tuple)) for x in v):
            base = to_bgr(v[0])
            alt1 = to_bgr(v[1])
            alt2 = to_bgr(v[2])
            if base is None:
                continue
            if alt1 is None:
                alt1 = _blend_to_white(base, 0.35)
            if alt2 is None:
                alt2 = _blend_to_black(base, 0.35)
            out[kk] = [base, alt1, alt2]
            continue

    # include defaults for missing keys
    for k, v in DEFAULT_COLOR_MAP.items():
        kk = norm_key(k)
        if kk in out:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            base = (int(v[0]), int(v[1]), int(v[2]))
            out[kk] = [base, _blend_to_white(base, 0.35), _blend_to_black(base, 0.35)]
        elif isinstance(v, (list, tuple)) and len(v) >= 1 and all(isinstance(x, (list, tuple)) for x in v):
            base = (int(v[0][0]), int(v[0][1]), int(v[0][2]))
            pal = [base]
            if len(v) > 1 and len(v[1]) == 3:
                pal.append((int(v[1][0]), int(v[1][1]), int(v[1][2])))
            else:
                pal.append(_blend_to_white(base, 0.35))
            if len(v) > 2 and len(v[2]) == 3:
                pal.append((int(v[2][0]), int(v[2][1]), int(v[2][2])))
            else:
                pal.append(_blend_to_black(base, 0.35))
            out[kk] = pal

    return out


def _fallback_base_color(cls_id: int) -> Tuple[int, int, int]:
    """
    Deterministic vivid-ish fallback color (BGR) to replace YOLOv5 utils.plots.colors().
    """
    # golden ratio hue stepping
    h = (cls_id * 0.618033988749895) % 1.0
    s = 0.85
    v = 0.95
    # HSV -> BGR
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(b * 255), int(g * 255), int(r * 255))


def _palette_for_class(
    cname: str,
    cls_id: int,
    cmap: Dict[str, List[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    k = norm_key(cname)
    if k in cmap:
        return cmap[k]
    base = _fallback_base_color(int(cls_id))
    return [base, _blend_to_white(base, 0.35), _blend_to_black(base, 0.35)]


def rect_touch_or_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], tol: int = 0) -> bool:
    """
    a,b are xyxy (inclusive endpoints).
    tol expands "touch" tolerance by tol pixels.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ax1 -= tol
    ay1 -= tol
    ax2 += tol
    ay2 += tol

    bx1 -= tol
    by1 -= tol
    bx2 += tol
    by2 += tol

    ox = min(ax2, bx2) - max(ax1, bx1)
    oy = min(ay2, by2) - max(ay1, by1)
    return (ox >= 0) and (oy >= 0)


def assign_variant_colors(
    dets: List[Dict[str, Any]],
    names: Dict[int, str],
    cmap: Dict[str, List[Tuple[int, int, int]]],
    touch_tol: int = 1,
) -> None:
    """
    Mutates dets by adding:
      - det["color"] : chosen BGR for this detection (variant palette)
      - det["prio"]  : tuple(conf, area) for ordering

    Rule:
      For each class, build connected components where boxes overlap OR touch (with tolerance).
      In each component:
        - order by highest conf (tie: larger area)
        - assign distinct palette colors: pal[0] to strongest, pal[1], pal[2]... to others
        - if component needs more than provided palette colors, extend by deriving more variants
    """
    n = len(dets)
    if n == 0:
        return

    for d in dets:
        d["prio"] = (float(d["conf"]), int(d["area"]))

    by_cls: Dict[int, List[int]] = {}
    for i, d in enumerate(dets):
        by_cls.setdefault(int(d["cls"]), []).append(i)

    for cls_id, idxs in by_cls.items():
        cname = names.get(int(cls_id), str(cls_id))
        base_pal = _palette_for_class(cname, int(cls_id), cmap)

        if len(idxs) == 1:
            i = idxs[0]
            dets[i]["color"] = base_pal[0]
            continue

        # Union-Find for connected components (touch/overlap)
        parent = {i: i for i in idxs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ai in range(len(idxs)):
            i = idxs[ai]
            bi = dets[i]["box"]
            for bj in range(ai + 1, len(idxs)):
                j = idxs[bj]
                if rect_touch_or_overlap(bi, dets[j]["box"], tol=int(touch_tol)):
                    union(i, j)

        comps: Dict[int, List[int]] = {}
        for i in idxs:
            comps.setdefault(find(i), []).append(i)

        for comp_idxs in comps.values():
            comp_sorted = sorted(
                comp_idxs,
                key=lambda k: (-float(dets[k]["conf"]), -int(dets[k]["area"]), k),
            )

            pal = _extend_palette(base_pal, needed=len(comp_sorted))
            for t, k in enumerate(comp_sorted):
                dets[k]["color"] = pal[t]


# -----------------------
# Dense label fit helpers
# -----------------------
def format_conf_trunc(conf: float, decimals: int) -> str:
    """Truncate confidence to N decimals without rounding up."""
    conf = float(conf)
    if decimals <= 0:
        return str(int(math.floor(conf)))
    factor = 10 ** int(decimals)
    v = math.floor(conf * factor) / factor
    return f"{v:.{decimals}f}"


def fit_label_compact(
    name: str,
    conf: Optional[float],
    max_w: int,
    label_pad: int,
    min_font_scale: float,
    max_font_scale: float,
    hide_conf: bool,
    thickness: int = 1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    min_prefix_chars: int = 2,
) -> Tuple[str, float]:
    """
    Tight rules (ported from detect_recursive.py):
      - Prefer confidence in tight spaces
      - Confidence is TRUNCATED (no rounding up)
      - Try: "name conf(2dp)" -> prefix+conf(2dp, >=2 chars) -> conf(2dp)
             "name conf(1dp)" -> prefix+conf(1dp, >=2 chars) -> conf(1dp)
             name -> prefix -> 1 char name
      - If extremely tiny => ""
    Only max_w width constraint is enforced.
    """
    if max_w <= 2:
        return "", max_font_scale

    def fits(text: str, fs: float) -> bool:
        (tw, _), _ = cv2.getTextSize(text, font, fs, thickness)
        return tw <= max_w

    def try_fit_text(text: str) -> Optional[float]:
        fs = float(max_font_scale)
        for _ in range(30):
            if fits(text, fs):
                return fs
            fs = max(float(min_font_scale), fs - 0.05)
            if fs <= float(min_font_scale):
                break
        return float(min_font_scale) if fits(text, float(min_font_scale)) else None

    def fits_at_minfs(text: str) -> bool:
        return fits(text, float(min_font_scale))

    def best_prefix_with_conf(conf_str: str) -> Optional[str]:
        lo, hi = 1, len(name)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = f"{name[:mid]} {conf_str}"
            if fits_at_minfs(cand):
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            prefix_len = len(best.split(" ")[0])
            if prefix_len < int(min_prefix_chars):
                return None
        return best

    def best_prefix_no_conf() -> Optional[str]:
        lo, hi = 1, len(name)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = f"{name[:mid]}"
            if fits_at_minfs(cand):
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    conf2 = None if (conf is None or hide_conf) else format_conf_trunc(conf, 2)
    conf1 = None if (conf is None or hide_conf) else format_conf_trunc(conf, 1)

    if conf2 is not None:
        full = f"{name} {conf2}"
        fs = try_fit_text(full)
        if fs is not None:
            return full, fs

        pref = best_prefix_with_conf(conf2)
        if pref is not None:
            return pref, float(min_font_scale)

        fs = try_fit_text(conf2)
        if fs is not None:
            return conf2, fs

    if conf1 is not None:
        full = f"{name} {conf1}"
        fs = try_fit_text(full)
        if fs is not None:
            return full, fs

        pref = best_prefix_with_conf(conf1)
        if pref is not None:
            return pref, float(min_font_scale)

        fs = try_fit_text(conf1)
        if fs is not None:
            return conf1, fs

    fs = try_fit_text(name)
    if fs is not None:
        return name, fs

    pref = best_prefix_no_conf()
    if pref is not None:
        return pref, float(min_font_scale)

    one = name[:1]
    fs = try_fit_text(one)
    if fs is not None:
        return one, fs

    return "", float(min_font_scale)


# -------------------------
# Placement/collision utils
# -------------------------
def rects_intersect(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def any_intersect(rect, rects) -> bool:
    for r in rects:
        if rects_intersect(rect, r):
            return True
    return False


def label_rect_from_origin(x_text, y_text, tw, th, baseline, pad):
    """
    Return label BG rect as half-open [x1,y1,x2,y2] consistent with draw_filled_rect_alpha slicing.
    Correct baseline handling:
      top = y_text - th - pad
      bottom = y_text + baseline + pad
    """
    x1 = x_text
    y1 = y_text - th - pad
    x2 = x_text + tw + pad * 2
    y2 = y_text + baseline + pad
    return (x1, y1, x2, y2)


def inside_image(rect, w, h) -> bool:
    x1, y1, x2, y2 = rect
    return x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h


def _too_close_to_any_rect(rect, others, gap_x: int, gap_y: int) -> bool:
    """
    rect, others are half-open rects (x1,y1,x2,y2).
    Too close if dx < gap_x AND dy < gap_y.
    """
    gx = int(max(0, gap_x))
    gy = int(max(0, gap_y))
    if gx == 0 and gy == 0:
        return False

    rx1, ry1, rx2, ry2 = rect
    for bx1, by1, bx2, by2 in others:
        dx = max(bx1 - rx2, rx1 - bx2, 0)
        dy = max(by1 - ry2, ry1 - by2, 0)
        if dx < gx and dy < gy:
            return True
    return False


def choose_label_origin(
    box_xyxy: Tuple[int, int, int, int],
    tw: int,
    th: int,
    baseline: int,
    img_w: int,
    img_h: int,
    other_boxes: List[Tuple[int, int, int, int]],
    placed_label_rects: List[Tuple[int, int, int, int]],
    pad: int,
    label_gap_ratio_w: float = 0.2,
    label_gap_ratio_h: float = 0.7,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]:
    """
    Ported strict ordering:
      Outside: top-mid, left-mid, right-mid, bottom-mid, corners
      Inside:  top-mid, bottom-mid, corners
    Outside gap rules enforced only when ANY other box exists.
    Inside ignores gap ratios.
    """
    x1, y1, x2, y2 = box_xyxy
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    label_w = tw + pad * 2
    label_h = th + baseline + pad * 2

    rw = float(max(0.0, label_gap_ratio_w))
    rh = float(max(0.0, label_gap_ratio_h))
    gap_x = int(math.ceil(label_w * rw)) if rw > 0 else 0
    gap_y = int(math.ceil(label_h * rh)) if rh > 0 else 0

    # other boxes: xyxy inclusive -> half-open
    other_boxes_ho = [(bx1, by1, bx2 + 1, by2 + 1) for (bx1, by1, bx2, by2) in other_boxes]
    has_other_boxes = len(other_boxes_ho) > 0

    # detect whether THIS box overlaps any other box at all (positive intersection)
    this_ho = (x1, y1, x2 + 1, y2 + 1)
    overlaps_any = False
    if has_other_boxes:
        for ob in other_boxes_ho:
            iw = max(0, min(this_ho[2], ob[2]) - max(this_ho[0], ob[0]))
            ih = max(0, min(this_ho[3], ob[3]) - max(this_ho[1], ob[1]))
            if iw * ih > 0:
                overlaps_any = True
                break

    def y_text_for_bg_top(y_bg1: int) -> int:
        return int(y_bg1 + th + pad)

    def y_text_for_bg_bottom(y_bg2: int) -> int:
        return int(y_bg2 - baseline - pad)

    def outside_candidates():
        # 1) top-middle: BG bottom touches y1
        x_bg1 = x1 + (box_w - label_w) // 2
        y_text = y_text_for_bg_bottom(y1)
        yield (x_bg1, y_text), True

        # 2) left-middle
        x_bg1 = x1 - label_w
        y_bg1 = y1 + (box_h - label_h) // 2
        y_text = y_text_for_bg_top(y_bg1)
        yield (x_bg1, y_text), True

        # 3) right-middle
        x_bg1 = x2
        y_bg1 = y1 + (box_h - label_h) // 2
        y_text = y_text_for_bg_top(y_bg1)
        yield (x_bg1, y_text), True

        # 4) bottom-middle
        x_bg1 = x1 + (box_w - label_w) // 2
        y_text = y_text_for_bg_top(y2)
        yield (x_bg1, y_text), True

        # 5) corners: TL, TR, BR, BL
        x_bg1 = x1
        y_text = y_text_for_bg_bottom(y1)
        yield (x_bg1, y_text), True

        x_bg1 = x2 - label_w
        y_text = y_text_for_bg_bottom(y1)
        yield (x_bg1, y_text), True

        x_bg1 = x2 - label_w
        y_text = y_text_for_bg_top(y2)
        yield (x_bg1, y_text), True

        x_bg1 = x1
        y_text = y_text_for_bg_top(y2)
        yield (x_bg1, y_text), True

    def inside_candidates():
        # 1) top-middle inside
        x_bg1 = x1 + (box_w - label_w) // 2
        y_text = y_text_for_bg_top(y1)
        yield (x_bg1, y_text), False

        # 2) bottom-middle inside
        x_bg1 = x1 + (box_w - label_w) // 2
        y_text = y_text_for_bg_bottom(y2)
        yield (x_bg1, y_text), False

        # 3) corners: TL, TR, BR, BL
        x_bg1 = x1
        y_text = y_text_for_bg_top(y1)
        yield (x_bg1, y_text), False

        x_bg1 = x2 - label_w
        y_text = y_text_for_bg_top(y1)
        yield (x_bg1, y_text), False

        x_bg1 = x2 - label_w
        y_text = y_text_for_bg_bottom(y2)
        yield (x_bg1, y_text), False

        x_bg1 = x1
        y_text = y_text_for_bg_bottom(y2)
        yield (x_bg1, y_text), False

    def ok_candidate(
        ox: int,
        oy: int,
        is_outside: bool,
        enforce_gap: bool,
        allow_box_overlap: bool,
    ):
        rect = label_rect_from_origin(ox, oy, tw, th, baseline, pad)
        if not inside_image(rect, img_w, img_h):
            return None

        # Always avoid label-vs-label overlap
        if any_intersect(rect, placed_label_rects):
            return None

        # If not allowing box overlap, reject overlaps with any other box
        if not allow_box_overlap and has_other_boxes:
            if any_intersect(rect, other_boxes_ho):
                return None

        # Outside: enforce gap only when asked (and only meaningful if other boxes exist)
        if is_outside and enforce_gap and has_other_boxes:
            if _too_close_to_any_rect(rect, other_boxes_ho, gap_x=gap_x, gap_y=gap_y):
                return None

        return rect

    # 1) OUTSIDE attempt: enforce gap only if other boxes exist
    outside_enforce_gap = has_other_boxes
    for (ox, oy), is_outside in outside_candidates():
        rect = ok_candidate(ox, oy, is_outside=is_outside, enforce_gap=outside_enforce_gap, allow_box_overlap=False)
        if rect is not None:
            return (ox, oy), rect

    # 2) INSIDE attempt (no gap): avoid box overlap first
    for (ox, oy), is_outside in inside_candidates():
        rect = ok_candidate(ox, oy, is_outside=is_outside, enforce_gap=False, allow_box_overlap=False)
        if rect is not None:
            return (ox, oy), rect

    # 3) INSIDE fallback: allow overlap with boxes if overlap scene anyway
    if overlaps_any:
        for (ox, oy), is_outside in inside_candidates():
            rect = ok_candidate(ox, oy, is_outside=is_outside, enforce_gap=False, allow_box_overlap=True)
            if rect is not None:
                return (ox, oy), rect

    # 4) Final fallback: outside without gap, allow overlap with boxes (still avoid label-label)
    if overlaps_any:
        for (ox, oy), is_outside in outside_candidates():
            rect = ok_candidate(ox, oy, is_outside=is_outside, enforce_gap=False, allow_box_overlap=True)
            if rect is not None:
                return (ox, oy), rect

    return None


# -------------------------
# Drawing helpers (alpha bg)
# -------------------------
def draw_filled_rect_alpha(im: np.ndarray, pt1, pt2, color_bgr, alpha: float):
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha <= 0.0:
        return
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = im.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return

    if alpha >= 1.0:
        cv2.rectangle(im, (x1, y1), (x2, y2), color_bgr, thickness=-1)
        return

    roi = im[y1:y2, x1:x2].copy()
    overlay = np.empty_like(roi)
    overlay[:] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    im[y1:y2, x1:x2] = blended


def draw_box_with_label(
    im: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    label: str,
    color_bgr: Tuple[int, int, int],
    line_thickness: int,
    font_scale: float,
    text_thickness: int,
    text_origin: Optional[Tuple[int, int]] = None,
    pad: int = 2,
    bg_alpha: float = 1.0,
    draw_box: bool = True,
):
    x1, y1, x2, y2 = xyxy
    h, w = im.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if draw_box:
        cv2.rectangle(im, (x1, y1), (x2, y2), color_bgr, line_thickness)

    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    if text_origin is None:
        x_text, y_text = x1, y1 + th + 2
    else:
        x_text, y_text = text_origin

    # Correct baseline handling
    x_bg1 = x_text
    y_bg1 = y_text - th - pad
    x_bg2 = x_text + tw + pad * 2
    y_bg2 = y_text + baseline + pad

    x_bg1 = max(0, min(int(x_bg1), w - 1))
    y_bg1 = max(0, min(int(y_bg1), h - 1))
    x_bg2 = max(0, min(int(x_bg2), w))
    y_bg2 = max(0, min(int(y_bg2), h))

    draw_filled_rect_alpha(im, (x_bg1, y_bg1), (x_bg2, y_bg2), color_bgr, bg_alpha)
    cv2.putText(
        im,
        label,
        (int(x_text) + pad, int(y_text)),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


# -------------------------
# Border conflict rendering
# -------------------------
def _draw_multicolor_dashes_axis(
    im: np.ndarray,
    orient: str,
    fixed: int,
    a: int,
    b: int,
    colors_bgr: List[Tuple[int, int, int]],
    dash_len: int,
    gap_len: int,
):
    """Draw multi-color dashed line on [a..b] (inclusive) at fixed coordinate."""
    if a > b or not colors_bgr:
        return

    dash_len = max(1, int(dash_len))
    gap_len = max(0, int(gap_len))

    pos = int(a)
    ci = 0
    n = len(colors_bgr)

    while pos <= b:
        col = colors_bgr[ci % n]
        seg_end = min(b, pos + dash_len - 1)

        if orient == "h":
            cv2.line(im, (int(pos), int(fixed)), (int(seg_end), int(fixed)), col, 1, cv2.LINE_AA)
        else:
            cv2.line(im, (int(fixed), int(pos)), (int(fixed), int(seg_end)), col, 1, cv2.LINE_AA)

        pos = seg_end + 1 + gap_len
        ci += 1


def draw_borders_with_conflicts(
    im: np.ndarray,
    dets: List[Dict[str, Any]],
    line_thickness: int,
):
    """
    Ported overlap handling:
      - Render thickness as multiple 1px strokes.
      - Split into disjoint segments.
      - If a segment is covered by >1 box edge: draw multi-color dashed interleaving.
    """
    if not dets:
        return

    h, w = im.shape[:2]
    t = int(max(1, line_thickness))

    half1 = t // 2
    half2 = t - half1 - 1
    offsets = list(range(-half1, half2 + 1))  # len = t

    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for det_i, d in enumerate(dets):
        x1, y1, x2, y2 = d["box"]
        x1 = int(max(0, min(int(x1), w - 1)))
        x2 = int(max(0, min(int(x2), w - 1)))
        y1 = int(max(0, min(int(y1), h - 1)))
        y2 = int(max(0, min(int(y2), h - 1)))
        if x2 < x1 or y2 < y1:
            continue

        prio = d.get("prio", (0.0, 0))
        col = d["color"]

        # horizontal strokes (top/bottom)
        for fixed_y, edge_name in ((y1, "top"), (y2, "bottom")):
            for off in offsets:
                fy = int(fixed_y + off)
                if 0 <= fy < h:
                    key = ("h", fy)
                    groups.setdefault(key, []).append({
                        "det": int(det_i),
                        "edge": edge_name,
                        "a": int(x1),
                        "b": int(x2),
                        "color": col,
                        "prio": prio,
                    })

        # vertical strokes (left/right)
        for fixed_x, edge_name in ((x1, "left"), (x2, "right")):
            for off in offsets:
                fx = int(fixed_x + off)
                if 0 <= fx < w:
                    key = ("v", fx)
                    groups.setdefault(key, []).append({
                        "det": int(det_i),
                        "edge": edge_name,
                        "a": int(y1),
                        "b": int(y2),
                        "color": col,
                        "prio": prio,
                    })

    solids: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []

    for (orient, fixed), edges in groups.items():
        if not edges:
            continue

        cuts = set()
        for e in edges:
            cuts.add(int(e["a"]))
            cuts.add(int(e["b"]) + 1)

        m = len(edges)
        for i in range(m):
            ai, bi = int(edges[i]["a"]), int(edges[i]["b"])
            for j in range(i + 1, m):
                aj, bj = int(edges[j]["a"]), int(edges[j]["b"])
                s = max(ai, aj)
                e = min(bi, bj)
                if e >= s:
                    cuts.add(int(s))
                    cuts.add(int(e) + 1)

        cuts = sorted(cuts)
        if len(cuts) < 2:
            continue

        for k in range(len(cuts) - 1):
            s = int(cuts[k])
            e = int(cuts[k + 1] - 1)
            if s > e:
                continue

            cover = [ed for ed in edges if int(ed["a"]) <= s and int(ed["b"]) >= e]
            if not cover:
                continue

            if len(cover) == 1:
                ed = cover[0]
                solids.append({"orient": orient, "fixed": fixed, "a": s, "b": e, "color": ed["color"]})
                continue

            # Dedup by det
            uniq: Dict[int, Dict[str, Any]] = {}
            for ed in cover:
                did = int(ed["det"])
                if did not in uniq:
                    uniq[did] = ed
                else:
                    if (float(ed["prio"][0]), int(ed["prio"][1])) > (float(uniq[did]["prio"][0]), int(uniq[did]["prio"][1])):
                        uniq[did] = ed

            cov = list(uniq.values())
            cov.sort(key=lambda ed: (-float(ed["prio"][0]), -int(ed["prio"][1]), int(ed["det"])))

            colors_cycle = [ed["color"] for ed in cov]
            L = int(e - s + 1)

            if L >= 3 * t:
                dash_len = max(2, min(12, L // max(1, len(colors_cycle) * 3)))
                gap_len = max(1, dash_len // 2)
            else:
                dash_len = 1
                gap_len = 0

            conflicts.append({
                "orient": orient,
                "fixed": fixed,
                "a": s,
                "b": e,
                "colors": colors_cycle,
                "dash_len": int(dash_len),
                "gap_len": int(gap_len),
            })

    for sg in solids:
        orient = sg["orient"]
        fixed = int(sg["fixed"])
        a = int(sg["a"])
        b = int(sg["b"])
        col = sg["color"]
        if orient == "h":
            cv2.line(im, (a, fixed), (b, fixed), col, 1, cv2.LINE_AA)
        else:
            cv2.line(im, (fixed, a), (fixed, b), col, 1, cv2.LINE_AA)

    for sg in conflicts:
        _draw_multicolor_dashes_axis(
            im=im,
            orient=sg["orient"],
            fixed=int(sg["fixed"]),
            a=int(sg["a"]),
            b=int(sg["b"]),
            colors_bgr=sg["colors"],
            dash_len=int(sg["dash_len"]),
            gap_len=int(sg["gap_len"]),
        )


# -------------------------
# Main renderer (ported)
# -------------------------
def render_detections(
    im_vis: np.ndarray,
    det_scaled: Optional[np.ndarray],
    names: Dict[int, str],
    color_map: str,
    line_thickness: int,
    hide_labels: bool,
    hide_conf: bool,
    min_font_scale: float,
    max_font_scale: float,
    label_max_width_ratio: float,
    label_pad: int,
    label_bg_alpha: float,
    touch_tol: int,
    label_gap_ratio_w: float,
    label_gap_ratio_h: float,
) -> None:
    if det_scaled is None or len(det_scaled) == 0:
        return

    det_scaled = np.asarray(det_scaled)

    cmap = load_color_map(color_map)

    dets: List[Dict[str, Any]] = []
    for row in det_scaled:
        x1, y1, x2, y2, conf, cls = row.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 < x1 or y2 < y1:
            continue
        area = max(0, x2 - x1) * max(0, y2 - y1)
        dets.append({"box": (x1, y1, x2, y2), "conf": float(conf), "cls": int(cls), "area": int(area)})

    if not dets:
        return

    # Sort for label placement priority (bigger, higher-conf first)
    dets.sort(key=lambda d: (d["area"], d["conf"]), reverse=True)

    # Assign colors with SAME-CLASS overlap/touch variant palette
    assign_variant_colors(dets=dets, names=names, cmap=cmap, touch_tol=int(touch_tol))

    # PASS 1: draw borders with conflict handling
    draw_borders_with_conflicts(im=im_vis, dets=dets, line_thickness=int(line_thickness))

    # PASS 2: draw labels last
    all_boxes = [d["box"] for d in dets]
    placed_label_rects: List[Tuple[int, int, int, int]] = []

    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d["box"]
        c = d["cls"]
        confv = d["conf"]
        cname = names.get(c, str(c))

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        col = d["color"]

        font_th = max(1, int(line_thickness) - 1)
        fs_cap = min(float(max_font_scale), max(float(min_font_scale), box_h / 80.0))

        max_w = max(1, int(box_w * float(label_max_width_ratio)))

        label_text = ""
        fs = fs_cap
        if not hide_labels:
            label_text, fs = fit_label_compact(
                name=cname,
                conf=confv,
                max_w=max_w,
                label_pad=int(label_pad),
                min_font_scale=float(min_font_scale),
                max_font_scale=float(fs_cap),
                hide_conf=bool(hide_conf),
                thickness=int(font_th),
            )

        if not label_text:
            continue

        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fs, font_th)
        other_boxes = [b for j, b in enumerate(all_boxes) if j != i]

        choice = choose_label_origin(
            box_xyxy=(x1, y1, x2, y2),
            tw=tw,
            th=th,
            baseline=baseline,
            img_w=im_vis.shape[1],
            img_h=im_vis.shape[0],
            other_boxes=other_boxes,
            placed_label_rects=placed_label_rects,
            pad=int(label_pad),
            label_gap_ratio_w=float(label_gap_ratio_w),
            label_gap_ratio_h=float(label_gap_ratio_h),
        )

        if choice is None:
            continue

        text_origin, rect = choice
        placed_label_rects.append(rect)

        draw_box_with_label(
            im=im_vis,
            xyxy=(x1, y1, x2, y2),
            label=label_text,
            color_bgr=col,
            line_thickness=int(line_thickness),
            font_scale=float(fs),
            text_thickness=int(font_th),
            text_origin=text_origin,
            pad=int(label_pad),
            bg_alpha=float(label_bg_alpha),
            draw_box=False,  # borders already drawn
        )
