#!/usr/bin/env python3
"""
Binary content mask synthetic generator (NO neighbor fragments).

Mask:
  255 = Content (real image pixels inside sampled content region)
  0   = Non-content (padding, border, canvas/background, gutters)

Gutters here = separator artifacts ONLY in outer canvas margins (lines/strips).
No other panels/images are pasted anywhere -> prevents overlaps.

NumPy 2.x safe, robust paste shape alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    c = hex_color.strip()
    if c.startswith("#"):
        c = c[1:]
    if len(c) != 6:
        raise ValueError(f"Bad hex color: {hex_color}")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def weighted_choice(rng: random.Random, probs: Dict[str, float]) -> str:
    items = list(probs.items())
    keys = [k for k, _ in items]
    weights = [max(0.0, float(w)) for _, w in items]
    s = sum(weights)
    if s <= 0:
        return keys[0]
    r = rng.random() * s
    acc = 0.0
    for k, w in zip(keys, weights):
        acc += w
        if r <= acc:
            return k
    return keys[-1]


def weighted_choice_int(rng: random.Random, probs: Dict[int, float]) -> int:
    items = list(probs.items())
    keys = [k for k, _ in items]
    weights = [max(0.0, float(w)) for _, w in items]
    s = sum(weights)
    if s <= 0:
        return keys[0]
    r = rng.random() * s
    acc = 0.0
    for k, w in zip(keys, weights):
        acc += w
        if r <= acc:
            return k
    return keys[-1]


def list_images(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_l = set(e.lower() for e in exts)
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in exts_l]
    files.sort()
    return files


def safe_read_image(path: Path) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def ensure_hw(img: np.ndarray, target_h: int, target_w: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def paste_where(dst: np.ndarray, src: np.ndarray, mask_bool: np.ndarray) -> None:
    th = min(dst.shape[0], src.shape[0], mask_bool.shape[0])
    tw = min(dst.shape[1], src.shape[1], mask_bool.shape[1])
    if th <= 0 or tw <= 0:
        return
    d = dst[:th, :tw]
    s = src[:th, :tw]
    m = mask_bool[:th, :tw]
    d[m] = s[m]
    dst[:th, :tw] = d


# ----------------------------
# Background / degradations
# ----------------------------
def add_paper_texture(np_rng: np.random.Generator, img: np.ndarray, sigma: float) -> np.ndarray:
    n = np_rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def add_background_gradient(rng: random.Random, img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(0, 2 * math.pi)
    gx = math.cos(angle)
    gy = math.sin(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xx * gx + yy * gy)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)  # NumPy 2.x safe
    grad = (proj * 2 - 1) * (strength * 255.0)
    out = np.clip(img.astype(np.float32) + grad[..., None], 0, 255).astype(np.uint8)
    return out


def apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * contrast + (brightness - 1.0) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(1e-6, gamma)
    lut = (np.linspace(0, 1, 256) ** inv) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return img
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def add_noise(np_rng: np.random.Generator, img: np.ndarray, sigma: float) -> np.ndarray:
    n = np_rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def apply_vignette(rng: random.Random, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx = rng.uniform(0.4 * w, 0.6 * w)
    cy = rng.uniform(0.4 * h, 0.6 * h)
    rx = rng.uniform(0.6 * w, 1.0 * w)
    ry = rng.uniform(0.6 * h, 1.0 * h)
    dist = ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2)
    mask = np.exp(-dist).astype(np.float32)
    mask = (0.75 + 0.25 * mask)
    out = np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)
    return out


def resize_artifact(rng: random.Random, img: np.ndarray, scale_range: Tuple[float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    s = rng.uniform(scale_range[0], scale_range[1])
    nh = max(2, int(h * s))
    nw = max(2, int(w * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def warp_rotate_rgb(img: np.ndarray, angle_deg: float, border_value: Tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def warp_rotate_mask(mask: np.ndarray, angle_deg: float, border_value: int) -> np.ndarray:
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=int(border_value))


# ----------------------------
# Panel shape + content fit
# ----------------------------
def make_rounded_rect_mask(w: int, h: int, radius: int) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(im)
    try:
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    except Exception:
        draw.rectangle([0, 0, w - 1, h - 1], fill=255)
    return np.array(im, dtype=np.uint8)


def fit_cover_strict_ar(rng: random.Random, img_rgb: np.ndarray, target_w: int, target_h: int, crop_jitter: float) -> np.ndarray:
    """
    Resize preserving AR to cover target, then crop. NO padding/filler pixels introduced.
    """
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    h, w = img_rgb.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    max_x = max(0, nw - target_w)
    max_y = max(0, nh - target_h)

    jx = rng.randint(0, max_x) if (max_x > 0 and rng.random() < crop_jitter) else (max_x // 2 if max_x > 0 else 0)
    jy = rng.randint(0, max_y) if (max_y > 0 and rng.random() < crop_jitter) else (max_y // 2 if max_y > 0 else 0)

    crop = resized[jy:jy + target_h, jx:jx + target_w]
    return ensure_hw(crop, target_h, target_w, interp=cv2.INTER_LINEAR)


# ----------------------------
# Sampling
# ----------------------------
def sample_panel_size(rng: random.Random, cfg: Dict[str, Any]) -> Tuple[int, int]:
    long_side = rng.choice(cfg["panel"]["size"]["long_side_choices"])
    ar_min, ar_max = cfg["panel"]["size"]["aspect_ratio_range"]
    ar = rng.uniform(ar_min, ar_max)  # W/H
    if ar >= 1.0:
        W = int(long_side)
        H = int(round(long_side / ar))
    else:
        H = int(long_side)
        W = int(round(long_side * ar))
    return clamp_int(W, 256, 2400), clamp_int(H, 256, 2400)


def sample_canvas_margins(rng: random.Random, cfg: Dict[str, Any], panel_w: int, panel_h: int) -> Tuple[str, int, int, int, int]:
    canvas_cfg = cfg["canvas"]
    mode = weighted_choice(rng, canvas_cfg["mode_probs"])
    per_side = bool(canvas_cfg.get("per_side_margins", True))

    if mode == "none":
        return mode, 0, 0, 0, 0

    if mode == "small":
        lo, hi = canvas_cfg["small_margin_px_range"]
        if per_side:
            ml = rng.randint(lo, hi); mr = rng.randint(lo, hi); mt = rng.randint(lo, hi); mb = rng.randint(lo, hi)
        else:
            v = rng.randint(lo, hi); ml = mr = mt = mb = v
        return mode, ml, mr, mt, mb

    f_lo, f_hi = canvas_cfg["large_margin_frac_range"]
    if per_side:
        ml = int(round(rng.uniform(f_lo, f_hi) * panel_w))
        mr = int(round(rng.uniform(f_lo, f_hi) * panel_w))
        mt = int(round(rng.uniform(f_lo, f_hi) * panel_h))
        mb = int(round(rng.uniform(f_lo, f_hi) * panel_h))
    else:
        f = rng.uniform(f_lo, f_hi)
        ml = int(round(f * panel_w)); mr = int(round(f * panel_w)); mt = int(round(f * panel_h)); mb = int(round(f * panel_h))
    return mode, max(1, ml), max(1, mr), max(1, mt), max(1, mb)


def sample_tight_crop(rng: random.Random, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, int]]:
    tc = cfg["panel"].get("tight_crop", {})
    if rng.random() >= float(tc.get("enabled_prob", 0.0)):
        return False, {}
    sides_probs = {int(k): float(v) for k, v in tc.get("sides_count_probs", {1: 1.0}).items()}
    sides_count = weighted_choice_int(rng, sides_probs)
    sides = ["left", "right", "top", "bottom"]
    rng.shuffle(sides)
    chosen = sides[:sides_count]
    cut_lo, cut_hi = tc.get("cut_px_range", [1, 20])
    cuts = {s: rng.randint(int(cut_lo), int(cut_hi)) for s in chosen}
    return True, cuts


def sample_border_padding_flags(rng: random.Random, cfg: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    sc = cfg["structure_control"]
    both_none = (rng.random() < float(sc["both_none_prob"]))
    if both_none:
        return True, False, False

    border_enabled = True
    padding_enabled = True

    if rng.random() < float(sc.get("allow_only_one_prob", 0.0)):
        if rng.random() < 0.5:
            border_enabled = False
        else:
            padding_enabled = False

    if border_enabled and rng.random() > float(sc["border"].get("enabled_prob", 1.0)):
        border_enabled = False
    if padding_enabled and rng.random() > float(sc["padding"].get("enabled_prob", 1.0)):
        padding_enabled = False

    return False, border_enabled, padding_enabled


def choose_sides(rng: random.Random, k: int) -> List[str]:
    sides = ["left", "right", "top", "bottom"]
    rng.shuffle(sides)
    return sides[:k]


def sample_side_thickness(rng: random.Random, mode_probs: Dict[str, float], small_range: List[int], large_range: List[int]) -> int:
    mode = weighted_choice(rng, mode_probs)
    lo, hi = (small_range if mode == "small" else large_range)
    return int(rng.randint(int(lo), int(hi)))


def sample_border_sides_and_thickness(rng: random.Random, border_cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, int], str]:
    k = weighted_choice_int(rng, {int(k): float(v) for k, v in border_cfg["sides_count_probs"].items()})
    sides = choose_sides(rng, k)

    thick = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    for s in thick.keys():
        if s in sides:
            thick[s] = sample_side_thickness(
                rng,
                border_cfg["thickness_mode_probs"],
                border_cfg["small_px_range"],
                border_cfg["large_px_range"],
            )
    style = weighted_choice(rng, border_cfg["style_probs"])
    return sides, thick, style


def sample_padding_thicknesses(rng: random.Random, pad_cfg: Dict[str, Any]) -> Dict[str, int]:
    mode_probs = pad_cfg["thickness_mode_probs"]
    small_range = pad_cfg["small_px_range"]
    large_range = pad_cfg["large_px_range"]

    if rng.random() < float(pad_cfg.get("asym_prob", 1.0)):
        return {
            "top": sample_side_thickness(rng, mode_probs, small_range, large_range),
            "right": sample_side_thickness(rng, mode_probs, small_range, large_range),
            "bottom": sample_side_thickness(rng, mode_probs, small_range, large_range),
            "left": sample_side_thickness(rng, mode_probs, small_range, large_range),
        }
    v = sample_side_thickness(rng, mode_probs, small_range, large_range)
    return {"top": v, "right": v, "bottom": v, "left": v}


def sample_content_rect(
    rng: random.Random,
    cfg: Dict[str, Any],
    avail_x0: int,
    avail_y0: int,
    avail_x1: int,
    avail_y1: int,
    src_ar: float,
) -> Tuple[int, int, int, int, str]:
    cr = cfg["content_region"]
    mode = weighted_choice(rng, cr["mode_probs"])

    aw = max(1, avail_x1 - avail_x0)
    ah = max(1, avail_y1 - avail_y0)

    if mode == "full":
        return avail_x0, avail_y0, avail_x1, avail_y1, mode

    area_lo, area_hi = cr["area_frac_range"]
    area_frac = rng.uniform(area_lo, area_hi)
    target_area = max(16 * 16, int(area_frac * aw * ah))

    if mode == "random_square":
        ar = 1.0
    else:
        if rng.random() < float(cr.get("match_source_ar_prob", 0.0)):
            ar = float(src_ar)
        else:
            ar_lo, ar_hi = cr["aspect_ratio_range"]
            ar = rng.uniform(ar_lo, ar_hi)
        ar = max(0.2, min(5.0, ar))

    w = int(round(math.sqrt(target_area * ar)))
    h = int(round(target_area / max(1, w)))

    w = clamp_int(w, 16, aw)
    h = clamp_int(h, 16, ah)

    x0 = rng.randint(avail_x0, max(avail_x0, avail_x1 - w))
    y0 = rng.randint(avail_y0, max(avail_y0, avail_y1 - h))
    return x0, y0, x0 + w, y0 + h, mode


# ----------------------------
# Border drawing (visual only)
# ----------------------------
def draw_border_visual(
    rng: random.Random,
    panel_rgb: np.ndarray,
    outer_shape: np.ndarray,
    sides: List[str],
    thick: Dict[str, int],
    style: str,
    border_rgb: Tuple[int, int, int],
    border_cfg: Dict[str, Any],
) -> None:
    h, w = panel_rgb.shape[:2]
    pil = Image.fromarray(panel_rgb, mode="RGB")
    draw = ImageDraw.Draw(pil)

    if style in ("solid", "low_contrast"):
        if "top" in sides and thick["top"] > 0:
            draw.rectangle([0, 0, w - 1, thick["top"] - 1], fill=border_rgb)
        if "bottom" in sides and thick["bottom"] > 0:
            draw.rectangle([0, h - thick["bottom"], w - 1, h - 1], fill=border_rgb)
        if "left" in sides and thick["left"] > 0:
            draw.rectangle([0, 0, thick["left"] - 1, h - 1], fill=border_rgb)
        if "right" in sides and thick["right"] > 0:
            draw.rectangle([w - thick["right"], 0, w - 1, h - 1], fill=border_rgb)
        panel_rgb[:] = np.array(pil, dtype=np.uint8)
        return

    # dashed / broken
    dash_len = rng.randint(*border_cfg["dash_length_px_range"])
    gap_len = rng.randint(*border_cfg["dash_gap_px_range"])
    missing = rng.uniform(*border_cfg["broken_missing_frac_range"]) if style == "broken" else 0.0

    line = np.zeros((h, w), dtype=np.uint8)

    def maybe_line(p0, p1, t):
        if t <= 0:
            return
        if missing > 0 and rng.random() < missing:
            return
        cv2.line(line, p0, p1, 255, thickness=int(t))

    for side in ["top", "bottom"]:
        if side not in sides:
            continue
        t = int(thick.get(side, 0))
        if t <= 0:
            continue
        y = 0 if side == "top" else (h - 1)
        x = 0
        while x < w:
            x2 = min(w - 1, x + dash_len)
            maybe_line((x, y), (x2, y), t)
            x += dash_len + gap_len

    for side in ["left", "right"]:
        if side not in sides:
            continue
        t = int(thick.get(side, 0))
        if t <= 0:
            continue
        x = 0 if side == "left" else (w - 1)
        y = 0
        while y < h:
            y2 = min(h - 1, y + dash_len)
            maybe_line((x, y), (x, y2), t)
            y += dash_len + gap_len

    line = cv2.bitwise_and(line, outer_shape)
    panel_rgb[line > 0] = np.array(border_rgb, dtype=np.uint8)


# ----------------------------
# Gutters (visual only, in canvas margins)
# ----------------------------
def add_gutters(
    rng: random.Random,
    cfg: Dict[str, Any],
    canvas_rgb: np.ndarray,
    canvas_mode: str,
    panel_visible_bbox_xywh: Tuple[int, int, int, int],
    bg_rgb: Tuple[int, int, int],
) -> Dict[str, Any]:
    """
    Draws separator lines/strips ONLY in canvas margin area (outside panel bbox).
    No images/panels are pasted. Mask is unchanged (still non-content).
    """
    gcfg = cfg.get("gutters", {})
    meta: Dict[str, Any] = {"enabled": False, "items": []}

    if canvas_mode == "none":
        return meta

    if rng.random() >= float(gcfg.get("enabled_prob", 0.0)):
        return meta

    x, y, w, h = panel_visible_bbox_xywh
    H, W = canvas_rgb.shape[:2]

    # which margin sides exist in this crop
    sides = []
    if x > 0:
        sides.append("left")
    if x + w < W:
        sides.append("right")
    if y > 0:
        sides.append("top")
    if y + h < H:
        sides.append("bottom")
    if not sides:
        return meta

    meta["enabled"] = True
    k = weighted_choice_int(rng, {int(a): float(b) for a, b in gcfg.get("sides_count_probs", {1: 1.0}).items()})
    rng.shuffle(sides)
    sides = sides[: min(k, len(sides))]

    style = weighted_choice(rng, gcfg.get("style_probs", {"line": 1.0}))

    if style == "line":
        t_lo, t_hi = gcfg.get("line_thickness_px_range", [1, 2])
        thickness = rng.randint(int(t_lo), int(t_hi))
    else:
        t_lo, t_hi = gcfg.get("strip_thickness_px_range", [4, 20])
        thickness = rng.randint(int(t_lo), int(t_hi))

    # color selection
    col = np.array(hex_to_rgb(rng.choice(gcfg.get("colors", ["#ffffff"]))), dtype=np.int32)

    if bool(gcfg.get("contrast_with_background", True)):
        bg = np.array(bg_rgb, dtype=np.int32)
        # if too close to background, push a bit lighter/darker
        if int(np.abs(col - bg).mean()) < 8:
            delta = 20 if int(bg.mean()) < 128 else -20
            col = np.clip(bg + delta, 0, 255)

    col_t = (int(col[0]), int(col[1]), int(col[2]))

    for side in sides:
        if side == "left":
            # draw within left margin only
            x0 = 0
            x1 = min(x, W)
            if x1 <= x0:
                continue
            if style == "line":
                xx = rng.randint(x0, max(x0, x1 - 1))
                cv2.line(canvas_rgb, (xx, 0), (xx, H - 1), col_t, thickness=thickness)
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "pos": xx})
            else:
                # strip near panel boundary
                strip_x1 = x
                strip_x0 = max(0, strip_x1 - thickness)
                canvas_rgb[:, strip_x0:strip_x1] = col_t
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "rect": [strip_x0, 0, strip_x1 - strip_x0, H]})

        elif side == "right":
            x0 = x + w
            x1 = W
            if x1 <= x0:
                continue
            if style == "line":
                xx = rng.randint(x0, max(x0, x1 - 1))
                cv2.line(canvas_rgb, (xx, 0), (xx, H - 1), col_t, thickness=thickness)
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "pos": xx})
            else:
                strip_x0 = x + w
                strip_x1 = min(W, strip_x0 + thickness)
                canvas_rgb[:, strip_x0:strip_x1] = col_t
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "rect": [strip_x0, 0, strip_x1 - strip_x0, H]})

        elif side == "top":
            y0 = 0
            y1 = min(y, H)
            if y1 <= y0:
                continue
            if style == "line":
                yy = rng.randint(y0, max(y0, y1 - 1))
                cv2.line(canvas_rgb, (0, yy), (W - 1, yy), col_t, thickness=thickness)
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "pos": yy})
            else:
                strip_y1 = y
                strip_y0 = max(0, strip_y1 - thickness)
                canvas_rgb[strip_y0:strip_y1, :] = col_t
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "rect": [0, strip_y0, W, strip_y1 - strip_y0]})

        else:  # bottom
            y0 = y + h
            y1 = H
            if y1 <= y0:
                continue
            if style == "line":
                yy = rng.randint(y0, max(y0, y1 - 1))
                cv2.line(canvas_rgb, (0, yy), (W - 1, yy), col_t, thickness=thickness)
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "pos": yy})
            else:
                strip_y0 = y + h
                strip_y1 = min(H, strip_y0 + thickness)
                canvas_rgb[strip_y0:strip_y1, :] = col_t
                meta["items"].append({"side": side, "style": style, "thickness": thickness, "rect": [0, strip_y0, W, strip_y1 - strip_y0]})

    return meta


# ----------------------------
# Output bundle
# ----------------------------
@dataclass
class SampleOutputs:
    image_rgb: np.ndarray
    bin_mask: np.ndarray
    metadata: Dict[str, Any]


# ----------------------------
# One sample
# ----------------------------
def synthesize_one(cfg: Dict[str, Any], rng: random.Random, real_images: List[Path]) -> SampleOutputs:
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))

    content_val = int(cfg["mask"]["content_value"])
    non_val = int(cfg["mask"]["non_content_value"])

    panel_w, panel_h = sample_panel_size(rng, cfg)
    canvas_mode, ml, mr, mt, mb = sample_canvas_margins(rng, cfg, panel_w, panel_h)
    tight_crop_enabled, cuts = sample_tight_crop(rng, cfg)

    x0 = ml - cuts.get("left", 0)
    y0 = mt - cuts.get("top", 0)

    W = panel_w + ml + mr
    H = panel_h + mt + mb

    bg_cfg = cfg["canvas"]["background"]
    bg_rgb = hex_to_rgb(rng.choice(bg_cfg["base_colors"]))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:, :] = np.array(bg_rgb, dtype=np.uint8)

    if rng.random() < float(bg_cfg["paper_texture_prob"]):
        canvas = add_paper_texture(np_rng, canvas, float(bg_cfg["paper_noise_sigma"]))
    if rng.random() < float(bg_cfg["gradient_prob"]):
        canvas = add_background_gradient(rng, canvas, float(bg_cfg["gradient_strength"]))

    bin_mask = np.full((H, W), non_val, dtype=np.uint8)

    main_src_path = rng.choice(real_images)
    main_src = safe_read_image(main_src_path)
    if main_src is None:
        main_src = np.zeros((256, 256, 3), dtype=np.uint8)
    src_ar = main_src.shape[1] / max(1, main_src.shape[0])

    both_none, border_enabled, padding_enabled = sample_border_padding_flags(rng, cfg)

    rounded = (rng.random() < float(cfg["panel"]["rounded_prob"]))
    radius = 0
    if rounded:
        rmin, rmax = cfg["panel"]["corner_radius_frac_range"]
        radius = int(rng.uniform(rmin, rmax) * min(panel_w, panel_h))
    outer_shape = make_rounded_rect_mask(panel_w, panel_h, radius)

    pad_cfg = cfg["structure_control"]["padding"]
    pad_rgb = np.array(hex_to_rgb(rng.choice(pad_cfg["colors"])), dtype=np.uint8)

    panel_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel_rgb[:, :] = pad_rgb
    if padding_enabled and rng.random() < float(pad_cfg.get("texture_prob", 0.0)):
        panel_rgb = add_noise(np_rng, panel_rgb, float(pad_cfg.get("texture_sigma", 3.0)))

    border_cfg = cfg["structure_control"]["border"]
    border_sides: List[str] = []
    border_thick = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    border_style = "none"

    border_rgb = np.array(hex_to_rgb(rng.choice(border_cfg["base_colors"])), dtype=np.uint8)
    if rng.random() < float(border_cfg.get("colored_prob", 0.0)):
        border_rgb = np.array((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), dtype=np.uint8)

    if border_enabled:
        border_sides, border_thick, border_style = sample_border_sides_and_thickness(rng, border_cfg)
        if border_style == "low_contrast":
            jitter = rng.randint(-12, 12)
            border_rgb = np.clip(pad_rgb.astype(np.int32) + jitter, 0, 255).astype(np.uint8)

    padding_thick = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    if padding_enabled:
        padding_thick = sample_padding_thicknesses(rng, pad_cfg)

    avail_x0 = int((border_thick["left"] if border_enabled else 0) + padding_thick["left"])
    avail_y0 = int((border_thick["top"] if border_enabled else 0) + padding_thick["top"])
    avail_x1 = int(panel_w - ((border_thick["right"] if border_enabled else 0) + padding_thick["right"]))
    avail_y1 = int(panel_h - ((border_thick["bottom"] if border_enabled else 0) + padding_thick["bottom"]))

    avail_x0 = clamp_int(avail_x0, 0, panel_w - 16)
    avail_y0 = clamp_int(avail_y0, 0, panel_h - 16)
    avail_x1 = clamp_int(avail_x1, avail_x0 + 16, panel_w)
    avail_y1 = clamp_int(avail_y1, avail_y0 + 16, panel_h)

    cx0, cy0, cx1, cy1, content_mode = sample_content_rect(
        rng, cfg, avail_x0, avail_y0, avail_x1, avail_y1, src_ar=src_ar
    )

    fit_cfg = cfg["content_fit"]
    crop_jitter = float(fit_cfg.get("crop_jitter", 0.2))
    patch = fit_cover_strict_ar(rng, main_src, cx1 - cx0, cy1 - cy0, crop_jitter=crop_jitter)

    if rng.random() < float(fit_cfg.get("color_jitter_prob", 0.0)):
        br = rng.uniform(*fit_cfg["brightness_range"])
        ct = rng.uniform(*fit_cfg["contrast_range"])
        gm = rng.uniform(*fit_cfg["gamma_range"])
        patch = apply_brightness_contrast(patch, brightness=br, contrast=ct)
        patch = apply_gamma(patch, gamma=gm)

    patch = ensure_hw(patch, cy1 - cy0, cx1 - cx0, interp=cv2.INTER_LINEAR)
    panel_rgb[cy0:cy1, cx0:cx1] = patch

    if border_enabled and border_sides:
        draw_border_visual(
            rng=rng,
            panel_rgb=panel_rgb,
            outer_shape=outer_shape,
            sides=border_sides,
            thick=border_thick,
            style=border_style,
            border_rgb=(int(border_rgb[0]), int(border_rgb[1]), int(border_rgb[2])),
            border_cfg=border_cfg,
        )

    # panel-level binary content mask
    panel_bin = np.full((panel_h, panel_w), non_val, dtype=np.uint8)
    inside = (outer_shape > 0)
    content_rect = np.zeros((panel_h, panel_w), dtype=np.uint8)
    content_rect[cy0:cy1, cx0:cx1] = 1
    content_pixels = inside & (content_rect > 0)
    panel_bin[content_pixels] = content_val

    # minimal rotation
    rot_cfg = cfg["panel"]["rotation"]
    angle = 0.0
    if rng.random() < float(rot_cfg["prob"]):
        angle = rng.uniform(*rot_cfg["deg_range"])
        panel_rgb = warp_rotate_rgb(panel_rgb, angle, border_value=(int(pad_rgb[0]), int(pad_rgb[1]), int(pad_rgb[2])))
        panel_bin = warp_rotate_mask(panel_bin, angle, border_value=non_val)
        outer_shape = warp_rotate_mask(outer_shape, angle, border_value=0)

    # paste to canvas (respect rounded shape)
    px0, py0 = x0, y0
    px1, py1 = x0 + panel_w, y0 + panel_h

    cx0c = max(0, px0)
    cy0c = max(0, py0)
    cx1c = min(W, px1)
    cy1c = min(H, py1)
    if cx1c <= cx0c or cy1c <= cy0c:
        meta = {"note": "panel_outside_canvas", "canvas_size": [W, H]}
        return SampleOutputs(canvas, bin_mask, meta)

    sx0 = cx0c - px0
    sy0 = cy0c - py0
    sx1 = sx0 + (cx1c - cx0c)
    sy1 = sy0 + (cy1c - cy0c)

    panel_crop_rgb = panel_rgb[sy0:sy1, sx0:sx1]
    panel_crop_bin = panel_bin[sy0:sy1, sx0:sx1]
    shape_crop = outer_shape[sy0:sy1, sx0:sx1] > 0

    # RGB
    region = canvas[cy0c:cy1c, cx0c:cx1c]
    paste_where(region, panel_crop_rgb, shape_crop)
    canvas[cy0c:cy1c, cx0c:cx1c] = region

    # Mask
    mreg = bin_mask[cy0c:cy1c, cx0c:cx1c]
    content_here = (panel_crop_bin == content_val) & shape_crop
    mreg[content_here] = content_val
    bin_mask[cy0c:cy1c, cx0c:cx1c] = mreg

    # Gutters (visual only, in margins)
    gutter_meta = add_gutters(
        rng=rng,
        cfg=cfg,
        canvas_rgb=canvas,
        canvas_mode=canvas_mode,
        panel_visible_bbox_xywh=(int(cx0c), int(cy0c), int(cx1c - cx0c), int(cy1c - cy0c)),
        bg_rgb=bg_rgb,
    )

    # degradations
    deg = cfg["degradations"]
    deg_list = []
    max_deg = int(deg["max_per_sample"])

    candidates = []
    if rng.random() < float(deg["jpeg_prob"]): candidates.append("jpeg")
    if rng.random() < float(deg["blur_prob"]): candidates.append("blur")
    if rng.random() < float(deg["noise_prob"]): candidates.append("noise")
    if rng.random() < float(deg["gamma_prob"]): candidates.append("gamma")
    if rng.random() < float(deg["vignette_prob"]): candidates.append("vignette")
    if rng.random() < float(deg["resize_artifact_prob"]): candidates.append("resize_artifact")

    rng.shuffle(candidates)
    candidates = candidates[:max_deg]

    for d in candidates:
        if d == "jpeg":
            q = rng.randint(deg["jpeg_quality_range"][0], deg["jpeg_quality_range"][1])
            canvas = jpeg_compress(canvas, q)
            deg_list.append({"name": "jpeg", "quality": int(q)})
        elif d == "blur":
            k = int(rng.choice(deg["blur_ksize_choices"]))
            if k % 2 == 0:
                k += 1
            canvas = cv2.GaussianBlur(canvas, (k, k), 0)
            deg_list.append({"name": "blur", "ksize": int(k)})
        elif d == "noise":
            s = float(rng.uniform(*deg["noise_sigma_range"]))
            canvas = add_noise(np_rng, canvas, s)
            deg_list.append({"name": "noise", "sigma": float(s)})
        elif d == "gamma":
            g = float(rng.uniform(*deg["gamma_range"]))
            canvas = apply_gamma(canvas, g)
            deg_list.append({"name": "gamma", "gamma": float(g)})
        elif d == "vignette":
            canvas = apply_vignette(rng, canvas)
            deg_list.append({"name": "vignette"})
        elif d == "resize_artifact":
            canvas = resize_artifact(rng, canvas, tuple(deg["resize_scale_range"]))
            deg_list.append({"name": "resize_artifact"})

    meta = {
        "canvas_size": [int(W), int(H)],
        "panel_size": [int(panel_w), int(panel_h)],
        "canvas_mode": canvas_mode,
        "margins_px": {"left": int(ml), "right": int(mr), "top": int(mt), "bottom": int(mb)},
        "tight_crop": {"enabled": bool(tight_crop_enabled), "cuts_px": cuts},
        "panel_origin_xy": [int(x0), int(y0)],
        "panel_visible_bbox_xywh": [int(cx0c), int(cy0c), int(cx1c - cx0c), int(cy1c - cy0c)],
        "rounded": bool(rounded),
        "corner_radius_px": int(radius),
        "rotation_deg": float(angle),
        "structure_flags": {"both_none": bool(both_none), "border_enabled": bool(border_enabled), "padding_enabled": bool(padding_enabled)},
        "border": {"style": border_style, "sides": border_sides, "thickness_px": border_thick},
        "padding": {"thickness_px": padding_thick},
        "content": {
            "source": str(main_src_path),
            "mode": content_mode,
            "rect_xyxy_in_panel": [int(cx0), int(cy0), int(cx1), int(cy1)],
            "fit": {"strategy": "cover_strict_ar", "crop_jitter": float(crop_jitter)},
        },
        "gutters": gutter_meta,
        "degradations": deg_list,
        "mask_values": {"content": content_val, "non_content": non_val},
        "note": "No neighbor fragments; non-content is only padding/border/canvas/gutters.",
    }

    return SampleOutputs(canvas, bin_mask, meta)


# ----------------------------
# IO
# ----------------------------
def write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(path))


def write_png_mask(path: Path, mask: np.ndarray) -> None:
    ok, enc = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Failed to encode mask PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(path))


def run_generation(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    in_dir = Path(cfg["dataset"]["input_images_dir"]).expanduser()
    out_dir = Path(cfg["dataset"]["output_dir"]).expanduser()

    n_samples = int(cfg["dataset"]["n_samples"])
    val_ratio = float(cfg["dataset"]["val_ratio"])
    seed = int(cfg["dataset"]["seed"])
    recursive = bool(cfg["dataset"]["recursive"])
    exts = list(cfg["dataset"]["exts"])
    part_size = int(cfg["dataset"]["part_size"])
    overwrite = bool(cfg["dataset"]["overwrite"])

    if not in_dir.exists():
        raise SystemExit(f"Input dir not found: {in_dir}")

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output dir exists. Set dataset.overwrite=true to replace: {out_dir}")

    ensure_dir(out_dir)

    real_images = list_images(in_dir, recursive=recursive, exts=exts)
    if not real_images:
        raise SystemExit(f"No images found under {in_dir} with exts={exts}")

    images_root = out_dir / "images"
    masks_root = out_dir / "masks"
    meta_root = out_dir / "meta"
    ensure_dir(images_root); ensure_dir(masks_root); ensure_dir(meta_root)

    train_manifest = out_dir / "manifest_train.jsonl"
    val_manifest = out_dir / "manifest_val.jsonl"

    rng_master = random.Random(seed)
    indices = list(range(n_samples))
    rng_master.shuffle(indices)
    n_val = int(round(n_samples * val_ratio))
    val_set = set(indices[:n_val])

    stats = {
        "n_samples": n_samples,
        "val_ratio": val_ratio,
        "n_val": n_val,
        "n_train": n_samples - n_val,
        "input_images_count": len(real_images),
        "mask_values": cfg["mask"],
        "note": "No neighbor fragments in this dataset version.",
    }

    with train_manifest.open("w", encoding="utf-8") as f_train, val_manifest.open("w", encoding="utf-8") as f_val:
        for i in tqdm(range(n_samples), desc="Generating"):
            rng = random.Random(seed + 1000003 * i)

            out_part = i // part_size
            part_name = f"part_{out_part:04d}"
            sample_id = f"{i:06d}"

            img_path = images_root / part_name / f"sample_{sample_id}.png"
            mask_path = masks_root / part_name / f"sample_{sample_id}_mask.png"
            meta_path = meta_root / part_name / f"sample_{sample_id}_meta.json"

            out = synthesize_one(cfg, rng, real_images)

            write_png_rgb(img_path, out.image_rgb)
            write_png_mask(mask_path, out.bin_mask)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(out.metadata, indent=2), encoding="utf-8")

            entry = {
                "id": f"sample_{sample_id}",
                "image": os.path.relpath(img_path, out_dir),
                "mask": os.path.relpath(mask_path, out_dir),
                "meta": os.path.relpath(meta_path, out_dir),
                "mask_values": cfg["mask"],
            }

            if i in val_set:
                f_val.write(json.dumps(entry) + "\n")
            else:
                f_train.write(json.dumps(entry) + "\n")

    (out_dir / "stats_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "config_used.yml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\nDone.\nOutput: {out_dir}\nTrain: {train_manifest}\nVal:   {val_manifest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate binary content masks (no neighbor fragments).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    run_generation(cfg_path)


if __name__ == "__main__":
    main()
