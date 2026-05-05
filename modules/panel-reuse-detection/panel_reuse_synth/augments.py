from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import cv2


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def augment_panel(img: np.ndarray, rng: np.random.Generator, cfg) -> Tuple[np.ndarray, List[Dict]]:
    """
    Applies up to cfg.max_ops ops. Respects cfg.allow_geometric.
    Logs full decisions trace.

    cfg fields used:
      max_ops, allow_geometric
      rotate90_prob, flip_prob, brightness_contrast_prob, gamma_prob,
      blur_prob, noise_prob, jpeg_prob, jpeg_quality_range, color_remap_prob
    """
    out = img.copy()
    decisions: List[Dict] = []
    applied_ops = 0
    max_ops = max(0, int(cfg.max_ops))
    allow_geo = bool(getattr(cfg, "allow_geometric", True))

    def can_apply() -> bool:
        return applied_ops < max_ops

    def record(op: str, p: float, roll: float, applied: bool, params: Dict, reason: str = ""):
        decisions.append({
            "op": op, "p": float(p), "roll": float(roll),
            "applied": bool(applied), "params": params,
            "reason": reason,
        })

    # ROTATE90
    p = float(cfg.rotate90_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not allow_geo:
            reason = "geometric_disabled"
        elif not can_apply():
            reason = "max_ops_reached"
        else:
            k = int(rng.integers(1, 4))
            out = np.rot90(out, k).copy()
            params = {"k": k}
            applied = True
            applied_ops += 1
    record("ROTATE90", p, roll, applied, params, reason)

    # FLIP
    p = float(cfg.flip_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not allow_geo:
            reason = "geometric_disabled"
        elif not can_apply():
            reason = "max_ops_reached"
        else:
            if rng.random() < 0.5:
                out = out[:, ::-1].copy()
                params = {"mode": "H"}
            else:
                out = out[::-1, :].copy()
                params = {"mode": "V"}
            applied = True
            applied_ops += 1
    record("FLIP", p, roll, applied, params, reason)

    # BRIGHTNESS_CONTRAST
    p = float(cfg.brightness_contrast_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not can_apply():
            reason = "max_ops_reached"
        else:
            alpha = float(rng.uniform(0.80, 1.25))
            beta = float(rng.uniform(-20, 20))
            out = _clip_u8(out.astype(np.float32) * alpha + beta)
            params = {"alpha": alpha, "beta": beta}
            applied = True
            applied_ops += 1
    record("BRIGHTNESS_CONTRAST", p, roll, applied, params, reason)

    # GAMMA
    p = float(cfg.gamma_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not can_apply():
            reason = "max_ops_reached"
        else:
            gamma = float(rng.uniform(0.75, 1.35))
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
            out = _clip_u8(lut[out].astype(np.float32))
            params = {"gamma": gamma}
            applied = True
            applied_ops += 1
    record("GAMMA", p, roll, applied, params, reason)

    # BLUR
    p = float(cfg.blur_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not can_apply():
            reason = "max_ops_reached"
        else:
            k = int(rng.choice([3, 5]))
            out = cv2.GaussianBlur(out, (k, k), sigmaX=0)
            params = {"k": k}
            applied = True
            applied_ops += 1
    record("BLUR", p, roll, applied, params, reason)

    # NOISE
    p = float(cfg.noise_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not can_apply():
            reason = "max_ops_reached"
        else:
            sigma = float(rng.uniform(3.0, 12.0))
            n = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
            out = _clip_u8(out.astype(np.float32) + n)
            params = {"sigma": sigma}
            applied = True
            applied_ops += 1
    record("NOISE", p, roll, applied, params, reason)

    # JPEG
    p = float(cfg.jpeg_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not can_apply():
            reason = "max_ops_reached"
        else:
            qmin, qmax = int(cfg.jpeg_quality_range[0]), int(cfg.jpeg_quality_range[1])
            q = int(rng.integers(qmin, qmax + 1))
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                params = {"quality": q}
                applied = True
                applied_ops += 1
    record("JPEG", p, roll, applied, params, reason)

    # COLOR_REMAP
    p = float(cfg.color_remap_prob)
    roll = float(rng.random())
    applied = False
    params = {}
    reason = ""
    if roll < p:
        if not can_apply():
            reason = "max_ops_reached"
        else:
            gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            cmap = int(rng.choice([cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT]))
            colored = cv2.applyColorMap(gray, cmap)
            out = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            params = {"cmap": cmap}
            applied = True
            applied_ops += 1
    record("COLOR_REMAP", p, roll, applied, params, reason)

    return out, decisions
