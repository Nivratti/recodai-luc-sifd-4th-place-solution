from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hashlib
import os
import tempfile

import numpy as np

from .base import BackendInfo, MatcherBackend
from ..types import MatchPrediction


@dataclass
class ImageMatchingModelsBackendConfig:
    """Config for https://github.com/gmberton/image-matching-models (IMM).

    IMM matchers often perform their own resizing (e.g. to make H/W divisible).
    This backend optionally applies a *pre-downscale* for speed on very large crops:

      - if max_side_sum > 0 and (H + W) > max_side_sum:
          resize isotropically (aspect preserved) so (H' + W') <= max_side_sum

    Keypoints returned by IMM are in the coordinates of the *inputs we pass to IMM*.
    When pre-downscaling is used, this backend maps keypoints back to the original crop
    coordinate system so downstream evaluation remains consistent.
    """
    imm_root: str = "modules/image-matching-models"  # folder that contains the `matching/` package
    matcher: str = "loftr"  # e.g. loftr, eloftr, matchformer, sift-lg, ...
    device: str = "auto"  # auto|cpu|cuda|mps

    # forwarded to matching.get_matcher(...)
    max_num_keypoints: int = 2048

    # match decision
    min_inliers: int = 20
    min_inlier_ratio: float = 0.0  # 0 disables
    score_mode: str = "num_inliers"  # num_inliers | inlier_ratio

    # optional rough masks from inlier keypoints
    mask_mode: str = "none"  # none | convex_hull
    mask_dilate: int = 0  # pixels, 0 disables

    # optional pre-downscale for speed
    max_side_sum: int = 0  # 0 disables; if >0, ensure (H+W) <= max_side_sum
    resize_cache_dir: Optional[str] = None  # where resized images are cached (optional)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _as_float_xy(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.size == 0:
        return None
    a = a.reshape(-1, 2).astype(np.float32, copy=False)
    return a


def _convex_hull_mask(pts_xy: np.ndarray, h: int, w: int, dilate: int = 0) -> np.ndarray:
    """Create a binary mask (0/255) from convex hull of points."""
    import cv2  # optional dependency used only if mask_mode=convex_hull

    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    k = int(dilate or 0)
    if k > 0:
        k = max(1, k)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


class ImageMatchingModelsBackend(MatcherBackend):
    """Backend wrapper for IMM matchers."""

    def __init__(self, cfg: Optional[ImageMatchingModelsBackendConfig] = None):
        self.cfg = cfg or ImageMatchingModelsBackendConfig()

        imm_root = Path(self.cfg.imm_root).resolve()
        if not imm_root.exists():
            raise FileNotFoundError(
                f"IMM root not found: {imm_root}. "
                f"Point --imm-root to the folder that contains the `matching/` package."
            )

        # Ensure we import IMM from the provided root.
        import sys
        sys.path.insert(0, str(imm_root))

        from matching import get_matcher, available_models  # type: ignore
        try:
            from matching import __version__ as imm_version  # type: ignore
        except Exception:
            imm_version = None

        # device selection
        device = str(self.cfg.device).lower()
        if device == "auto":
            try:
                from matching import get_default_device  # type: ignore
                device = str(get_default_device())
            except Exception:
                device = "cpu"

        self._device = device
        self._imm_version = imm_version
        self._unknown_matcher = (isinstance(self.cfg.matcher, str) and (self.cfg.matcher not in list(available_models)))

        self.matcher = get_matcher(
            self.cfg.matcher,
            device=self._device,
            max_num_keypoints=int(self.cfg.max_num_keypoints),
        )

        # optional pre-downscale cache
        self._resize_cache_dir: Optional[Path] = Path(self.cfg.resize_cache_dir).resolve() if self.cfg.resize_cache_dir else None
        if self._resize_cache_dir is not None:
            self._resize_cache_dir.mkdir(parents=True, exist_ok=True)
        self._resize_cache: Dict[str, Dict[str, Any]] = {}

    def info(self) -> BackendInfo:
        extras: Dict[str, Any] = {
            "matcher": self.cfg.matcher,
            "device": self._device,
            "max_num_keypoints": int(self.cfg.max_num_keypoints),
            "min_inliers": int(self.cfg.min_inliers),
            "min_inlier_ratio": float(self.cfg.min_inlier_ratio),
            "score_mode": str(self.cfg.score_mode),
            "mask_mode": str(self.cfg.mask_mode),
            "mask_dilate": int(self.cfg.mask_dilate),
            "max_side_sum": int(self.cfg.max_side_sum),
            "resize_cache_dir": str(self._resize_cache_dir) if self._resize_cache_dir else None,
        }
        if self._unknown_matcher:
            extras["warning"] = "matcher not in matching.available_models; assuming it is valid in your IMM checkout."
        return BackendInfo(
            name="image-matching-models",
            version=str(self._imm_version) if self._imm_version else None,
            notes="Runs selected IMM matcher; uses inlier count/ratio for match decision.",
            extras=extras,
        )

    def _score_and_decide(self, num_inliers: int, num_matches: int) -> Tuple[float, bool, float]:
        inlier_ratio = float(num_inliers) / float(max(1, num_matches))
        if str(self.cfg.score_mode).lower() == "inlier_ratio":
            score = inlier_ratio
        else:
            score = float(num_inliers)

        is_match = (num_inliers >= int(self.cfg.min_inliers))
        if float(self.cfg.min_inlier_ratio) > 0:
            is_match = is_match and (inlier_ratio >= float(self.cfg.min_inlier_ratio))

        return float(score), bool(is_match), float(inlier_ratio)

    def _maybe_resized_path(self, path: str) -> Tuple[str, float, float, int, int]:
        """Optionally pre-downscale an image and return (path_for_imm, sx, sy, orig_w, orig_h).

        When downscaling, aspect ratio is preserved. Keypoints returned by IMM (in resized coords)
        should be mapped back by dividing by (sx, sy).
        """
        p = str(Path(path).resolve())
        try:
            st = os.stat(p)
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        except Exception:
            mtime_ns = 0

        max_sum = int(self.cfg.max_side_sum or 0)
        cache_key = f"{p}|{mtime_ns}|{max_sum}"

        rec = self._resize_cache.get(cache_key)
        if rec is not None:
            return rec["path"], float(rec["sx"]), float(rec["sy"]), int(rec["ow"]), int(rec["oh"])

        from PIL import Image  # local import

        with Image.open(p) as im:
            ow, oh = im.size

        if max_sum <= 0 or (ow + oh) <= max_sum:
            rec = {"path": p, "sx": 1.0, "sy": 1.0, "ow": ow, "oh": oh}
            self._resize_cache[cache_key] = rec
            return p, 1.0, 1.0, ow, oh

        # Downscale isotropically so that (W+H) <= max_sum (never upscale).
        scale = min(1.0, max_sum / float(ow + oh))
        rw = max(1, int(round(ow * scale)))
        rh = max(1, int(round(oh * scale)))

        if rw == ow and rh == oh:
            rec = {"path": p, "sx": 1.0, "sy": 1.0, "ow": ow, "oh": oh}
            self._resize_cache[cache_key] = rec
            return p, 1.0, 1.0, ow, oh

        sx = rw / float(ow)
        sy = rh / float(oh)

        cache_dir = self._resize_cache_dir
        if cache_dir is None:
            cache_dir = Path(os.path.join(tempfile.gettempdir(), "recodai_sifd_imm_resize")).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

        out_path = str(cache_dir / f"{_sha1(cache_key)}.png")
        if not os.path.exists(out_path):
            from PIL import Image  # local import
            with Image.open(p) as im:
                im2 = im.resize((rw, rh), resample=Image.LANCZOS)
                im2.save(out_path)

        rec = {"path": out_path, "sx": sx, "sy": sy, "ow": ow, "oh": oh}
        self._resize_cache[cache_key] = rec
        return out_path, sx, sy, ow, oh

    def predict_pair(self, a_path: str, b_path: str, *, save_dir: Optional[str] = None) -> MatchPrediction:
        # IMM matchers accept file paths and return keypoints in the *input* image coordinates.
        # We optionally pre-downscale very large panels for speed, then map inlier keypoints back to original size.
        a_in, ax, ay, aw, ah = self._maybe_resized_path(a_path)
        b_in, bx, by, bw, bh = self._maybe_resized_path(b_path)

        out: Dict[str, Any] = self.matcher(a_in, b_in)

        num_inliers = int(out.get("num_inliers", 0) or 0)
        matched0 = _as_float_xy(out.get("matched_kpts0", None))
        num_matches = int(matched0.shape[0]) if matched0 is not None else 0

        score, is_match, inlier_ratio = self._score_and_decide(num_inliers, num_matches)

        inlier0 = _as_float_xy(out.get("inlier_kpts0", None))
        inlier1 = _as_float_xy(out.get("inlier_kpts1", None))

        # Map inliers back to original crop coords if pre-resize was used.
        if inlier0 is not None and (ax != 1.0 or ay != 1.0):
            inlier0 = np.stack([inlier0[:, 0] / ax, inlier0[:, 1] / ay], axis=1).astype(np.float32, copy=False)
        if inlier1 is not None and (bx != 1.0 or by != 1.0):
            inlier1 = np.stack([inlier1[:, 0] / bx, inlier1[:, 1] / by], axis=1).astype(np.float32, copy=False)

        mask_a: Optional[np.ndarray] = None
        mask_b: Optional[np.ndarray] = None
        if str(self.cfg.mask_mode).lower() == "convex_hull" and inlier0 is not None and inlier1 is not None:
            if inlier0.shape[0] >= 3 and inlier1.shape[0] >= 3:
                try:
                    mask_a = _convex_hull_mask(inlier0, h=ah, w=aw, dilate=int(self.cfg.mask_dilate))
                    mask_b = _convex_hull_mask(inlier1, h=bh, w=bw, dilate=int(self.cfg.mask_dilate))
                except Exception:
                    mask_a = None
                    mask_b = None

        extras: Dict[str, Any] = {
            "num_inliers": num_inliers,
            "num_matches": num_matches,
            "inlier_ratio": float(inlier_ratio),
            "pre_resize_a": {"sx": float(ax), "sy": float(ay), "used": str(a_in) != str(Path(a_path).resolve())},
            "pre_resize_b": {"sx": float(bx), "sy": float(by), "used": str(b_in) != str(Path(b_path).resolve())},
            "max_side_sum": int(self.cfg.max_side_sum or 0),
        }

        # Optional debug dump
        if save_dir:
            try:
                sd = Path(save_dir)
                sd.mkdir(parents=True, exist_ok=True)
                import json
                (sd / "imm_pred.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")
                if inlier0 is not None:
                    np.save(sd / "inlier_kpts0.npy", inlier0)
                if inlier1 is not None:
                    np.save(sd / "inlier_kpts1.npy", inlier1)
            except Exception:
                pass

        return MatchPrediction(
            is_match=bool(is_match),
            score=float(score),
            matched_keypoints=int(num_inliers),
            shared_area_a=0.0,
            shared_area_b=0.0,
            is_flipped=False,
            mask_a=mask_a,
            mask_b=mask_b,
            extras=extras,
        )
