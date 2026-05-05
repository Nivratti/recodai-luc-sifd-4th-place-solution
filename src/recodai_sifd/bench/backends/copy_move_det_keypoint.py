from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
import hashlib
import os
import pickle
import json
from pathlib import Path
from enum import Enum
from typing import Any, Dict, Optional

from .base import BackendInfo, MatcherBackend
from ..types import MatchPrediction


@dataclass
class CopyMoveDetKeypointBackendConfig:
    """Thin config wrapper.

    We forward all parameters to copy_move_det_keypoint.api.DetectorConfig where possible.
    Any attribute not present in your version will be ignored (best-effort).
    """
    descriptor_type: str = "cv_rsift"
    alignment_strategy: str = "CV_MAGSAC"
    matching_method: str = "BF"
    min_keypoints: int = 20
    min_area: float = 0.01
    check_flip: bool = True
    cross_kp_count: int = 1000
    keep_image: bool = False  # keep images inside FeatureSet (needed for viz in save_dir)
    assume_bgr: bool = True

    # prepared-feature caching for speed (prepare() is expensive when the same crop repeats)
    # - none: no caching
    # - mem: in-memory LRU cache (fastest)
    # - disk: on-disk pickle cache + in-memory hot cache (best for repeated runs)
    prep_cache: str = "mem"  # none | mem | disk
    prep_cache_dir: Optional[str] = None
    prep_cache_max: int = 10000  # max items in in-memory cache (0 = unlimited)

    # how to map DetectionResult into a single scalar score
    score_key: str = "shared_area_min"  # shared_area_min | shared_area_mean | matched_keypoints


class CopyMoveDetKeypointBackend(MatcherBackend):
    """Adapter for your copy-move-det-keypoint module.

    Uses:
      - prepare() to cache features per image
      - match_prepared() to compute is_match + masks + stats
    """

    def __init__(self, cfg: Optional[CopyMoveDetKeypointBackendConfig] = None):
        self.cfg = cfg or CopyMoveDetKeypointBackendConfig()
        # In-memory hot cache (LRU) for prepared FeatureSet objects
        self._mem_cache: "OrderedDict[str, object]" = OrderedDict()
        self._mem_cache_max: int = int(getattr(self.cfg, "prep_cache_max", 0) or 0)
        self._prep_cache_mode: str = str(getattr(self.cfg, "prep_cache", "mem") or "mem").lower()
        self._disk_cache_dir: Optional[Path] = Path(self.cfg.prep_cache_dir).resolve() if getattr(self.cfg, "prep_cache_dir", None) else None
        self._cfg_hash: str = ""

        # Import lazily so your wrapper repo can decide envs.
        try:
            from copy_move_det_keypoint.api import DetectorConfig, prepare, match_prepared
        except Exception as e:
            raise ImportError(
                "Failed to import copy_move_det_keypoint.api. "
                "Make sure the copy-move-det-keypoint repo is installed (pip install -e ...) "
                "or available on PYTHONPATH."
            ) from e

        self._DetectorConfig = DetectorConfig
        self._prepare = prepare
        self._match_prepared = match_prepared
        # build DetectorConfig best-effort (DetectorConfig is frozen in your API)
        base_cfg = DetectorConfig()

        def _coerce(default: object, val: object) -> object:
            # If the field is an Enum and user provided a string, try name/value conversions.
            if isinstance(default, Enum) and isinstance(val, str):
                enum_cls = type(default)
                # try member name
                try:
                    return enum_cls[val]
                except Exception:
                    pass
                # try member value
                try:
                    return enum_cls(val)
                except Exception:
                    pass
                # try normalized variants
                for cand in (val.upper(), val.lower()):
                    try:
                        return enum_cls[cand]
                    except Exception:
                        pass
                    try:
                        return enum_cls(cand)
                    except Exception:
                        pass
            return val

        updates = {}
        for k, v in vars(self.cfg).items():
            if hasattr(base_cfg, k):
                updates[k] = _coerce(getattr(base_cfg, k), v)

        self.det_cfg = replace(base_cfg, **updates) if updates else base_cfg

        # Build a small, stable signature of the detector config to separate caches across settings.
        try:
            cfg_dict: Dict[str, object] = {}
            for fname in getattr(self.det_cfg, "__dataclass_fields__", {}).keys():
                val = getattr(self.det_cfg, fname)
                cfg_dict[fname] = val.name if isinstance(val, Enum) else val
            cfg_dict["_keep_image"] = bool(self.cfg.keep_image)
            cfg_dict["_assume_bgr"] = bool(self.cfg.assume_bgr)
            self._cfg_hash = hashlib.sha1(
                json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
        except Exception:
            self._cfg_hash = "nohash"

        if self._prep_cache_mode == "disk" and self._disk_cache_dir is not None:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="copy-move-det-keypoint",
            version=None,
            notes="Keypoint matching + geometric verification; masks are convex hulls of inlier keypoints.",
        )

    def _get_features(self, path: str):
        p = Path(path).resolve()

        # If caching is disabled, always re-prepare.
        if self._prep_cache_mode == "none":
            return self._prepare(
                str(p),
                config=self.det_cfg,
                image_id=p.stem,
                keep_image=self.cfg.keep_image,
                assume_bgr=self.cfg.assume_bgr,
                extract_flip=self.det_cfg.check_flip,
                kp_count=self.det_cfg.cross_kp_count,
            )

        # Cache key includes: detector-config signature + absolute path + file stat
        try:
            st = p.stat()
            key_s = f"{self._cfg_hash}|{p}|{st.st_size}|{st.st_mtime_ns}"
        except Exception:
            key_s = f"{self._cfg_hash}|{p}"

        key_h = hashlib.sha1(key_s.encode("utf-8")).hexdigest()

        fs = self._mem_cache.get(key_h)
        if fs is not None:
            self._mem_cache.move_to_end(key_h)
            return fs

        # Disk cache lookup
        if self._prep_cache_mode == "disk" and self._disk_cache_dir is not None:
            fpath = self._disk_cache_dir / f"{key_h}.pkl"
            if fpath.exists():
                try:
                    with open(fpath, "rb") as f:
                        fs = pickle.load(f)
                    self._mem_cache[key_h] = fs
                    self._evict_if_needed()
                    return fs
                except Exception:
                    # Corrupt cache entry - remove and recompute
                    try:
                        fpath.unlink()
                    except Exception:
                        pass

        # Miss -> compute
        fs = self._prepare(
            str(p),
            config=self.det_cfg,
            image_id=p.stem,
            keep_image=self.cfg.keep_image,
            assume_bgr=self.cfg.assume_bgr,
            extract_flip=self.det_cfg.check_flip,
            kp_count=self.det_cfg.cross_kp_count,
        )

        # Store in memory
        self._mem_cache[key_h] = fs
        self._evict_if_needed()

        # Store on disk
        if self._prep_cache_mode == "disk" and self._disk_cache_dir is not None:
            fpath = self._disk_cache_dir / f"{key_h}.pkl"
            tmp = fpath.with_suffix(".tmp")
            try:
                with open(tmp, "wb") as f:
                    pickle.dump(fs, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, fpath)
            except Exception:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

        return fs


    def _evict_if_needed(self) -> None:
        # LRU eviction for in-memory cache
        if self._mem_cache_max and self._mem_cache_max > 0:
            while len(self._mem_cache) > self._mem_cache_max:
                self._mem_cache.popitem(last=False)

    def _score(self, *, shared_a: float, shared_b: float, matched_kpts: int) -> float:
        if self.cfg.score_key == "shared_area_mean":
            return float((shared_a + shared_b) / 2.0)
        if self.cfg.score_key == "matched_keypoints":
            return float(matched_kpts)
        # default: "shared_area_min"
        return float(min(shared_a, shared_b))

    def predict_pair(self, a_path: str, b_path: str, *, save_dir: Optional[str] = None) -> MatchPrediction:
        fs_a = self._get_features(a_path)
        fs_b = self._get_features(b_path)

        res = self._match_prepared(fs_a, fs_b, config=self.det_cfg, save_dir=save_dir)

        # DetectionResult fields as per your api.py:
        # is_match, mask_source, mask_target, shared_area_source/target, matched_keypoints, is_flipped
        shared_a = float(getattr(res, "shared_area_source", 0.0))
        shared_b = float(getattr(res, "shared_area_target", 0.0))
        matched_kpts = int(getattr(res, "matched_keypoints", 0))
        is_flipped = bool(getattr(res, "is_flipped", False))
        score = self._score(shared_a=shared_a, shared_b=shared_b, matched_kpts=matched_kpts)

        mask_a = getattr(res, "mask_source", None)
        mask_b = getattr(res, "mask_target", None)

        return MatchPrediction(
            is_match=bool(getattr(res, "is_match", False)),
            score=score,
            matched_keypoints=matched_kpts,
            shared_area_a=shared_a,
            shared_area_b=shared_b,
            is_flipped=is_flipped,
            mask_a=mask_a,
            mask_b=mask_b,
            extras={"outputs": getattr(res, "outputs", None)},
        )