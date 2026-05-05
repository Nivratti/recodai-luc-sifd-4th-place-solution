"""copy_move_det_keypoint.api

Public Python API for keypoint-based copy-move / overlap detection.

Highlights
- `detect(...)`: pairwise single/cross-image detection
- Prepared/cached features for fast multi-matching:
  - `FeatureSet`, `prepare(...)`, `match_prepared(...)`
  - Optional pruning: `match_keypoints_only(...)` + `build_masks_from_matches(...)`

Input types
- str / pathlib.Path: image path
- np.ndarray: image array (assumed BGR by default, like cv2.imread)
- PIL.Image.Image: PIL image (RGB)

If you pass np.ndarray that is already RGB, set assume_bgr=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import os
import cv2
import numpy as np
from PIL import Image

from .feature_extraction import (
    extract_features,  # for strict parity on path-based detect()
    extract_features_from_image,
    DescriptorType,
)
from .matching import (
    match_keypoints,
    match_and_verify,
    verify_geometric_consistency,
    compute_shared_area,
    AlignmentStrategy,
    MatchingMethod,
)
from .clustering import cluster_keypoints
from .visualization import (
    create_mask_from_keypoints,
    save_mask,
    draw_matches_on_single_image,
    draw_matches_with_hulls,
    draw_clusters,
    draw_clusters_with_hulls,
)

ImageInput = Union[str, Path, np.ndarray, Image.Image]



@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the public detection API.

    Defaults mirror the existing CLI / KeypointCopyMoveDetector behavior.
    """

    descriptor_type: DescriptorType = DescriptorType.CV_RSIFT
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC
    matching_method: MatchingMethod = MatchingMethod.BF
    check_flip: bool = True
    min_keypoints: int = 20
    min_area: float = 0.01
    timeout: int = 600  # kept for parity; not enforced in the core algorithm

    # Feature extraction knobs (kept identical to existing detector defaults)
    cross_kp_count: int = 2000
    single_kp_count: int = 5000  # detector.detect_single_image override


@dataclass(frozen=True)
class DetectionResult:
    """Result object returned by the public API."""

    is_match: bool

    mask_source: np.ndarray
    mask_target: np.ndarray

    matched_kpts_source: np.ndarray
    matched_kpts_target: np.ndarray

    shared_area_source: float
    shared_area_target: float

    is_flipped: bool
    matched_keypoints: int

    # only present if save_dir is provided
    outputs: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class FeatureSet:
    """Prepared features for reuse across many matches."""

    image_id: str
    shape_hw: Tuple[int, int]  # (H, W)

    keypoints: np.ndarray
    descriptors: np.ndarray

    flip_keypoints: Optional[np.ndarray] = None
    flip_descriptors: Optional[np.ndarray] = None

    # optional: only needed if you want to save match visualizations
    image_bgr: Optional[np.ndarray] = None


@dataclass(frozen=True)
class MatchInfo:
    """Match output without mask creation (useful for pruning)."""

    is_match: bool
    matched_kpts_source: np.ndarray
    matched_kpts_target: np.ndarray
    shared_area_source: float
    shared_area_target: float
    is_flipped: bool
    matched_keypoints: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _empty_kpts() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _empty_mask(shape_hw: Tuple[int, int]) -> np.ndarray:
    return np.zeros(shape_hw, dtype=np.uint8)


def _is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def _id_from_input(x: ImageInput, fallback: str) -> str:
    if isinstance(x, Path):
        return x.stem
    if isinstance(x, str):
        return Path(x).stem
    return fallback


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(np.nanmax(arr)) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _as_rgb_for_features(x: ImageInput, *, assume_bgr: bool) -> np.ndarray:
    """Return RGB (or grayscale) uint8 array for feature extraction."""
    if _is_pathlike(x):
        pil = Image.open(str(x))
        return _to_uint8(np.array(pil))
    if isinstance(x, Image.Image):
        return _to_uint8(np.array(x))
    if isinstance(x, np.ndarray):
        arr = _to_uint8(x)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if assume_bgr:
                arr = arr[..., :3][..., ::-1]  # BGR -> RGB
            else:
                arr = arr[..., :3]
        return arr
    raise TypeError(f"Unsupported image input type: {type(x)}")


def _as_bgr_for_viz(x: ImageInput) -> Optional[np.ndarray]:
    """Return BGR uint8 array for visualization (best effort)."""
    if _is_pathlike(x):
        return cv2.imread(str(x))
    if isinstance(x, Image.Image):
        arr = _to_uint8(np.array(x))
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3][..., ::-1]  # RGB -> BGR
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return None
    if isinstance(x, np.ndarray):
        arr = _to_uint8(x)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3]  # assume already BGR
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return None
    return None


def prepare(
    image: ImageInput,
    *,
    config: Optional[DetectorConfig] = None,
    image_id: Optional[str] = None,
    keep_image: bool = False,
    assume_bgr: bool = True,
    extract_flip: Optional[bool] = None,
    kp_count: Optional[int] = None,
) -> FeatureSet:
    """Extract features once and return a reusable FeatureSet.

    Args:
        image: path / numpy array / PIL image
        config: DetectorConfig
        image_id: optional identifier (used in graph edges and filenames)
        keep_image: if True, store BGR image for optional visualization saving
        assume_bgr: only applies to numpy arrays; if True, treats np.ndarray as BGR
        extract_flip: override whether to compute flip features (default: config.check_flip)
        kp_count: override maximum keypoints to retain (default: config.cross_kp_count)

    Returns:
        FeatureSet
    """
    cfg = config or DetectorConfig()
    img_id = image_id or _id_from_input(image, fallback="image")

    do_flip = cfg.check_flip if extract_flip is None else bool(extract_flip)
    n_kp = cfg.cross_kp_count if kp_count is None else int(kp_count)

    rgb = _as_rgb_for_features(image, assume_bgr=assume_bgr)
    shape_hw = (int(rgb.shape[0]), int(rgb.shape[1]))

    kps, descs, fkps, fdescs = extract_features_from_image(
        rgb,
        descriptor_type=cfg.descriptor_type,
        extract_flip=do_flip,
        kp_count=n_kp,
        source_name=str(img_id),
    )

    img_bgr = _as_bgr_for_viz(image) if keep_image else None

    return FeatureSet(
        image_id=str(img_id),
        shape_hw=shape_hw,
        keypoints=kps.astype(np.float32, copy=False) if len(kps) else _empty_kpts(),
        descriptors=descs.astype(np.float32, copy=False) if len(descs) else np.empty((0, 128), dtype=np.float32),
        flip_keypoints=None if fkps is None else fkps.astype(np.float32, copy=False),
        flip_descriptors=None if fdescs is None else fdescs.astype(np.float32, copy=False),
        image_bgr=img_bgr,
    )


def match_keypoints_only(
    source: FeatureSet,
    target: FeatureSet,
    *,
    config: Optional[DetectorConfig] = None,
) -> MatchInfo:
    """Run matching + geometric verification without creating masks.

    Useful for pruning in large runs.
    """
    cfg = config or DetectorConfig()

    if len(source.keypoints) == 0 or len(target.keypoints) == 0:
        return MatchInfo(
            is_match=False,
            matched_kpts_source=_empty_kpts(),
            matched_kpts_target=_empty_kpts(),
            shared_area_source=0.0,
            shared_area_target=0.0,
            is_flipped=False,
            matched_keypoints=0,
        )

    mr = match_and_verify(
        source.keypoints,
        source.descriptors,
        target.keypoints,
        target.descriptors,
        flip_keypoints1=source.flip_keypoints,
        flip_descriptors1=source.flip_descriptors,
        image1_shape=source.shape_hw,
        image2_shape=target.shape_hw,
        alignment_strategy=cfg.alignment_strategy,
        matching_method=cfg.matching_method,
        min_keypoints=cfg.min_keypoints,
        min_area=cfg.min_area,
        check_flip=cfg.check_flip,
    )

    k1 = mr["matched_kpts1"]
    k2 = mr["matched_kpts2"]

    return MatchInfo(
        is_match=bool(mr["is_match"]),
        matched_kpts_source=k1.astype(np.float32, copy=False) if len(k1) else _empty_kpts(),
        matched_kpts_target=k2.astype(np.float32, copy=False) if len(k2) else _empty_kpts(),
        shared_area_source=float(mr["shared_area_img1"]),
        shared_area_target=float(mr["shared_area_img2"]),
        is_flipped=bool(mr["is_flipped_match"]),
        matched_keypoints=int(mr["matched_keypoints"]),
    )


def build_masks_from_matches(
    match: MatchInfo,
    source_shape_hw: Tuple[int, int],
    target_shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create convex-hull masks from MatchInfo."""
    if len(match.matched_kpts_source) >= 3:
        mask1 = create_mask_from_keypoints(match.matched_kpts_source, source_shape_hw)
    else:
        mask1 = _empty_mask(source_shape_hw)

    if len(match.matched_kpts_target) >= 3:
        mask2 = create_mask_from_keypoints(match.matched_kpts_target, target_shape_hw)
    else:
        mask2 = _empty_mask(target_shape_hw)

    return mask1, mask2


def match_prepared(
    source: FeatureSet,
    target: FeatureSet,
    *,
    config: Optional[DetectorConfig] = None,
    save_dir: Optional[str] = None,
) -> DetectionResult:
    """Match two prepared FeatureSets and return a DetectionResult.

    Notes:
        - Matching logic is unchanged (calls match_and_verify()).
        - If save_dir is provided:
            - Masks are always written.
            - Match/clusters visualizations are written only if both FeatureSets contain `image_bgr`.
    """
    cfg = config or DetectorConfig()
    mi = match_keypoints_only(source, target, config=cfg)

    mask1, mask2 = build_masks_from_matches(mi, source.shape_hw, target.shape_hw)

    outputs = None
    if save_dir is not None:
        _ensure_dir(save_dir)
        base_name = f"{source.image_id}_vs_{target.image_id}"

        mask1_path = os.path.join(save_dir, f"{base_name}_maskA.png")
        mask2_path = os.path.join(save_dir, f"{base_name}_maskB.png")
        combined_mask_path = os.path.join(save_dir, f"{base_name}_mask.png")

        save_mask(mask1, mask1_path)
        save_mask(mask2, mask2_path)

        combined_h = max(mask1.shape[0], mask2.shape[0])
        combined_mask = np.zeros((combined_h, mask1.shape[1] + mask2.shape[1]), dtype=np.uint8)
        combined_mask[: mask1.shape[0], : mask1.shape[1]] = mask1
        combined_mask[: mask2.shape[0], mask1.shape[1] :] = mask2
        save_mask(combined_mask, combined_mask_path)

        outputs = {
            "mask_path": combined_mask_path,
            "mask_source_path": mask1_path,
            "mask_target_path": mask2_path,
        }

        if source.image_bgr is not None and target.image_bgr is not None:
            matches_path = os.path.join(save_dir, f"{base_name}_matches.png")
            clusters_path = os.path.join(save_dir, f"{base_name}_clusters.png")

            draw_matches_with_hulls(
                source.image_bgr,
                mi.matched_kpts_source,
                target.image_bgr,
                mi.matched_kpts_target,
                matches_path,
            )

            clusters1 = (
                cluster_keypoints(mi.matched_kpts_source, source.shape_hw)
                if len(mi.matched_kpts_source) >= 3
                else []
            )
            clusters2 = (
                cluster_keypoints(mi.matched_kpts_target, target.shape_hw)
                if len(mi.matched_kpts_target) >= 3
                else []
            )
            match_indices = [
                (i, i)
                for i in range(min(len(mi.matched_kpts_source), len(mi.matched_kpts_target)))
            ]

            draw_clusters_with_hulls(
                source.image_bgr,
                mi.matched_kpts_source,
                target.image_bgr,
                mi.matched_kpts_target,
                clusters1,
                clusters2,
                clusters_path,
                match_indices=match_indices,
            )

            outputs.update({"matches_path": matches_path, "clusters_path": clusters_path})

    return DetectionResult(
        is_match=mi.is_match,
        mask_source=mask1,
        mask_target=mask2,
        matched_kpts_source=mi.matched_kpts_source,
        matched_kpts_target=mi.matched_kpts_target,
        shared_area_source=float(mi.shared_area_source),
        shared_area_target=float(mi.shared_area_target),
        is_flipped=bool(mi.is_flipped),
        matched_keypoints=int(mi.matched_keypoints),
        outputs=outputs,
    )


def detect(
    source: ImageInput,
    target: Optional[ImageInput] = None,
    *,
    config: Optional[DetectorConfig] = None,
    save_dir: Optional[str] = None,
    assume_bgr: bool = True,
) -> DetectionResult:
    """High-level detection entrypoint.

    Args:
        source: image path / array / PIL image. If target is None => single-image mode.
        target: optional second image
        config: DetectorConfig
        save_dir: if provided, outputs are written and returned in `outputs`
        assume_bgr: only applies when source/target are numpy arrays
    """
    cfg = config or DetectorConfig()

    if target is None:
        if _is_pathlike(source):
            return _detect_single_path(str(source), cfg, save_dir=save_dir)

        fs = prepare(
            source,
            config=cfg,
            image_id=_id_from_input(source, "source"),
            keep_image=(save_dir is not None),
            assume_bgr=assume_bgr,
            extract_flip=False,
            kp_count=cfg.single_kp_count,
        )
        return _detect_single_prepared(fs, cfg, save_dir=save_dir)

    if _is_pathlike(source) and _is_pathlike(target):
        return _detect_cross_path(str(source), str(target), cfg, save_dir=save_dir)

    fs = prepare(
        source,
        config=cfg,
        image_id=_id_from_input(source, "source"),
        keep_image=(save_dir is not None),
        assume_bgr=assume_bgr,
        extract_flip=cfg.check_flip,
        kp_count=cfg.cross_kp_count,
    )
    ft = prepare(
        target,
        config=cfg,
        image_id=_id_from_input(target, "target"),
        keep_image=(save_dir is not None),
        assume_bgr=assume_bgr,
        extract_flip=False,
        kp_count=cfg.cross_kp_count,
    )
    return match_prepared(fs, ft, config=cfg, save_dir=save_dir)


class CopyMoveDetector:
    """Convenience OO wrapper around `detect()` for repeated calls."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

    def detect(
        self,
        source: ImageInput,
        target: Optional[ImageInput] = None,
        *,
        save_dir: Optional[str] = None,
        assume_bgr: bool = True,
    ) -> DetectionResult:
        return detect(source, target, config=self.config, save_dir=save_dir, assume_bgr=assume_bgr)


# ------------------------------
# Internal implementations
# ------------------------------


def _detect_single_path(image_path: str, cfg: DetectorConfig, *, save_dir: Optional[str]) -> DetectionResult:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    img_shape = img.shape[:2]
    base_name = Path(image_path).stem

    kps, descs, _, _ = extract_features(
        image_path,
        descriptor_type=cfg.descriptor_type,
        extract_flip=False,
        kp_count=cfg.single_kp_count,
    )

    if len(kps) < cfg.min_keypoints:
        mask = _empty_mask(img_shape)
        outputs = None
        if save_dir is not None:
            _ensure_dir(save_dir)
            mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
            matches_path = os.path.join(save_dir, f"{base_name}_matches.png")
            clusters_path = os.path.join(save_dir, f"{base_name}_clusters.png")
            save_mask(mask, mask_path)
            cv2.imwrite(matches_path, img)
            cv2.imwrite(clusters_path, img)
            outputs = {"mask_path": mask_path, "matches_path": matches_path, "clusters_path": clusters_path}

        return DetectionResult(
            is_match=False,
            mask_source=mask,
            mask_target=mask.copy(),
            matched_kpts_source=_empty_kpts(),
            matched_kpts_target=_empty_kpts(),
            shared_area_source=0.0,
            shared_area_target=0.0,
            is_flipped=False,
            matched_keypoints=0,
            outputs=outputs,
        )

    indices1, indices2 = match_keypoints(
        kps,
        descs,
        kps,
        descs,
        matching_method=cfg.matching_method,
        ignore_self_matches=True,
    )
    initial_matches = [(i, j) for i, j in zip(indices1, indices2) if i != j]

    if len(initial_matches) < cfg.min_keypoints:
        mask = _empty_mask(img_shape)
        outputs = None
        if save_dir is not None:
            _ensure_dir(save_dir)
            mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
            matches_path = os.path.join(save_dir, f"{base_name}_matches.png")
            clusters_path = os.path.join(save_dir, f"{base_name}_clusters.png")
            save_mask(mask, mask_path)
            cv2.imwrite(matches_path, img)
            cv2.imwrite(clusters_path, img)
            outputs = {"mask_path": mask_path, "matches_path": matches_path, "clusters_path": clusters_path}

        return DetectionResult(
            is_match=False,
            mask_source=mask,
            mask_target=mask.copy(),
            matched_kpts_source=_empty_kpts(),
            matched_kpts_target=_empty_kpts(),
            shared_area_source=0.0,
            shared_area_target=0.0,
            is_flipped=False,
            matched_keypoints=0,
            outputs=outputs,
        )

    matched_kpts1 = kps[np.array([m[0] for m in initial_matches])]
    matched_kpts2 = kps[np.array([m[1] for m in initial_matches])]

    consistent_kpts1, consistent_kpts2 = verify_geometric_consistency(
        matched_kpts1,
        matched_kpts2,
        alignment_strategy=cfg.alignment_strategy,
        min_keypoints=4,
    )

    valid_matches = []
    for i, (kp1_orig, kp2_orig) in enumerate(zip(matched_kpts1, matched_kpts2)):
        for ck1 in consistent_kpts1:
            if np.allclose(kp1_orig, ck1, atol=0.5):
                valid_matches.append(initial_matches[i])
                break

    matched_indices = list(set([m[0] for m in valid_matches] + [m[1] for m in valid_matches]))
    clusters = []
    if len(matched_indices) > 0:
        subset_kps = kps[matched_indices]
        subset_map = {i: orig_idx for i, orig_idx in enumerate(matched_indices)}
        clusters_subset = cluster_keypoints(subset_kps, img_shape)
        clusters = [[subset_map[i] for i in cluster] for cluster in clusters_subset]

    all_matched_kps = np.vstack([consistent_kpts1, consistent_kpts2]) if len(consistent_kpts1) > 0 else np.array([])
    if len(all_matched_kps) > 0:
        mask = create_mask_from_keypoints(all_matched_kps, img_shape)
    else:
        mask = _empty_mask(img_shape)

    shared1 = compute_shared_area(img_shape, consistent_kpts1) if len(consistent_kpts1) >= 3 else 0.0
    shared2 = compute_shared_area(img_shape, consistent_kpts2) if len(consistent_kpts2) >= 3 else 0.0

    outputs = None
    if save_dir is not None:
        _ensure_dir(save_dir)
        mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        matches_path = os.path.join(save_dir, f"{base_name}_matches.png")
        clusters_path = os.path.join(save_dir, f"{base_name}_clusters.png")

        save_mask(mask, mask_path)
        draw_matches_on_single_image(img, kps, valid_matches, matches_path)
        draw_clusters(img, kps, clusters, clusters_path)
        outputs = {"mask_path": mask_path, "matches_path": matches_path, "clusters_path": clusters_path}

    return DetectionResult(
        is_match=(len(valid_matches) >= cfg.min_keypoints),
        mask_source=mask,
        mask_target=mask.copy(),
        matched_kpts_source=consistent_kpts1.astype(np.float32, copy=False) if len(consistent_kpts1) else _empty_kpts(),
        matched_kpts_target=consistent_kpts2.astype(np.float32, copy=False) if len(consistent_kpts2) else _empty_kpts(),
        shared_area_source=float(shared1),
        shared_area_target=float(shared2),
        is_flipped=False,
        matched_keypoints=int(len(valid_matches)),
        outputs=outputs,
    )


def _detect_single_prepared(fs: FeatureSet, cfg: DetectorConfig, *, save_dir: Optional[str]) -> DetectionResult:
    img_shape = fs.shape_hw
    base_name = fs.image_id

    kps = fs.keypoints
    descs = fs.descriptors

    if len(kps) < cfg.min_keypoints:
        mask = _empty_mask(img_shape)
        outputs = None
        if save_dir is not None:
            _ensure_dir(save_dir)
            mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
            save_mask(mask, mask_path)
            outputs = {"mask_path": mask_path}
        return DetectionResult(False, mask, mask.copy(), _empty_kpts(), _empty_kpts(), 0.0, 0.0, False, 0, outputs)

    indices1, indices2 = match_keypoints(
        kps,
        descs,
        kps,
        descs,
        matching_method=cfg.matching_method,
        ignore_self_matches=True,
    )
    initial_matches = [(i, j) for i, j in zip(indices1, indices2) if i != j]

    if len(initial_matches) < cfg.min_keypoints:
        mask = _empty_mask(img_shape)
        outputs = None
        if save_dir is not None:
            _ensure_dir(save_dir)
            mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
            save_mask(mask, mask_path)
            outputs = {"mask_path": mask_path}
        return DetectionResult(False, mask, mask.copy(), _empty_kpts(), _empty_kpts(), 0.0, 0.0, False, 0, outputs)

    matched_kpts1 = kps[np.array([m[0] for m in initial_matches])]
    matched_kpts2 = kps[np.array([m[1] for m in initial_matches])]

    consistent_kpts1, consistent_kpts2 = verify_geometric_consistency(
        matched_kpts1,
        matched_kpts2,
        alignment_strategy=cfg.alignment_strategy,
        min_keypoints=4,
    )

    valid_matches = []
    for i, (kp1_orig, kp2_orig) in enumerate(zip(matched_kpts1, matched_kpts2)):
        for ck1 in consistent_kpts1:
            if np.allclose(kp1_orig, ck1, atol=0.5):
                valid_matches.append(initial_matches[i])
                break

    all_matched_kps = np.vstack([consistent_kpts1, consistent_kpts2]) if len(consistent_kpts1) > 0 else np.array([])
    mask = create_mask_from_keypoints(all_matched_kps, img_shape) if len(all_matched_kps) > 0 else _empty_mask(img_shape)

    shared1 = compute_shared_area(img_shape, consistent_kpts1) if len(consistent_kpts1) >= 3 else 0.0
    shared2 = compute_shared_area(img_shape, consistent_kpts2) if len(consistent_kpts2) >= 3 else 0.0

    outputs = None
    if save_dir is not None:
        _ensure_dir(save_dir)
        mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        save_mask(mask, mask_path)
        outputs = {"mask_path": mask_path}

        if fs.image_bgr is not None:
            matches_path = os.path.join(save_dir, f"{base_name}_matches.png")
            clusters_path = os.path.join(save_dir, f"{base_name}_clusters.png")
            draw_matches_on_single_image(fs.image_bgr, kps, valid_matches, matches_path)

            matched_indices = list(set([m[0] for m in valid_matches] + [m[1] for m in valid_matches]))
            clusters = []
            if len(matched_indices) > 0:
                subset_kps = kps[matched_indices]
                subset_map = {i: orig_idx for i, orig_idx in enumerate(matched_indices)}
                clusters_subset = cluster_keypoints(subset_kps, img_shape)
                clusters = [[subset_map[i] for i in cluster] for cluster in clusters_subset]
            draw_clusters(fs.image_bgr, kps, clusters, clusters_path)
            outputs.update({"matches_path": matches_path, "clusters_path": clusters_path})

    return DetectionResult(
        is_match=(len(valid_matches) >= cfg.min_keypoints),
        mask_source=mask,
        mask_target=mask.copy(),
        matched_kpts_source=consistent_kpts1.astype(np.float32, copy=False) if len(consistent_kpts1) else _empty_kpts(),
        matched_kpts_target=consistent_kpts2.astype(np.float32, copy=False) if len(consistent_kpts2) else _empty_kpts(),
        shared_area_source=float(shared1),
        shared_area_target=float(shared2),
        is_flipped=False,
        matched_keypoints=int(len(valid_matches)),
        outputs=outputs,
    )


def _detect_cross_path(source_path: str, target_path: str, cfg: DetectorConfig, *, save_dir: Optional[str]) -> DetectionResult:
    img1 = cv2.imread(source_path)
    img2 = cv2.imread(target_path)
    if img1 is None:
        raise FileNotFoundError(f"Failed to load image: {source_path}")
    if img2 is None:
        raise FileNotFoundError(f"Failed to load image: {target_path}")

    img1_shape = img1.shape[:2]
    img2_shape = img2.shape[:2]

    base_name = f"{Path(source_path).stem}_vs_{Path(target_path).stem}"

    kps1, descs1, flip_kps1, flip_descs1 = extract_features(
        source_path,
        descriptor_type=cfg.descriptor_type,
        extract_flip=cfg.check_flip,
        kp_count=cfg.cross_kp_count,
    )
    kps2, descs2, _, _ = extract_features(
        target_path,
        descriptor_type=cfg.descriptor_type,
        extract_flip=False,
        kp_count=cfg.cross_kp_count,
    )

    if len(kps1) == 0 or len(kps2) == 0:
        mask1 = _empty_mask(img1_shape)
        mask2 = _empty_mask(img2_shape)
        outputs = None
        if save_dir is not None:
            _ensure_dir(save_dir)
            mask1_path = os.path.join(save_dir, f"{base_name}_maskA.png")
            mask2_path = os.path.join(save_dir, f"{base_name}_maskB.png")
            combined_mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
            save_mask(mask1, mask1_path)
            save_mask(mask2, mask2_path)
            combined_h = max(mask1.shape[0], mask2.shape[0])
            combined_mask = np.zeros((combined_h, mask1.shape[1] + mask2.shape[1]), dtype=np.uint8)
            combined_mask[: mask1.shape[0], : mask1.shape[1]] = mask1
            combined_mask[: mask2.shape[0], mask1.shape[1] :] = mask2
            save_mask(combined_mask, combined_mask_path)
            outputs = {"mask_path": combined_mask_path, "mask_source_path": mask1_path, "mask_target_path": mask2_path}

        return DetectionResult(False, mask1, mask2, _empty_kpts(), _empty_kpts(), 0.0, 0.0, False, 0, outputs)

    mr = match_and_verify(
        kps1,
        descs1,
        kps2,
        descs2,
        flip_keypoints1=flip_kps1,
        flip_descriptors1=flip_descs1,
        image1_shape=img1_shape,
        image2_shape=img2_shape,
        alignment_strategy=cfg.alignment_strategy,
        matching_method=cfg.matching_method,
        min_keypoints=cfg.min_keypoints,
        min_area=cfg.min_area,
        check_flip=cfg.check_flip,
    )

    matched_kpts1 = mr["matched_kpts1"]
    matched_kpts2 = mr["matched_kpts2"]
    is_flipped = bool(mr["is_flipped_match"])
    shared1 = float(mr["shared_area_img1"])
    shared2 = float(mr["shared_area_img2"])
    matched_count = int(mr["matched_keypoints"])
    is_match = bool(mr["is_match"])

    mask1 = create_mask_from_keypoints(matched_kpts1, img1_shape) if len(matched_kpts1) >= 3 else _empty_mask(img1_shape)
    mask2 = create_mask_from_keypoints(matched_kpts2, img2_shape) if len(matched_kpts2) >= 3 else _empty_mask(img2_shape)

    outputs = None
    if save_dir is not None:
        _ensure_dir(save_dir)

        mask1_path = os.path.join(save_dir, f"{base_name}_maskA.png")
        mask2_path = os.path.join(save_dir, f"{base_name}_maskB.png")
        combined_mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        matches_path = os.path.join(save_dir, f"{base_name}_matches.png")
        clusters_path = os.path.join(save_dir, f"{base_name}_clusters.png")

        save_mask(mask1, mask1_path)
        save_mask(mask2, mask2_path)

        combined_h = max(mask1.shape[0], mask2.shape[0])
        combined_mask = np.zeros((combined_h, mask1.shape[1] + mask2.shape[1]), dtype=np.uint8)
        combined_mask[: mask1.shape[0], : mask1.shape[1]] = mask1
        combined_mask[: mask2.shape[0], mask1.shape[1] :] = mask2
        save_mask(combined_mask, combined_mask_path)

        draw_matches_with_hulls(img1, matched_kpts1, img2, matched_kpts2, matches_path)

        clusters1 = cluster_keypoints(matched_kpts1, img1_shape) if len(matched_kpts1) >= 3 else []
        clusters2 = cluster_keypoints(matched_kpts2, img2_shape) if len(matched_kpts2) >= 3 else []
        match_indices = [(i, i) for i in range(min(len(matched_kpts1), len(matched_kpts2)))]

        draw_clusters_with_hulls(
            img1,
            matched_kpts1,
            img2,
            matched_kpts2,
            clusters1,
            clusters2,
            clusters_path,
            match_indices=match_indices,
        )

        outputs = {
            "mask_path": combined_mask_path,
            "mask_source_path": mask1_path,
            "mask_target_path": mask2_path,
            "matches_path": matches_path,
            "clusters_path": clusters_path,
        }

    return DetectionResult(
        is_match=is_match,
        mask_source=mask1,
        mask_target=mask2,
        matched_kpts_source=matched_kpts1.astype(np.float32, copy=False) if len(matched_kpts1) else _empty_kpts(),
        matched_kpts_target=matched_kpts2.astype(np.float32, copy=False) if len(matched_kpts2) else _empty_kpts(),
        shared_area_source=shared1,
        shared_area_target=shared2,
        is_flipped=is_flipped,
        matched_keypoints=matched_count,
        outputs=outputs,
    )
