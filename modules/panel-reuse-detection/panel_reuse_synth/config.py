from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml


KNOWN_TYPES = ["FULL_DUPLICATE", "FULL_OVERLAP_CROP", "PARTIAL_OVERLAP", "NO_MATCH"]


@dataclass
class InputCfg:
    recursive: bool
    exts: List[str]


@dataclass
class OutputCfg:
    out_dir: str
    image_format: str
    mask_format: str
    jpg_quality: int
    fixed_size: Optional[Tuple[int, int]]
    mask_generation_space: str  # "final" or "source"
    batch_size: int
    write_per_sample_meta: bool


@dataclass
class ContentFilterCfg:
    enabled: bool
    method: str
    min_score: float


@dataclass
class CropCfg:
    area_range: Tuple[float, float]
    aspect_ratio_range: Tuple[float, float]
    min_side_px: int
    max_tries: int
    content_filter: ContentFilterCfg


@dataclass
class SamplingCfg:
    strategy: str  # per_type_counts | probabilistic
    per_image_counts: Dict[str, int]
    pairs_per_image: int
    type_probs: Dict[str, float]
    use_full_image_prob: float
    crop: CropCfg
    enabled_types: List[str]


@dataclass
class NoMatchFallbackCfg:
    enabled: bool
    gap_px: int
    schemes: List[str]


@dataclass
class NoMatchCfg:
    require_disjoint: bool
    max_tries: int
    fallback: NoMatchFallbackCfg


@dataclass
class AugCfg:
    apply_prob: float
    max_ops: int
    allow_geometric: bool
    rotate90_prob: float
    flip_prob: float
    brightness_contrast_prob: float
    gamma_prob: float
    blur_prob: float
    noise_prob: float
    jpeg_prob: float
    jpeg_quality_range: Tuple[int, int]
    color_remap_prob: float


@dataclass
class FullDuplicateCfg:
    augment_prob: float
    augments: AugCfg


@dataclass
class FullOverlapBigCropCfg:
    use_full_image_prob: float
    area_range: Tuple[float, float]
    aspect_ratio_range: Tuple[float, float]


@dataclass
class FullOverlapSmallInBigCfg:
    area_ratio_of_big: Tuple[float, float]
    min_side_px: int
    gap_px: int


@dataclass
class FullOverlapCropCfg:
    max_tries: int
    direction_probs: Dict[str, float]  # A_IN_B/B_IN_A
    big_crop: FullOverlapBigCropCfg
    small_in_big: FullOverlapSmallInBigCfg
    augments: AugCfg


@dataclass
class PartialBaseCropCfg:
    use_full_image_prob: float
    area_range: Tuple[float, float]
    aspect_ratio_range: Tuple[float, float]


@dataclass
class PartialOverlapCfg:
    max_tries: int
    base_crop: PartialBaseCropCfg
    overlap_ratio_A: Tuple[float, float]
    overlap_ratio_B: Tuple[float, float]
    forbid_containment: bool
    patterns: Dict[str, float]
    augments: AugCfg


@dataclass
class Config:
    seed: int
    input: InputCfg
    output: OutputCfg
    sampling: SamplingCfg
    no_match: NoMatchCfg
    full_duplicate: FullDuplicateCfg
    full_overlap_crop: FullOverlapCropCfg
    partial_overlap: PartialOverlapCfg


def _tuple2f(x) -> Tuple[float, float]:
    return (float(x[0]), float(x[1]))


def _tuple2i(x) -> Tuple[int, int]:
    return (int(x[0]), int(x[1]))


def _default_aug(d: Dict[str, Any], apply_prob: float, allow_geo: bool) -> AugCfg:
    return AugCfg(
        apply_prob=float(d.get("apply_prob", apply_prob)),
        max_ops=int(d.get("max_ops", 3)),
        allow_geometric=bool(d.get("allow_geometric", allow_geo)),
        rotate90_prob=float(d.get("rotate90_prob", 0.25)),
        flip_prob=float(d.get("flip_prob", 0.25)),
        brightness_contrast_prob=float(d.get("brightness_contrast_prob", 0.35)),
        gamma_prob=float(d.get("gamma_prob", 0.20)),
        blur_prob=float(d.get("blur_prob", 0.15)),
        noise_prob=float(d.get("noise_prob", 0.15)),
        jpeg_prob=float(d.get("jpeg_prob", 0.25)),
        jpeg_quality_range=_tuple2i(d.get("jpeg_quality_range", [35, 95])),
        color_remap_prob=float(d.get("color_remap_prob", 0.05)),
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    seed = int(data.get("seed", 123))

    inp = data.get("input", {}) or {}
    out = data.get("output", {}) or {}
    sampling = data.get("sampling", {}) or {}
    crop = sampling.get("crop", {}) or {}
    cf = crop.get("content_filter", {}) or {}

    nm = data.get("no_match", {}) or {}
    fb = nm.get("fallback", {}) or {}

    fd = data.get("full_duplicate", {}) or {}
    fda = fd.get("augments", {}) or {}

    foc = data.get("full_overlap_crop", {}) or {}
    foc_big = foc.get("big_crop", {}) or {}
    foc_small = foc.get("small_in_big", {}) or {}
    foc_aug = foc.get("augments", {}) or {}

    po = data.get("partial_overlap", {}) or {}
    po_base = po.get("base_crop", {}) or {}
    po_aug = po.get("augments", {}) or {}

    fixed_size = out.get("fixed_size", None)
    if fixed_size is not None:
        fixed_size = _tuple2i(fixed_size)

    strategy = str(sampling.get("strategy", "per_type_counts")).strip()
    per_image_counts = {k: int(v) for k, v in (sampling.get("per_image_counts", {}) or {}).items()}
    pairs_per_image = int(sampling.get("pairs_per_image", 10))
    type_probs = {k: float(v) for k, v in (sampling.get("type_probs", {}) or {}).items()}

    # enabled types derived from YAML keys
    if strategy == "per_type_counts":
        keys = list(per_image_counts.keys())
    else:
        keys = list(type_probs.keys())
    enabled_types = [t for t in KNOWN_TYPES if t in keys]
    if not enabled_types:
        enabled_types = ["FULL_DUPLICATE", "NO_MATCH"]

    cfg = Config(
        seed=seed,
        input=InputCfg(
            recursive=bool(inp.get("recursive", True)),
            exts=[str(e).lower() for e in inp.get("exts", [".png", ".jpg", ".jpeg"])],
        ),
        output=OutputCfg(
            out_dir=str(out.get("out_dir", "./out/pairs_v1")),
            image_format=str(out.get("image_format", "png")).lower().lstrip("."),
            mask_format=str(out.get("mask_format", "png")).lower().lstrip("."),
            jpg_quality=int(out.get("jpg_quality", 95)),
            fixed_size=fixed_size,
            mask_generation_space=str(out.get("mask_generation_space", "final")).strip().lower(),
            batch_size=int(out.get("batch_size", 100)),
            write_per_sample_meta=bool(out.get("write_per_sample_meta", True)),
        ),
        sampling=SamplingCfg(
            strategy=strategy,
            per_image_counts=per_image_counts,
            pairs_per_image=pairs_per_image,
            type_probs=type_probs if type_probs else {t: 1.0 / len(enabled_types) for t in enabled_types},
            use_full_image_prob=float(sampling.get("use_full_image_prob", 0.30)),
            crop=CropCfg(
                area_range=_tuple2f(crop.get("area_range", [0.15, 0.95])),
                aspect_ratio_range=_tuple2f(crop.get("aspect_ratio_range", [0.6, 1.7])),
                min_side_px=int(crop.get("min_side_px", 96)),
                max_tries=int(crop.get("max_tries", 80)),
                content_filter=ContentFilterCfg(
                    enabled=bool(cf.get("enabled", False)),
                    method=str(cf.get("method", "edge_density")),
                    min_score=float(cf.get("min_score", 0.02)),
                ),
            ),
            enabled_types=enabled_types,
        ),
        no_match=NoMatchCfg(
            require_disjoint=bool(nm.get("require_disjoint", True)),
            max_tries=int(nm.get("max_tries", 200)),
            fallback=NoMatchFallbackCfg(
                enabled=bool(fb.get("enabled", True)),
                gap_px=int(fb.get("gap_px", 1)),
                schemes=[str(s) for s in (fb.get("schemes", ["HALVES_LR", "HALVES_TB", "QUADRANTS_2X2"]) or [])],
            ),
        ),
        full_duplicate=FullDuplicateCfg(
            augment_prob=float(fd.get("augment_prob", 0.85)),
            augments=_default_aug(fda, apply_prob=1.0, allow_geo=True),
        ),
        full_overlap_crop=FullOverlapCropCfg(
            max_tries=int(foc.get("max_tries", 250)),
            direction_probs={k: float(v) for k, v in (foc.get("direction_probs", {"A_IN_B": 0.5, "B_IN_A": 0.5}) or {}).items()},
            big_crop=FullOverlapBigCropCfg(
                use_full_image_prob=float(foc_big.get("use_full_image_prob", 0.20)),
                area_range=_tuple2f(foc_big.get("area_range", [0.40, 0.95])),
                aspect_ratio_range=_tuple2f(foc_big.get("aspect_ratio_range", [0.6, 1.7])),
            ),
            small_in_big=FullOverlapSmallInBigCfg(
                area_ratio_of_big=_tuple2f(foc_small.get("area_ratio_of_big", [0.25, 0.75])),
                min_side_px=int(foc_small.get("min_side_px", 96)),
                gap_px=int(foc_small.get("gap_px", 0)),
            ),
            augments=_default_aug(foc_aug, apply_prob=0.60, allow_geo=False),
        ),
        partial_overlap=PartialOverlapCfg(
            max_tries=int(po.get("max_tries", 350)),
            base_crop=PartialBaseCropCfg(
                use_full_image_prob=float(po_base.get("use_full_image_prob", 0.30)),
                area_range=_tuple2f(po_base.get("area_range", [0.50, 0.98])),
                aspect_ratio_range=_tuple2f(po_base.get("aspect_ratio_range", [0.6, 1.7])),
            ),
            overlap_ratio_A=_tuple2f(po.get("overlap_ratio_A", [0.10, 0.60])),
            overlap_ratio_B=_tuple2f(po.get("overlap_ratio_B", [0.10, 0.60])),
            forbid_containment=bool(po.get("forbid_containment", True)),
            patterns={k: float(v) for k, v in (po.get("patterns", {"SHIFTED_VIEW": 0.4, "OFFSET_CROPS": 0.35, "STRIP_OVERLAP": 0.15, "CORNER_OVERLAP": 0.10}) or {}).items()},
            augments=_default_aug(po_aug, apply_prob=0.60, allow_geo=False),
        ),
    )
    return cfg
