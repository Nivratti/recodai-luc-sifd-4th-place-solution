from __future__ import annotations

import json
import math
import re
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from copy_move_det_keypoint import DetectorConfig, DetectionResult, FeatureSet, match_prepared, prepare


# -------------------------
# Performance / pruning configs
# -------------------------

@dataclass(frozen=True)
class ReuseSavePolicy:
    """Controls output I/O only (must not affect matching decisions)."""

    # "all"     -> legacy behavior: save match + no_match pair folders for every pair
    # "matches" -> save artifacts only for matched pairs
    # "none"    -> do not write per-pair artifacts (still writes reuse_summary.json when save_dir is set)
    artifacts: str = "all"

    # When artifacts="matches", optionally write minimal no_match pair.json files (still creates many folders).
    write_no_match_pair_json: bool = False


@dataclass(frozen=True)
class ReusePruningConfig:
    """Controls candidate pruning and early-stop.

    IMPORTANT:
    - When `enabled=False`, the caller should pass `prune=None` to ensure true legacy behavior.
      (We also treat enabled=False as legacy internally, as a safety net.)
    """

    enabled: bool = True

    # Individual feature toggles
    enable_cbir: bool = True
    enable_grouping: bool = True
    enable_geometry: bool = True
    enable_early_stop: bool = True
    enable_fast_resize: bool = False
    enable_fullres_rerun: bool = True  # only meaningful with fast_resize

    # Grouping
    group_mode: str = "broad"          # "none" | "class" | "broad"
    only_within_group: bool = True

    # CBIR shortlist
    cbir_topk: int = 12
    cbir_cfg: Dict[str, Any] = field(default_factory=lambda: dict(device="cuda", backend="timm", model_name="resnet50", batch_size=64, fp16=True, score_fp16=True))

    # Geometry filters (cheap rejects)
    aspect_ratio_log_tol: Optional[float] = 0.9
    area_ratio_min: Optional[float] = 0.20

    # Early stop (applies to ranked candidate list)
    stop_after_no_match_streak: int = 0
    stop_after_matches_per_source: Optional[int] = None

    # Fast resize stage (optional; may reduce recall)
    max_side_sum_fast: Optional[int] = None  # e.g., 1800
    rerun_fullres_on_match: bool = True      # only if enable_fullres_rerun


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class PanelNode:
    uid: str
    figure_id: str
    panel_id: Optional[int]
    det_index: int
    class_id: int
    class_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class RegionNode:
    uid: str
    panel_a: str
    panel_b: str
    bbox_a: Optional[Tuple[int, int, int, int]]
    bbox_b: Optional[Tuple[int, int, int, int]]
    shared_area_a: float
    shared_area_b: float
    is_flipped: bool
    matched_keypoints: int
    outputs: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class PairMatchEdge:
    panel_a: str
    panel_b: str
    region: str
    matched_keypoints: int
    is_flipped: bool
    shared_area_a: float
    shared_area_b: float
    bbox_a: Optional[Tuple[int, int, int, int]]
    bbox_b: Optional[Tuple[int, int, int, int]]
    mask_a: np.ndarray  # crop coords for panel_a
    mask_b: np.ndarray  # crop coords for panel_b
    outputs: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class Link:
    src: str
    dst: str
    kind: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionGraph:
    panels: Dict[str, PanelNode] = field(default_factory=dict)
    regions: Dict[str, RegionNode] = field(default_factory=dict)
    pairs: List[PairMatchEdge] = field(default_factory=list)  # matched only
    links: List[Link] = field(default_factory=list)


@dataclass
class ReuseDetectionResult:
    figure_id: str
    graph: RegionGraph
    matches_by_source: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    no_matches_by_source: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "graph": {
                "panels": {k: asdict(v) for k, v in self.graph.panels.items()},
                "regions": {k: asdict(v) for k, v in self.graph.regions.items()},
                "links": [asdict(x) for x in self.graph.links],
                "matched_pairs": [
                    {
                        "panel_a": p.panel_a,
                        "panel_b": p.panel_b,
                        "region": p.region,
                        "matched_keypoints": p.matched_keypoints,
                        "is_flipped": p.is_flipped,
                        "shared_area_a": p.shared_area_a,
                        "shared_area_b": p.shared_area_b,
                        "bbox_a": p.bbox_a,
                        "bbox_b": p.bbox_b,
                        "outputs": p.outputs,
                        "mask_a_shape": list(getattr(p.mask_a, "shape", ())),
                        "mask_b_shape": list(getattr(p.mask_b, "shape", ())),
                    }
                    for p in self.graph.pairs
                ],
            },
            "matches_by_source": self.matches_by_source,
            "no_matches_by_source": self.no_matches_by_source,
        }


# -------------------------
# Small helpers
# -------------------------

def make_panel_uid(figure_id: str, crop: Any) -> str:
    """
    Stable unique id for a crop. Works even if crop.panel_id is None.
    """
    pid = crop.panel_id if getattr(crop, "panel_id", None) is not None else int(getattr(crop, "det_index"))
    det_index = int(getattr(crop, "det_index"))
    class_id = int(getattr(crop, "class_id"))
    return f"{figure_id}::panel={pid}::det={det_index}::cls={class_id}"


def build_crops_by_uid(crops: Sequence[Any], *, figure_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in crops:
        out[make_panel_uid(figure_id, c)] = c
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_slug(s: str, *, max_len: int = 180) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9._=-]+", "_", s).strip("_")
    return (s2 or "item")[:max_len]


def _mask_bbox_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    m = mask
    if m.ndim != 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        return None
    ys, xs = np.where(m.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _relpath_outputs(outputs: Optional[Dict[str, str]], root: Optional[Path]) -> Optional[Dict[str, str]]:
    if outputs is None:
        return None
    if root is None:
        return dict(outputs)
    out: Dict[str, str] = {}
    for k, v in outputs.items():
        try:
            out[k] = str(Path(v).relative_to(root))
        except Exception:
            out[k] = v
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _suppress_sklearn_connectivity_warnings() -> None:
    """
    Suppress sklearn AgglomerativeClustering warning about multiple connected components.

    This warning is common/harmless for sparse connectivity; sklearn auto-completes
    it internally. We suppress it to keep logs clean.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"the number of connected components of the connectivity matrix is .* > 1\..*",
        category=UserWarning,
        module=r"sklearn\.cluster\._agglomerative",
    )


def _to_rgb_numpy(img: Any, *, assume_bgr: bool) -> Optional[np.ndarray]:
    """Best-effort conversion to RGB uint8 numpy."""
    if img is None:
        return None

    # PIL
    try:
        from PIL import Image  # type: ignore
        if isinstance(img, Image.Image):
            return np.asarray(img.convert("RGB"))
    except Exception:
        pass

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        arr = arr[:, :, :3]
        if assume_bgr:
            arr = arr[:, :, ::-1]
        if arr.dtype != np.uint8:
            # allow float/uint16 -> uint8 best-effort
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return arr

    return None


def _resize_to_max_side_sum(rgb: np.ndarray, max_side_sum: int) -> np.ndarray:
    """Resize RGB image so (W+H)<=max_side_sum (keeps aspect)."""
    if max_side_sum is None or int(max_side_sum) <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if (w + h) <= int(max_side_sum):
        return rgb
    scale = float(max_side_sum) / float(w + h)
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    try:
        import cv2  # type: ignore
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        # very rare: fallback nearest using numpy
        return rgb[:: max(1, int(round(h / new_h))), :: max(1, int(round(w / new_w))), :]


def _panel_group(class_name: str, *, mode: str) -> str:
    m = str(mode or "none").lower()
    if m == "none":
        return "all"
    if m == "class":
        return class_name or "unknown"
    # broad
    s = (class_name or "").lower()
    if "blot" in s:
        return "blot"
    if "micro" in s:
        return "microscopy"
    return "other"


def _aspect_ratio(w: int, h: int) -> float:
    h2 = max(1, int(h))
    return float(w) / float(h2)


def _geom_ok(
    w1: int,
    h1: int,
    w2: int,
    h2: int,
    *,
    aspect_log_tol: Optional[float],
    area_ratio_min: Optional[float],
) -> bool:
    if aspect_log_tol is not None:
        a1 = _aspect_ratio(w1, h1)
        a2 = _aspect_ratio(w2, h2)
        if a1 <= 0 or a2 <= 0:
            return False
        if abs(math.log(a1 / a2)) > float(aspect_log_tol):
            return False

    if area_ratio_min is not None:
        area1 = float(max(1, int(w1)) * max(1, int(h1)))
        area2 = float(max(1, int(w2)) * max(1, int(h2)))
        r = min(area1, area2) / max(area1, area2)
        if r < float(area_ratio_min):
            return False

    return True


# -------------------------
# Output folders (match/no_match top-level)
# -------------------------

def _pair_final_dir(root: Path, *, is_match: bool, src_uid: str, tgt_uid: str) -> Path:
    top = "match" if is_match else "no_match"
    return root / top / _safe_slug(src_uid) / _safe_slug(tgt_uid)


def _pair_staging_dir(root: Path, *, src_uid: str, tgt_uid: str) -> Path:
    return root / "_staging" / _safe_slug(src_uid) / _safe_slug(tgt_uid)


def _move_dir(src: Path, dst: Path) -> None:
    _ensure_dir(dst.parent)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), str(dst))


# -------------------------
# Core steps (modular)
# -------------------------

def prepare_feature_sets(
    crops: Sequence[Any],
    *,
    figure_id: str,
    config: DetectorConfig,
    keep_image: bool,
    assume_bgr: bool,
    extract_flip: Optional[bool],
    kp_count: Optional[int],
    graph: RegionGraph,
) -> Tuple[List[str], List[FeatureSet]]:
    """
    Create PanelNodes + FeatureSets for all crops.

    Returns
    -------
    (uids, feats)
    """
    uids: List[str] = []
    feats: List[FeatureSet] = []

    for idx, c in enumerate(crops):
        uid = make_panel_uid(figure_id, c)

        # Panel node
        if uid not in graph.panels:
            try:
                xyxy = tuple(map(int, getattr(c, "xyxy")))
            except Exception:
                logger.warning(f"Invalid crop.xyxy for uid={uid}; skipping.")
                continue

            graph.panels[uid] = PanelNode(
                uid=uid,
                figure_id=figure_id,
                panel_id=getattr(c, "panel_id", None),
                det_index=int(getattr(c, "det_index")),
                class_id=int(getattr(c, "class_id")),
                class_name=str(getattr(c, "class_name")),
                conf=float(getattr(c, "conf")),
                xyxy=xyxy,
            )

        # Feature set
        try:
            fs = prepare(
                getattr(c, "image"),
                config=config,
                image_id=uid,
                keep_image=keep_image,
                assume_bgr=assume_bgr,
                extract_flip=extract_flip,
                kp_count=kp_count,
            )
        except Exception as e:
            logger.warning(f"prepare() failed crop[{idx}] uid={uid} | {type(e).__name__}: {e}")
            continue

        uids.append(uid)
        feats.append(fs)

    return uids, feats


def match_pair_once(
    src_fs: FeatureSet,
    tgt_fs: FeatureSet,
    *,
    config: DetectorConfig,
    save_dir: Optional[Path],
) -> DetectionResult:
    """
    Run match_prepared with warning suppression.
    """
    with warnings.catch_warnings():
        _suppress_sklearn_connectivity_warnings()
        return match_prepared(
            src_fs,
            tgt_fs,
            config=config,
            save_dir=str(save_dir) if save_dir is not None else None,
        )


def _decide_match_flag(det: DetectionResult, *, min_matched_keypoints: Optional[int]) -> bool:
    is_match = bool(det.is_match)
    if is_match and min_matched_keypoints is not None:
        if int(det.matched_keypoints) < int(min_matched_keypoints):
            is_match = False
    return is_match


def record_pair_result_legacy(
    *,
    det: DetectionResult,
    figure_id: str,
    src_uid: str,
    tgt_uid: str,
    root: Optional[Path],
    staging_dir: Optional[Path],
    min_matched_keypoints: Optional[int],
) -> Tuple[bool, Dict[str, Any], Optional[Path], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    Legacy behavior:
    - Decide match flag
    - Move staging outputs into match/no_match
    - Always write pair.json when root provided
    """
    is_match = _decide_match_flag(det, min_matched_keypoints=min_matched_keypoints)

    bbox_a = _mask_bbox_xyxy(det.mask_source)
    bbox_b = _mask_bbox_xyxy(det.mask_target)

    final_dir: Optional[Path] = None
    outputs_rel: Optional[Dict[str, str]] = None

    if root is not None and staging_dir is not None:
        final_dir = _pair_final_dir(root, is_match=is_match, src_uid=src_uid, tgt_uid=tgt_uid)
        try:
            _move_dir(staging_dir, final_dir)
        except Exception as e:
            logger.warning(f"Failed to move outputs {staging_dir} -> {final_dir}: {e}")
            final_dir = staging_dir

        outputs_rel = _relpath_outputs(det.outputs, root)

        pair_summary = {
            "source": src_uid,
            "target": tgt_uid,
            "is_match": is_match,
            "matched_keypoints": int(det.matched_keypoints),
            "is_flipped": bool(det.is_flipped),
            "shared_area_source": float(det.shared_area_source),
            "shared_area_target": float(det.shared_area_target),
            "bbox_source": bbox_a,
            "bbox_target": bbox_b,
            "outputs": outputs_rel,
            "pair_dir": str(final_dir.relative_to(root)) if final_dir is not None else None,
        }

        _write_json(final_dir / "pair.json", pair_summary)
        return is_match, pair_summary, final_dir, bbox_a, bbox_b

    # no save_dir case
    pair_summary = {
        "source": src_uid,
        "target": tgt_uid,
        "is_match": is_match,
        "matched_keypoints": int(det.matched_keypoints),
        "is_flipped": bool(det.is_flipped),
        "shared_area_source": float(det.shared_area_source),
        "shared_area_target": float(det.shared_area_target),
        "bbox_source": bbox_a,
        "bbox_target": bbox_b,
        "outputs": det.outputs,
        "pair_dir": None,
    }
    return is_match, pair_summary, None, bbox_a, bbox_b


def add_match_to_graph(
    graph: RegionGraph,
    *,
    figure_id: str,
    src_uid: str,
    tgt_uid: str,
    det: DetectionResult,
    bbox_a: Optional[Tuple[int, int, int, int]],
    bbox_b: Optional[Tuple[int, int, int, int]],
    outputs_rel: Optional[Dict[str, str]],
) -> None:
    """
    Create RegionNode + PairMatchEdge (+ links) for a matched pair.
    """
    region_uid = f"{figure_id}::region::{src_uid}__{tgt_uid}"

    graph.regions[region_uid] = RegionNode(
        uid=region_uid,
        panel_a=src_uid,
        panel_b=tgt_uid,
        bbox_a=bbox_a,
        bbox_b=bbox_b,
        shared_area_a=float(det.shared_area_source),
        shared_area_b=float(det.shared_area_target),
        is_flipped=bool(det.is_flipped),
        matched_keypoints=int(det.matched_keypoints),
        outputs=outputs_rel,
    )

    graph.pairs.append(
        PairMatchEdge(
            panel_a=src_uid,
            panel_b=tgt_uid,
            region=region_uid,
            matched_keypoints=int(det.matched_keypoints),
            is_flipped=bool(det.is_flipped),
            shared_area_a=float(det.shared_area_source),
            shared_area_b=float(det.shared_area_target),
            bbox_a=bbox_a,
            bbox_b=bbox_b,
            mask_a=det.mask_source,
            mask_b=det.mask_target,
            outputs=outputs_rel,
        )
    )

    graph.links.append(
        Link(
            src=src_uid,
            dst=region_uid,
            kind="panel_to_region",
            attrs={"side": "source", "bbox": bbox_a, "shared_area": float(det.shared_area_source)},
        )
    )
    graph.links.append(
        Link(
            src=tgt_uid,
            dst=region_uid,
            kind="panel_to_region",
            attrs={"side": "target", "bbox": bbox_b, "shared_area": float(det.shared_area_target)},
        )
    )


# -------------------------
# Candidate building (optional)
# -------------------------

def _cbir_embed_all(
    items_rgb: Sequence[np.ndarray],
    *,
    ids: Sequence[str],
    cbir_cfg: Dict[str, Any],
    figure_id: str,
) -> Tuple[List[str], np.ndarray]:
    """Embed in-memory RGB arrays using panel_cbir, with compatibility fallback."""
    try:
        from panel_cbir.config import CBIRConfig  # type: ignore
        from panel_cbir.embedders import build_embedder  # type: ignore
        from panel_cbir import embedders as emb_mod  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"panel_cbir is not available but CBIR pruning was requested: {type(e).__name__}: {e}"
        )

    cfg = CBIRConfig(**(cbir_cfg or {}))
    embedder = build_embedder(cfg)

    # Correct API (if panel_cbir fixed)
    if hasattr(embedder, "embed_inputs"):
        good_ids, embs = embedder.embed_inputs(items_rgb, ids=list(ids), desc=f"CBIR embed ({figure_id})")
        return list(good_ids), embs

    # Compatibility fallback for panel_cbir bug (embed_inputs defined at module top-level)
    if hasattr(emb_mod, "embed_inputs"):
        fn = getattr(emb_mod, "embed_inputs")
        good_ids, embs = fn(embedder, items_rgb, ids=list(ids), desc=f"CBIR embed ({figure_id})")
        return list(good_ids), embs

    raise RuntimeError("panel_cbir embedder has no embed_inputs() and no module-level embed_inputs fallback")


def build_candidates_by_src(
    *,
    crops_by_uid: Dict[str, Any],
    uids: Sequence[str],
    graph: RegionGraph,
    assume_bgr: bool,
    prune: ReusePruningConfig,
    figure_id: str,
    debug: bool,
) -> Tuple[Dict[int, List[int]], Optional[np.ndarray]]:
    """
    Build per-source candidate target indices.

    Returns:
      candidates_by_src: dict src_i -> list of tgt indices (any order, but we keep ranked order when CBIR enabled)
      score_matrix: (N,N) cosine similarity if CBIR enabled else None
    """
    n = len(uids)
    if n < 2:
        return {}, None

    # Precompute group + sizes
    groups = ["all"] * n
    sizes = [(0, 0)] * n
    for i, uid in enumerate(uids):
        pn = graph.panels.get(uid)
        class_name = pn.class_name if pn is not None else ""
        groups[i] = _panel_group(class_name, mode=prune.group_mode) if prune.enable_grouping else "all"
        crop = crops_by_uid.get(uid)
        rgb = _to_rgb_numpy(getattr(crop, "image", None), assume_bgr=assume_bgr) if crop is not None else None
        if rgb is None:
            sizes[i] = (0, 0)
        else:
            h, w = rgb.shape[:2]
            sizes[i] = (int(w), int(h))

    # CBIR score matrix
    score = None
    if prune.enable_cbir and int(prune.cbir_topk or 0) > 0:
        items = []
        ok_uids = []
        for uid in uids:
            crop = crops_by_uid.get(uid)
            rgb = _to_rgb_numpy(getattr(crop, "image", None), assume_bgr=assume_bgr) if crop is not None else None
            if rgb is None:
                continue
            items.append(rgb)
            ok_uids.append(uid)

        if len(ok_uids) >= 2:
            good_ids, embs = _cbir_embed_all(items, ids=ok_uids, cbir_cfg=prune.cbir_cfg, figure_id=figure_id)
            # Map embeddings back to indices in uids
            uid_to_idx = {u: i for i, u in enumerate(uids)}
            idxs = [uid_to_idx[u] for u in good_ids if u in uid_to_idx]
            embs2 = embs[: len(idxs)]
            # Normalize for safety
            denom = np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-12
            embs2 = (embs2 / denom).astype(np.float32, copy=False)

            score = np.full((n, n), -1.0, dtype=np.float32)
            # compute full similarity only among embeddable items
            S = embs2 @ embs2.T
            for a, ia in enumerate(idxs):
                for b, ib in enumerate(idxs):
                    score[ia, ib] = float(S[a, b])

            if debug:
                logger.debug(f"[reuse/cbir] embedded={len(idxs)}/{n} dim={embs2.shape[1]} device={prune.cbir_cfg.get('device','?')}")
        else:
            if debug:
                logger.debug("[reuse/cbir] not enough embeddable items; skipping CBIR pruning")

    # Build candidates
    candidates: Dict[int, List[int]] = {}

    def allow_pair(i: int, j: int) -> bool:
        # i<j assumed
        if prune.enable_grouping and prune.only_within_group:
            if groups[i] != groups[j]:
                return False

        if prune.enable_geometry:
            w1, h1 = sizes[i]
            w2, h2 = sizes[j]
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                # if we can't compute sizes, don't prune by geometry
                return True
            if not _geom_ok(
                w1, h1, w2, h2,
                aspect_log_tol=prune.aspect_ratio_log_tol,
                area_ratio_min=prune.area_ratio_min,
            ):
                return False
        return True

    # If CBIR score is available: ranked shortlist
    if score is not None:
        topk = int(prune.cbir_topk or 0)
        for i in range(n - 1):
            # take more then filter (to keep enough after grouping/geometry)
            k0 = max(topk * 3, topk + 5)
            row = score[i].copy()
            row[i] = -1.0
            # sort by score desc
            order = np.argsort(-row)
            picked: List[int] = []
            for j in order:
                j = int(j)
                if j <= i:
                    continue
                if row[j] <= -0.5:
                    # unknown/unembedded pairs: skip in pruning mode
                    continue
                if not allow_pair(i, j):
                    continue
                picked.append(j)
                if len(picked) >= topk:
                    break

            # If CBIR couldn't pick enough and you want safety, you can fall back.
            # We choose NOT to auto-fallback here (user controls cbir_topk / toggles).
            candidates[i] = picked

        # Symmetric completion: ensure if j ranks i but i didn't rank j, still consider pair.
        # We do this by adding to min(i,j)'s list and then re-sorting by score.
        for j in range(1, n):
            row = score[j].copy()
            row[j] = -1.0
            order = np.argsort(-row)
            picked_j = []
            for i2 in order[: max(topk * 2, topk + 5)]:
                i2 = int(i2)
                if i2 == j:
                    continue
                if row[i2] <= -0.5:
                    continue
                a, b = (i2, j) if i2 < j else (j, i2)
                if a == b:
                    continue
                if not allow_pair(a, b):
                    continue
                picked_j.append((a, b))
                if len(picked_j) >= topk:
                    break
            for a, b in picked_j:
                if a >= n - 1:
                    continue
                lst = candidates.setdefault(a, [])
                if b not in lst and b > a:
                    lst.append(b)
                    # keep sorted by score desc
                    lst.sort(key=lambda t: float(score[a, t]), reverse=True)

        # Debug summary
        if debug:
            lens = [len(candidates.get(i, [])) for i in range(n - 1)]
            if lens:
                logger.debug(f"[reuse/prune] cbir_topk={topk} candidates_per_src min={min(lens)} avg={sum(lens)/len(lens):.2f} max={max(lens)}")
        return candidates, score

    # No CBIR: if other pruning enabled, we can still filter full pairs cheaply
    for i in range(n - 1):
        lst = []
        for j in range(i + 1, n):
            if allow_pair(i, j):
                lst.append(j)
        candidates[i] = lst

    if debug and (prune.enable_grouping or prune.enable_geometry):
        lens = [len(candidates.get(i, [])) for i in range(n - 1)]
        if lens:
            logger.debug(f"[reuse/prune] candidates_per_src min={min(lens)} avg={sum(lens)/len(lens):.2f} max={max(lens)} (no CBIR)")
    return candidates, None


# -------------------------
# Matching loops (legacy + pruned)
# -------------------------

def match_one_to_many(
    *,
    src_i: int,
    uids: Sequence[str],
    feats: Sequence[FeatureSet],
    graph: RegionGraph,
    figure_id: str,
    config: DetectorConfig,
    root: Optional[Path],
    only_same_class: bool,
    min_matched_keypoints: Optional[int],
    matches_by_source: Dict[str, List[Dict[str, Any]]],
    no_matches_by_source: Dict[str, List[Dict[str, Any]]],
    save_policy: ReuseSavePolicy,
    # pruning controls (optional)
    candidate_js: Optional[Sequence[int]] = None,
    score_row: Optional[np.ndarray] = None,
    prune: Optional[ReusePruningConfig] = None,
    debug: bool = False,
    debug_pairs: bool = False,
    # fast resize support (optional)
    crops_by_uid: Optional[Dict[str, Any]] = None,
    assume_bgr: bool = True,
) -> None:
    """
    For one source crop, match against targets.

    Updates:
    - matches_by_source / no_matches_by_source
    - graph (for matches)
    - saves outputs depending on save_policy (if root is provided)
    """
    src_uid = uids[src_i]
    js: Sequence[int]
    if candidate_js is None:
        js = range(src_i + 1, len(uids))
    else:
        js = [int(j) for j in candidate_js if int(j) > src_i and int(j) < len(uids)]

    # early stop counters
    no_match_streak = 0
    match_count = 0

    # Fast resize: optionally prepare "fast" features on the fly.
    # We only use it when explicitly enabled.
    use_fast = bool(prune and prune.enabled and prune.enable_fast_resize and (prune.max_side_sum_fast or 0) > 0)
    rerun_full = bool(use_fast and prune and prune.rerun_fullres_on_match and prune.enable_fullres_rerun)

    # Cache per-panel fast features (simple dict) to avoid repeated prepare if many matches.
    fast_cache: Dict[str, FeatureSet] = {}

    def get_fast_fs(uid: str, full_fs: FeatureSet) -> FeatureSet:
        if uid in fast_cache:
            return fast_cache[uid]
        if crops_by_uid is None:
            fast_cache[uid] = full_fs
            return full_fs
        crop = crops_by_uid.get(uid)
        rgb = _to_rgb_numpy(getattr(crop, "image", None), assume_bgr=assume_bgr) if crop is not None else None
        if rgb is None:
            fast_cache[uid] = full_fs
            return full_fs
        rgb2 = _resize_to_max_side_sum(rgb, int(prune.max_side_sum_fast))
        # prepare expects same channel order as input image; our rgb2 is RGB.
        img_in = rgb2[:, :, ::-1] if assume_bgr else rgb2
        try:
            fs = prepare(
                img_in,
                config=config,
                image_id=uid,
                keep_image=False,
                assume_bgr=assume_bgr,
                extract_flip=None,
                kp_count=None,
            )
            fast_cache[uid] = fs
            return fs
        except Exception:
            fast_cache[uid] = full_fs
            return full_fs

    for j in js:
        tgt_uid = uids[j]

        if only_same_class:
            if graph.panels[src_uid].class_id != graph.panels[tgt_uid].class_id:
                continue

        # Optional per-pair debug line
        if debug_pairs:
            s = None
            if score_row is not None:
                try:
                    s = float(score_row[j])
                except Exception:
                    s = None
            logger.debug(f"[reuse/pair] {figure_id} src={src_uid} tgt={tgt_uid} cbir_score={s}")

        # Decide matching strategy (legacy / matches-only / none)
        artifacts_mode = str(save_policy.artifacts or "all").lower()

        # --- compute det (may be fast stage) ---
        if use_fast:
            det0 = match_pair_once(
                get_fast_fs(src_uid, feats[src_i]),
                get_fast_fs(tgt_uid, feats[j]),
                config=config,
                save_dir=None,
            )
            is_match0 = _decide_match_flag(det0, min_matched_keypoints=min_matched_keypoints)

            if rerun_full and is_match0:
                det = match_pair_once(feats[src_i], feats[j], config=config, save_dir=None)
            else:
                det = det0
        else:
            det = match_pair_once(feats[src_i], feats[j], config=config, save_dir=None)

        is_match = _decide_match_flag(det, min_matched_keypoints=min_matched_keypoints)

        bbox_a = _mask_bbox_xyxy(det.mask_source)
        bbox_b = _mask_bbox_xyxy(det.mask_target)

        outputs_rel: Optional[Dict[str, str]] = None
        pair_dir_rel: Optional[str] = None

        # --- write artifacts depending on policy ---
        if root is not None and artifacts_mode != "none":
            if artifacts_mode == "all":
                # Legacy: run again but with save_dir staging and move. This keeps exact old folder layout.
                staging_dir = _pair_staging_dir(root, src_uid=src_uid, tgt_uid=tgt_uid)
                _ensure_dir(staging_dir)
                det_save = match_pair_once(feats[src_i], feats[j], config=config, save_dir=staging_dir)
                # IMPORTANT: decision and graph MUST be based on `det` (no save I/O should affect match logic)
                # We still write outputs from det_save.
                is_match_save = _decide_match_flag(det, min_matched_keypoints=min_matched_keypoints)
                # Move and write pair.json
                final_dir = _pair_final_dir(root, is_match=is_match_save, src_uid=src_uid, tgt_uid=tgt_uid)
                try:
                    _move_dir(staging_dir, final_dir)
                except Exception as e:
                    logger.warning(f"Failed to move outputs {staging_dir} -> {final_dir}: {e}")
                    final_dir = staging_dir
                outputs_rel = _relpath_outputs(det_save.outputs, root)
                pair_dir_rel = str(final_dir.relative_to(root))

                pair_summary = {
                    "source": src_uid,
                    "target": tgt_uid,
                    "is_match": is_match,
                    "matched_keypoints": int(det.matched_keypoints),
                    "is_flipped": bool(det.is_flipped),
                    "shared_area_source": float(det.shared_area_source),
                    "shared_area_target": float(det.shared_area_target),
                    "bbox_source": bbox_a,
                    "bbox_target": bbox_b,
                    "outputs": outputs_rel,
                    "pair_dir": pair_dir_rel,
                }
                _write_json(final_dir / "pair.json", pair_summary)

            elif artifacts_mode == "matches":
                if is_match:
                    staging_dir = _pair_staging_dir(root, src_uid=src_uid, tgt_uid=tgt_uid)
                    _ensure_dir(staging_dir)
                    det_save = match_pair_once(feats[src_i], feats[j], config=config, save_dir=staging_dir)
                    final_dir = _pair_final_dir(root, is_match=True, src_uid=src_uid, tgt_uid=tgt_uid)
                    try:
                        _move_dir(staging_dir, final_dir)
                    except Exception as e:
                        logger.warning(f"Failed to move outputs {staging_dir} -> {final_dir}: {e}")
                        final_dir = staging_dir
                    outputs_rel = _relpath_outputs(det_save.outputs, root)
                    pair_dir_rel = str(final_dir.relative_to(root))

                    pair_summary = {
                        "source": src_uid,
                        "target": tgt_uid,
                        "is_match": True,
                        "matched_keypoints": int(det.matched_keypoints),
                        "is_flipped": bool(det.is_flipped),
                        "shared_area_source": float(det.shared_area_source),
                        "shared_area_target": float(det.shared_area_target),
                        "bbox_source": bbox_a,
                        "bbox_target": bbox_b,
                        "outputs": outputs_rel,
                        "pair_dir": pair_dir_rel,
                    }
                    _write_json(final_dir / "pair.json", pair_summary)
                else:
                    # optionally write minimal no_match json
                    if save_policy.write_no_match_pair_json:
                        final_dir = _pair_final_dir(root, is_match=False, src_uid=src_uid, tgt_uid=tgt_uid)
                        _ensure_dir(final_dir)
                        pair_summary = {
                            "source": src_uid,
                            "target": tgt_uid,
                            "is_match": False,
                            "matched_keypoints": int(det.matched_keypoints),
                            "is_flipped": bool(det.is_flipped),
                            "shared_area_source": float(det.shared_area_source),
                            "shared_area_target": float(det.shared_area_target),
                            "bbox_source": bbox_a,
                            "bbox_target": bbox_b,
                            "outputs": None,
                            "pair_dir": str(final_dir.relative_to(root)),
                        }
                        _write_json(final_dir / "pair.json", pair_summary)

        # --- store summaries ---
        pair_summary_mem = {
            "source": src_uid,
            "target": tgt_uid,
            "is_match": is_match,
            "matched_keypoints": int(det.matched_keypoints),
            "is_flipped": bool(det.is_flipped),
            "shared_area_source": float(det.shared_area_source),
            "shared_area_target": float(det.shared_area_target),
            "bbox_source": bbox_a,
            "bbox_target": bbox_b,
            "outputs": outputs_rel,
            "pair_dir": pair_dir_rel,
        }

        if is_match:
            matches_by_source.setdefault(src_uid, []).append(pair_summary_mem)
        else:
            no_matches_by_source.setdefault(src_uid, []).append(pair_summary_mem)

        # add to graph only if match
        if is_match:
            add_match_to_graph(
                graph,
                figure_id=figure_id,
                src_uid=src_uid,
                tgt_uid=tgt_uid,
                det=det,
                bbox_a=bbox_a,
                bbox_b=bbox_b,
                outputs_rel=outputs_rel,
            )

        # early stop updates
        if prune and prune.enabled and prune.enable_early_stop and int(prune.stop_after_no_match_streak or 0) > 0:
            # Only apply early stop when the loop is over ranked candidates (candidate_js provided).
            if candidate_js is not None:
                if is_match:
                    no_match_streak = 0
                    match_count += 1
                else:
                    no_match_streak += 1

                if prune.stop_after_matches_per_source is not None and int(prune.stop_after_matches_per_source) > 0:
                    if match_count >= int(prune.stop_after_matches_per_source):
                        if debug:
                            logger.debug(f"[reuse/early-stop] src={src_uid} reached matches_per_source={match_count}")
                        break

                if no_match_streak >= int(prune.stop_after_no_match_streak):
                    if debug:
                        logger.debug(f"[reuse/early-stop] src={src_uid} no_match_streak={no_match_streak}")
                    break


# -------------------------
# Public API
# -------------------------

def run_reuse_detection_all_pairs(
    crops: Sequence[Any],
    *,
    figure_id: str,
    config: Optional[DetectorConfig] = None,
    assume_bgr: bool = True,
    keep_image: bool = False,
    extract_flip: Optional[bool] = None,
    kp_count: Optional[int] = None,
    save_dir: Optional[str | Path] = None,
    only_same_class: bool = False,
    min_matched_keypoints: Optional[int] = None,
    # NEW: speed / pruning
    save_policy: Optional[ReuseSavePolicy] = None,
    prune: Optional[ReusePruningConfig] = None,
    debug: bool = False,
    debug_pairs: bool = False,
) -> ReuseDetectionResult:
    """
    Reuse detection over crops using one-to-many looping (each pair matched once).

    Legacy (no pruning)
    -------------------
    If `prune is None` or `prune.enabled is False`, the matcher enumerates **all unique pairs** (i<j)
    and behaves like the original version (match/no_match outputs when save_policy.artifacts="all").

    Save layout (if save_dir provided)
    ---------------------------------
    save_dir/
      match/<src>/<tgt>/*
      no_match/<src>/<tgt>/*          (only when artifacts="all" OR write_no_match_pair_json=True)
      _staging/<src>/<tgt>/*
      reuse_summary.json
    """
    cfg = config if config is not None else DetectorConfig()

    # Default policy: preserve legacy output behavior
    pol = save_policy if save_policy is not None else ReuseSavePolicy(artifacts="all", write_no_match_pair_json=True)

    root: Optional[Path] = Path(save_dir) if save_dir is not None else None
    if root is not None:
        _ensure_dir(root)
        if str(pol.artifacts).lower() != "none":
            _ensure_dir(root / "match")
            _ensure_dir(root / "_staging")
            if str(pol.artifacts).lower() == "all" or bool(pol.write_no_match_pair_json):
                _ensure_dir(root / "no_match")

    graph = RegionGraph()
    result = ReuseDetectionResult(figure_id=figure_id, graph=graph)

    # 1) Prepare feature sets
    uids, feats = prepare_feature_sets(
        crops,
        figure_id=figure_id,
        config=cfg,
        keep_image=keep_image,
        assume_bgr=assume_bgr,
        extract_flip=extract_flip,
        kp_count=kp_count,
        graph=graph,
    )

    if len(feats) < 2:
        logger.warning(f"Not enough prepared crops for reuse detection in figure_id={figure_id}: {len(feats)}")
        return result

    crops_by_uid = build_crops_by_uid(crops, figure_id=figure_id)

    # Determine whether pruning is active
    prune_active = bool(prune is not None and prune.enabled and (
        prune.enable_cbir or prune.enable_grouping or prune.enable_geometry
    ))
    if debug:
        logger.debug(
            f"[reuse] figure_id={figure_id} panels={len(feats)} "
            f"save_artifacts={pol.artifacts} prune={'on' if prune_active else 'off'}"
        )

    candidates_by_src: Optional[Dict[int, List[int]]] = None
    score = None

    if prune_active:
        candidates_by_src, score = build_candidates_by_src(
            crops_by_uid=crops_by_uid,
            uids=uids,
            graph=graph,
            assume_bgr=assume_bgr,
            prune=prune,  # type: ignore[arg-type]
            figure_id=figure_id,
            debug=debug,
        )

    # 2) Match loop
    t0 = time.time()
    for src_i in range(0, len(feats) - 1):
        cand = None if candidates_by_src is None else candidates_by_src.get(src_i, [])
        score_row = score[src_i] if score is not None else None

        match_one_to_many(
            src_i=src_i,
            uids=uids,
            feats=feats,
            graph=graph,
            figure_id=figure_id,
            config=cfg,
            root=root,
            only_same_class=only_same_class,
            min_matched_keypoints=min_matched_keypoints,
            matches_by_source=result.matches_by_source,
            no_matches_by_source=result.no_matches_by_source,
            save_policy=pol,
            candidate_js=cand if prune_active else None,
            score_row=score_row,
            prune=prune if prune_active else None,
            debug=debug,
            debug_pairs=debug_pairs,
            crops_by_uid=crops_by_uid if prune_active else None,
            assume_bgr=assume_bgr,
        )

    if debug:
        logger.debug(f"[reuse] finished figure_id={figure_id} in {time.time()-t0:.2f}s matches={len(graph.pairs)}")

    # 3) Summary JSON
    if root is not None:
        _write_json(root / "reuse_summary.json", result.to_json_dict())

    return result
