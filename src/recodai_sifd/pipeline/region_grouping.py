from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from typing import Mapping

import numpy as np
from loguru import logger
from PIL import Image

from recodai_sifd.pipeline.reuse_detection import PairMatchEdge


@dataclass(frozen=True)
class GroupResult:
    """One connected component (cluster) and its full-image instance mask."""
    group_id: int
    panel_uids: List[str]
    instance_mask: np.ndarray  # (H,W) uint8 {0,1}


@dataclass(frozen=True)
class GroupingResult:
    """Pure in-memory outputs for grouping (no file I/O)."""
    figure_shape_hw: tuple[int, int]          # (H, W) normalized
    groups: list[list[str]]                  # connected components (uids)
    group_results: list[GroupResult]         # each contains instance_mask np.ndarray
    combined_mask: np.ndarray                # (H,W) uint8 {0,1}

    @property
    def instance_masks_by_id(self) -> dict[int, np.ndarray]:
        # IMPORTANT: these are the actual arrays (no copies)
        return {gr.group_id: gr.instance_mask for gr in self.group_results}


def compute_grouping_result(
    *,
    figure_shape_hw: tuple[int, int],
    pairs: Sequence[PairMatchEdge],
    panel_xyxy_by_uid: Optional[dict[str, tuple[int, int, int, int]]] = None,
    crops_by_uid: Optional[dict[str, Any]] = None,
    shape_is_wh: bool = False,
    relaxed_uid_lookup: bool = True,
) -> GroupingResult:
    """
    Compute grouping masks in-memory (no file I/O).

    Returns GroupingResult with:
      - per-instance masks (uint8 0/1)
      - combined_mask (uint8 0/1)
    """
    H, W = _normalize_hw(figure_shape_hw, shape_is_wh=shape_is_wh)

    groups = group_connected_components(pairs)
    group_results = build_group_instance_masks(
        figure_shape_hw=(H, W),   # normalized (H,W)
        pairs=pairs,
        panel_xyxy_by_uid=panel_xyxy_by_uid,
        crops_by_uid=crops_by_uid,
        groups=groups,
        shape_is_wh=False,        # already normalized
        relaxed_uid_lookup=relaxed_uid_lookup,
    )

    combined_mask = build_combined_mask(group_results)

    return GroupingResult(
        figure_shape_hw=(H, W),
        groups=[list(g) for g in groups],
        group_results=list(group_results),
        combined_mask=combined_mask,
    )

def group_connected_components(pairs: Sequence[PairMatchEdge]) -> List[List[str]]:
    """
    Connected components over panel graph using matched edges.

    Parameters
    ----------
    pairs : sequence of PairMatchEdge
        Matched edges only.

    Returns
    -------
    list[list[str]]
        Groups (each is list of panel uids).
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in pairs:
        union(e.panel_a, e.panel_b)

    groups: Dict[str, List[str]] = {}
    for node in list(parent.keys()):
        r = find(node)
        groups.setdefault(r, []).append(node)

    out: List[List[str]] = []
    for g in groups.values():
        out.append(sorted(g))
    out.sort(key=lambda x: (-len(x), x[0] if x else ""))
    return out


def build_group_instance_masks(
    *,
    figure_shape_hw: Tuple[int, int],
    pairs: Sequence[PairMatchEdge],
    panel_xyxy_by_uid: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    crops_by_uid: Optional[Dict[str, Any]] = None,
    groups: Optional[Sequence[Sequence[str]]] = None,
    shape_is_wh: bool = False,
    relaxed_uid_lookup: bool = True,
) -> List[GroupResult]:
    """
    Build one full-image instance mask per connected component.

    Parameters
    ----------
    figure_shape_hw : (int, int)
        Full image shape (H, W) by default.
        If you pass PIL Image.size (W, H), set shape_is_wh=True.
    pairs : sequence of PairMatchEdge
        Matched edges with crop-sized masks (mask_a/mask_b).
    panel_xyxy_by_uid : dict[str, (x1,y1,x2,y2)], optional
        Recommended. Use: {uid: node.xyxy for uid,node in res.graph.panels.items()}.
    crops_by_uid : dict[str, Crop], optional
        Fallback. crop must have .xyxy.
    groups : optional
        If None, computed from pairs.
    shape_is_wh : bool, optional
        If True, interprets figure_shape_hw as (W, H) and flips to (H, W).
    relaxed_uid_lookup : bool, optional
        If True, tries a fallback lookup by stripping trailing '::cls=...'.

    Returns
    -------
    list[GroupResult]
        Instance masks per group.
    """
    H, W = _normalize_hw(figure_shape_hw, shape_is_wh=shape_is_wh)

    if groups is None:
        groups = group_connected_components(pairs)

    panel_to_gid: Dict[str, int] = {}
    for gid, panel_uids in enumerate(groups, start=1):
        for uid in panel_uids:
            panel_to_gid[uid] = gid

    group_masks: Dict[int, np.ndarray] = {
        gid: np.zeros((H, W), dtype=np.uint8) for gid in range(1, len(groups) + 1)
    }

    for e in pairs:
        gid_a = panel_to_gid.get(e.panel_a)
        gid_b = panel_to_gid.get(e.panel_b)
        if gid_a is None or gid_b is None:
            continue
        gid = gid_a if gid_a == gid_b else gid_a  # robust

        _paste_uid_mask_into_full(
            full_mask=group_masks[gid],
            uid=e.panel_a,
            crop_mask=e.mask_a,
            panel_xyxy_by_uid=panel_xyxy_by_uid,
            crops_by_uid=crops_by_uid,
            relaxed_uid_lookup=relaxed_uid_lookup,
        )
        _paste_uid_mask_into_full(
            full_mask=group_masks[gid],
            uid=e.panel_b,
            crop_mask=e.mask_b,
            panel_xyxy_by_uid=panel_xyxy_by_uid,
            crops_by_uid=crops_by_uid,
            relaxed_uid_lookup=relaxed_uid_lookup,
        )

    results: List[GroupResult] = []
    for gid, panel_uids in enumerate(groups, start=1):
        results.append(GroupResult(group_id=gid, panel_uids=list(panel_uids), instance_mask=group_masks[gid]))
    return results


def build_combined_mask(group_results: Sequence[GroupResult]) -> np.ndarray:
    """
    Union of all instance masks (binary uint8 0/1).
    """
    if not group_results:
        return np.zeros((1, 1), dtype=np.uint8)

    H, W = group_results[0].instance_mask.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)
    for gr in group_results:
        out[gr.instance_mask.astype(bool)] = 1
    return out


def save_mask_png(
    mask: np.ndarray,
    path: str | Path,
    *,
    scale01_to_255: bool = True,
) -> Path:
    """
    Save mask as PNG.

    - uint16 label masks saved as 16-bit PNG (mode "I;16") without scaling.
    - uint8 binary masks can be {0,1} or {0,255}.
      If scale01_to_255=True and mask looks like {0,1}, it is scaled to {0,255}
      for easier viewing in normal image viewers.

      Note: Pillow can save/display 0/1 masks, but most viewers will show them almost-black. So scaling is usually helpful for “debug viewing”.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if mask.dtype == np.uint16:
        im = Image.fromarray(mask, mode="I;16")
        im.save(out)
        return out

    m = mask.astype(np.uint8)

    if scale01_to_255:
        mx = int(m.max()) if m.size else 0
        if mx <= 1:
            m = (m * 255).astype(np.uint8)

    im = Image.fromarray(m, mode="L")
    im.save(out)
    return out


def save_grouping_outputs(
    *,
    figure_shape_hw: Tuple[int, int],
    pairs: Sequence[PairMatchEdge],
    out_dir: str | Path,
    panel_xyxy_by_uid: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    crops_by_uid: Optional[Dict[str, Any]] = None,
    shape_is_wh: bool = False,
    relaxed_uid_lookup: bool = True,
    scale01_to_255: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper that saves:
    - instances/instance_###.png  (binary per-group)
    - combined_mask.png           (binary union)
    - groups.json                 (metadata)

    Recommended input: panel_xyxy_by_uid from res.graph.panels.

    Returns
    -------
    dict
        Output paths + counts.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instances").mkdir(parents=True, exist_ok=True)

    groups = group_connected_components(pairs)
    group_results = build_group_instance_masks(
        figure_shape_hw=figure_shape_hw,
        pairs=pairs,
        panel_xyxy_by_uid=panel_xyxy_by_uid,
        crops_by_uid=crops_by_uid,
        groups=groups,
        shape_is_wh=shape_is_wh,
        relaxed_uid_lookup=relaxed_uid_lookup,
    )

    combined = build_combined_mask(group_results)
    combined_path = save_mask_png(
        combined, 
        out / "combined_mask.png",
        scale01_to_255=scale01_to_255
    )

    for gr in group_results:
        save_mask_png(
            gr.instance_mask, 
            out / "instances" / f"instance_{gr.group_id:03d}.png",
            scale01_to_255=scale01_to_255
        )

    H, W = _normalize_hw(figure_shape_hw, shape_is_wh=shape_is_wh)
    meta = {
        "figure_shape_hw": [H, W],
        "num_instances": len(group_results),
        "instances": [
            {
                "group_id": gr.group_id,
                "panel_uids": gr.panel_uids,
                "instance_mask_file": f"instances/instance_{gr.group_id:03d}.png",
            }
            for gr in group_results
        ],
        "outputs": {
            "combined_mask": str(combined_path.name),
        },
    }
    with (out / "groups.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "out_dir": str(out),
        "groups_json": str(out / "groups.json"),
        "instances_label": str(out / "instances_label.png"),
        "combined_mask": str(out / "combined_mask.png"),
        "instances_dir": str(out / "instances"),
        "num_instances": len(group_results),
    }


# -------------------------
# Internal helpers
# -------------------------

def _normalize_hw(shape_hw: Tuple[int, int], *, shape_is_wh: bool) -> Tuple[int, int]:
    a, b = int(shape_hw[0]), int(shape_hw[1])
    return (b, a) if shape_is_wh else (a, b)


def _strip_cls(uid: str) -> str:
    # removes trailing ::cls=NNN if present
    i = uid.rfind("::cls=")
    return uid[:i] if i != -1 else uid


def _get_xyxy_for_uid(
    uid: str,
    *,
    panel_xyxy_by_uid: Optional[Dict[str, Tuple[int, int, int, int]]],
    crops_by_uid: Optional[Dict[str, Any]],
    relaxed_uid_lookup: bool,
) -> Optional[Tuple[int, int, int, int]]:
    if panel_xyxy_by_uid is not None:
        xy = panel_xyxy_by_uid.get(uid)
        if xy is not None:
            return tuple(map(int, xy))
        if relaxed_uid_lookup:
            uid2 = _strip_cls(uid)
            xy = panel_xyxy_by_uid.get(uid2)
            if xy is not None:
                return tuple(map(int, xy))

    if crops_by_uid is not None:
        c = crops_by_uid.get(uid)
        if c is not None:
            return tuple(map(int, c.xyxy))
        if relaxed_uid_lookup:
            uid2 = _strip_cls(uid)
            c = crops_by_uid.get(uid2)
            if c is not None:
                return tuple(map(int, c.xyxy))

    return None


def _paste_uid_mask_into_full(
    *,
    full_mask: np.ndarray,
    uid: str,
    crop_mask: np.ndarray,
    panel_xyxy_by_uid: Optional[Dict[str, Tuple[int, int, int, int]]],
    crops_by_uid: Optional[Dict[str, Any]],
    relaxed_uid_lookup: bool,
) -> None:
    xyxy = _get_xyxy_for_uid(
        uid,
        panel_xyxy_by_uid=panel_xyxy_by_uid,
        crops_by_uid=crops_by_uid,
        relaxed_uid_lookup=relaxed_uid_lookup,
    )
    if xyxy is None:
        logger.warning(f"Missing xyxy for uid; cannot paste mask: {uid}")
        return
    _paste_crop_mask_into_full(full_mask=full_mask, xyxy=xyxy, crop_mask=crop_mask)


def _paste_crop_mask_into_full(*, full_mask: np.ndarray, xyxy: Tuple[int, int, int, int], crop_mask: np.ndarray) -> None:
    """
    Paste crop_mask (crop coords) into full_mask (figure coords) using xyxy.
    Union by OR (write 1 where crop_mask is nonzero).
    """
    x1, y1, x2, y2 = map(int, xyxy)
    Hf, Wf = full_mask.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return

    # clip to full image bounds
    xx1 = max(0, x1)
    yy1 = max(0, y1)
    xx2 = min(Wf, x2)
    yy2 = min(Hf, y2)
    if xx2 <= xx1 or yy2 <= yy1:
        return

    m = crop_mask
    if m is None:
        return
    if m.ndim != 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        logger.warning("crop_mask is not 2D after squeeze; skipping paste.")
        return

    exp_w = x2 - x1
    exp_h = y2 - y1
    mh, mw = int(m.shape[0]), int(m.shape[1])

    # best-effort align if mismatch
    if (mh != exp_h) or (mw != exp_w):
        h_use = min(mh, exp_h)
        w_use = min(mw, exp_w)
        m = m[:h_use, :w_use]
        xx2 = min(xx1 + w_use, Wf)
        yy2 = min(yy1 + h_use, Hf)

    off_x = xx1 - x1
    off_y = yy1 - y1
    h = yy2 - yy1
    w = xx2 - xx1

    m_slice = m[off_y : off_y + h, off_x : off_x + w].astype(bool)
    if not m_slice.any():
        return

    view = full_mask[yy1:yy2, xx1:xx2]
    view[m_slice] = 1 # 255
    full_mask[yy1:yy2, xx1:xx2] = view
