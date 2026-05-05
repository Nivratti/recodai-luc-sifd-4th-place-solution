from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import RunConfig


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def copy2_safe(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def write_run_config(out_dir: Path, cfg: RunConfig, extra: Optional[Dict[str, Any]] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = asdict(cfg)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if extra:
        payload.update(extra)
    p = out_dir / "run_config.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def xyxy_to_xywh_norm(det_xyxy: np.ndarray, im0_hw: Tuple[int, int]) -> np.ndarray:
    """Convert xyxy pixel boxes to YOLO-normalized xywh."""
    h, w = im0_hw
    x1, y1, x2, y2 = det_xyxy[:, 0], det_xyxy[:, 1], det_xyxy[:, 2], det_xyxy[:, 3]
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.stack([cx, cy, bw, bh], axis=1)


def save_labels_yolo(
    lbl_path: Path,
    det_scaled: Optional[np.ndarray],
    im0_shape_hw: Tuple[int, int],
    save_conf: bool,
    save_empty: bool = True,
) -> None:
    """Write YOLO txt labels from Nx6 detections in original coords."""
    ensure_parent(lbl_path)
    if det_scaled is None or len(det_scaled) == 0:
        if save_empty:
            lbl_path.write_text("", encoding="utf-8")
        return

    det_scaled = np.asarray(det_scaled, dtype=np.float32)
    xywh = xyxy_to_xywh_norm(det_scaled[:, 0:4], im0_shape_hw)
    cls = det_scaled[:, 5].astype(np.int64)
    conf = det_scaled[:, 4].astype(np.float32)

    lines: List[str] = []
    for i in range(det_scaled.shape[0]):
        if save_conf:
            lines.append(
                f"{int(cls[i])} {xywh[i,0]:.6f} {xywh[i,1]:.6f} {xywh[i,2]:.6f} {xywh[i,3]:.6f} {float(conf[i]):.6f}"
            )
        else:
            lines.append(
                f"{int(cls[i])} {xywh[i,0]:.6f} {xywh[i,1]:.6f} {xywh[i,2]:.6f} {xywh[i,3]:.6f}"
            )
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_layout_mapping(path: Path, items: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
