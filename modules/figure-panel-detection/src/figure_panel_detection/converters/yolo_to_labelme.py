from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def yolo_txt_to_labelme(
    image_path: Path,
    yolo_txt_path: Path,
    names: Dict[int, str],
    image_hw: Tuple[int, int],
) -> Dict[str, Any]:
    """Convert YOLO txt (normalized xywh) to LabelMe JSON. Placeholder; expand for polygons if needed."""
    h, w = image_hw
    shapes: List[Dict[str, Any]] = []
    if yolo_txt_path.exists():
        for line in yolo_txt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            shapes.append(
                {
                    "label": names.get(cls, str(cls)),
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )

    return {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
