from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import json
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import replace
Detections = np.ndarray


@dataclass(frozen=True)
class Detection:
    """Single detection box in ORIGINAL image pixel coordinates."""
    xyxy: Tuple[int, int, int, int]
    conf: float
    class_id: int
    class_name: str
    panel_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xyxy": [int(x) for x in self.xyxy],
            "conf": float(self.conf),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "panel_id": None if self.panel_id is None else int(self.panel_id),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Detection":
        xyxy = d.get("xyxy", d.get("box_xyxy"))
        panel_id = d.get("panel_id", None)
        if xyxy is None:
            raise ValueError("Detection.from_dict: missing 'xyxy'")
        return Detection(
            xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            conf=float(d.get("conf", 0.0)),
            class_id=int(d.get("class_id", d.get("cls", 0))),
            class_name=str(d.get("class_name", "")),
            panel_id=None if panel_id is None else int(panel_id),
        )


@dataclass(frozen=True)
class DetectionResult:
    """
    Result of a prediction on one image.
    """
    detections: List[Detection]
    det_xyxy_conf_cls: Detections
    image_shape: Tuple[int, int]
    names: Dict[int, str]

    def to_dict(self, *, include_raw: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "image_shape": [int(self.image_shape[0]), int(self.image_shape[1])],
            # JSON requires string keys; keep it explicit.
            "names": {str(k): str(v) for k, v in self.names.items()},
            "detections": [d.to_dict() for d in self.detections],
        }
        if include_raw:
            out["det_xyxy_conf_cls"] = (
                self.det_xyxy_conf_cls.tolist()
                if self.det_xyxy_conf_cls is not None
                else []
            )
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DetectionResult":
        image_shape = d.get("image_shape", [0, 0])
        names_in = d.get("names", {}) or {}
        names = {int(k): str(v) for k, v in names_in.items()}

        det_list = d.get("det_xyxy_conf_cls", None)
        if det_list is None:
            det = np.zeros((0, 6), dtype=np.float32)
        else:
            det = np.asarray(det_list, dtype=np.float32)
            if det.ndim != 2 or det.shape[1] != 6:
                raise ValueError(f"DetectionResult.from_dict: det_xyxy_conf_cls must be Nx6, got {det.shape}")

        dets_in = d.get("detections", None)
        if dets_in is None:
            # If not provided, reconstruct from raw array (if available)
            detections = []
            for row in det:
                x1, y1, x2, y2, conf, cls = row.tolist()
                cls_id = int(round(float(cls)))
                detections.append(
                    Detection(
                        xyxy=(int(x1), int(y1), int(x2), int(y2)),
                        conf=float(conf),
                        class_id=cls_id,
                        class_name=names.get(cls_id, str(cls_id)),
                    )
                )
        else:
            detections = [Detection.from_dict(x) for x in dets_in]

        return DetectionResult(
            detections=detections,
            det_xyxy_conf_cls=det,
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            names=names,
        )

    def to_json(self, path: Optional[Union[str, Path]] = None, *, include_raw: bool = False, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(include_raw=include_raw), indent=indent)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    @staticmethod
    def from_json(x: Union[str, Path, Dict[str, Any]]) -> "DetectionResult":
        if isinstance(x, dict):
            return DetectionResult.from_dict(x)
        p = Path(x)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            # treat as JSON string
            data = json.loads(str(x))
        return DetectionResult.from_dict(data)

    def boxes_xyxy(self) -> np.ndarray:
        """Return Nx4 float32 [x1,y1,x2,y2]."""
        if self.det_xyxy_conf_cls is None or len(self.det_xyxy_conf_cls) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return self.det_xyxy_conf_cls[:, :4].astype(np.float32, copy=False)

    def boxes_xywh(self) -> np.ndarray:
        """Return Nx4 float32 [x,y,w,h]."""
        from .geometry import xyxy_to_xywh
        return xyxy_to_xywh(self.boxes_xyxy())

    def boxes_xyxy_norm(self) -> np.ndarray:
        """Return Nx4 float32 normalized coords [0..1]."""
        from .geometry import normalize_xyxy
        H, W = self.image_shape
        return normalize_xyxy(self.boxes_xyxy(), w=int(W), h=int(H))

    def assign_ids(self, *, order: str = "yx") -> "DetectionResult":
        """
        Return a NEW DetectionResult with deterministic panel_id assigned.

        order:
        - "yx": top-to-bottom, then left-to-right (recommended)
        - "conf_desc": highest confidence first
        - "area_desc": largest area first
        - "as_is": keep current order, just assign 0..N-1
        """
        if not self.detections:
            return self

        order_l = (order or "yx").lower().strip()
        dets = list(self.detections)

        def _area(d) -> int:
            x1, y1, x2, y2 = d.xyxy
            return max(0, x2 - x1) * max(0, y2 - y1)

        if order_l == "as_is":
            idx = list(range(len(dets)))

        elif order_l == "yx":
            # top-to-bottom by y1, then left-to-right by x1
            idx = sorted(range(len(dets)), key=lambda i: (dets[i].xyxy[1], dets[i].xyxy[0]))

        elif order_l == "conf_desc":
            idx = sorted(
                range(len(dets)),
                key=lambda i: (-float(dets[i].conf), dets[i].xyxy[1], dets[i].xyxy[0]),
            )

        elif order_l == "area_desc":
            idx = sorted(
                range(len(dets)),
                key=lambda i: (-_area(dets[i]), dets[i].xyxy[1], dets[i].xyxy[0]),
            )

        else:
            raise ValueError(f"Unsupported order={order!r}. Use: yx|conf_desc|area_desc|as_is")

        # Reorder raw det array to match the reordered detections
        raw = self.det_xyxy_conf_cls
        if raw is None or len(raw) == 0:
            new_raw = raw
        else:
            import numpy as np
            new_raw = raw[np.asarray(idx, dtype=np.int64)]

        # Create new Detection objects with panel_id
        new_dets = []
        for pid, i in enumerate(idx):
            d = dets[i]
            new_dets.append(
                Detection(
                    xyxy=d.xyxy,
                    conf=d.conf,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    panel_id=int(pid),
                )
            )

        return DetectionResult(
            detections=new_dets,
            det_xyxy_conf_cls=new_raw,
            image_shape=self.image_shape,
            names=self.names,
        )


@dataclass(frozen=True)
class Crop:
    """
    One cropped region extracted from an image.
    """
    image: object
    xyxy: Tuple[int, int, int, int]
    conf: float
    class_id: int
    class_name: str
    det_index: int
    panel_id: Optional[int] = None

    def to_dict(self, *, meta_only: bool = True) -> Dict[str, Any]:
        if not meta_only:
            raise ValueError("Crop.to_dict(meta_only=False) is not supported (do not serialize pixels).")
        return {
            "xyxy": [int(x) for x in self.xyxy],
            "conf": float(self.conf),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "det_index": int(self.det_index),
            "panel_id": None if self.panel_id is None else int(self.panel_id),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Crop":
        xyxy = d.get("xyxy", d.get("box_xyxy"))
        panel_id = d.get("panel_id", None)
        if xyxy is None:
            raise ValueError("Crop.from_dict: missing 'xyxy'")
        # image is not serialized; keep as None placeholder
        return Crop(
            image=None,
            xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            conf=float(d.get("conf", 0.0)),
            class_id=int(d.get("class_id", d.get("cls", 0))),
            class_name=str(d.get("class_name", "")),
            det_index=int(d.get("det_index", -1)),
            panel_id=None if panel_id is None else int(panel_id),
        )
