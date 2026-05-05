from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
from pathlib import Path
import time

import numpy as np

from ..cropping.crop_regions import CropConfig, extract_crops as _extract_crops_np
from ..filtering.dedup import dedup_detections
from ..filtering.keep_classes import filter_by_class_ids, parse_keep_classes_tokens
from ..viz.render import render_detections
from ..yolo.onnx_predictor import OnnxPredictorConfig, YoloOnnxPredictor
from .image_io import load_image_bgr, to_pil, to_rgb
from .types import Crop, Detection, DetectionResult, Detections
from .postprocess import postprocess_det_xyxy_conf_cls
from .tiling import iter_tiles


def _load_names_required(model_onnx_path: str, names_arg: Optional[Union[str, Dict[int, str]]]) -> Tuple[Dict[int, str], Optional[Path]]:
    """
    Strict names loader:
    - If names_arg is a dict: use it
    - If names_arg is a path: load JSON
    - Else auto-load <model_stem>.json next to the ONNX model
    If nothing found -> SystemExit (consistent with CLI strictness).
    """
    if isinstance(names_arg, dict):
        # Normalize keys to int
        out: Dict[int, str] = {}
        for k, v in names_arg.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        if not out:
            raise SystemExit("[ERR] names mapping provided, but it is empty/invalid.")
        return out, None

    if isinstance(names_arg, (str, Path)):
        p = Path(names_arg)
        if not p.exists():
            raise SystemExit(f"[ERR] names file not found: {p}")
        data = _read_json(p)
        return _parse_names_payload(data, p), p

    model_p = Path(model_onnx_path)
    sidecar = model_p.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(
            "[ERR] Missing class names JSON.\n"
            f"      Expected sidecar next to model: {sidecar}\n"
            "      Or pass names=<path_to_names.json> or names=<dict>."
        )

    data = _read_json(sidecar)
    return _parse_names_payload(data, sidecar), sidecar


def _read_json(p: Path) -> Any:
    import json
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_names_payload(data: Any, src: Path) -> Dict[int, str]:
    """
    Accepts either:
      - {"class_names": {"0": "Blots", "1": "Microscopy", ...}}
      - {"names": ["Blots", "Microscopy", ...]} (YOLO style)
      - {"0": "Blots", "1": "Microscopy", ...} (flat)
    """
    if not isinstance(data, dict):
        raise SystemExit(f"[ERR] Invalid names JSON (expected object/dict): {src}")

    if "class_names" in data and isinstance(data["class_names"], dict):
        payload = data["class_names"]
        out: Dict[int, str] = {}
        for k, v in payload.items():
            out[int(k)] = str(v)
        return out

    if "names" in data and isinstance(data["names"], (list, tuple)):
        out = {int(i): str(name) for i, name in enumerate(list(data["names"]))}
        return out

    # flat mapping
    out2: Dict[int, str] = {}
    ok = True
    for k, v in data.items():
        try:
            out2[int(k)] = str(v)
        except Exception:
            ok = False
            break
    if ok and out2:
        return out2

    raise SystemExit(
        "[ERR] Could not parse names JSON.\n"
        f"      File: {src}\n"
        "      Expected one of:\n"
        "        {\"class_names\": {\"0\": \"Blots\", \"1\": \"Microscopy\"}}\n"
        "        {\"names\": [\"Blots\", \"Microscopy\"]}\n"
        "        {\"0\": \"Blots\", \"1\": \"Microscopy\"}\n"
    )


def _resolve_keep_ids(keep_classes: Optional[Sequence[Union[str, int]]], names: Dict[int, str]) -> Optional[List[int]]:
    if not keep_classes:
        return None
    toks: List[str] = []
    direct_ids: List[int] = []
    for x in keep_classes:
        if isinstance(x, int):
            direct_ids.append(int(x))
        else:
            s = str(x).strip()
            if s.isdigit():
                direct_ids.append(int(s))
            else:
                toks.append(s)
    ids: List[int] = []
    if toks:
        ids.extend(parse_keep_classes_tokens(toks, names) or [])
    if direct_ids:
        ids.extend(direct_ids)
    # unique, stable
    ids2 = sorted(set(int(i) for i in ids))
    return ids2 if ids2 else None


class FigurePanelDetector:
    """
    Import-friendly, in-memory API for figure panel detection (YOLOv5 ONNX).

    Example
    -------
    from figure_panel_detection import FigurePanelDetector

    det = FigurePanelDetector(model_onnx="model.onnx")  # auto providers
    res = det.predict("/path/to/image.png")
    vis = det.visualize("/path/to/image.png", res, return_format="rgb")
    crops = det.extract_crops("/path/to/image.png", res, pad_pct=0.05, return_format="pil")
    """

    def __init__(
        self,
        model_onnx: Union[str, Path],
        *,
        names: Optional[Union[str, Dict[int, str]]] = None,
        imgsz: int = 640,
        providers: Optional[List[str]] = None,
        fp16: bool = False,
    ):
        self.model_onnx = str(model_onnx)
        self.imgsz = int(imgsz)

        # Strict (matches CLI behavior): must exist via names arg or sidecar next to model
        self.names, self.names_source = _load_names_required(self.model_onnx, names)

        # Auto providers: CUDA -> CPU fallback (if available)
        if providers is None:
            providers = _default_ort_providers()

        self.predictor = YoloOnnxPredictor(
            model_path=self.model_onnx,
            cfg=OnnxPredictorConfig(imgsz=self.imgsz, fp16=bool(fp16), providers=providers),
        )

    @staticmethod
    def _shift_det_xyxy(det: np.ndarray, dx: int, dy: int) -> np.ndarray:
        if det is None or len(det) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        out = np.asarray(det, dtype=np.float32).copy()
        out[:, 0] += dx
        out[:, 2] += dx
        out[:, 1] += dy
        out[:, 3] += dy
        return out

    def predict(
        self,
        image: Any,
        *,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        keep_classes: Optional[Sequence[Union[str, int]]] = None,
        dedup: bool = False,
        dedup_iou: float = 0.90,
        dedup_merge: bool = False,
        dedup_class_agnostic: bool = False,
        input_color: str = "bgr",
        # --- post filters / ordering ---
        min_area_px: Optional[float] = None,
        min_area_frac: Optional[float] = None,
        max_area_px: Optional[float] = None,
        max_area_frac: Optional[float] = None,
        min_aspect: Optional[float] = None,
        max_aspect: Optional[float] = None,
        topk: Optional[int] = None,
        topk_per_class: Optional[int] = None,
        sort: str = "yx",
        return_timing: bool = False,
        assign_panel_ids: bool = True,
        panel_id_order: str = "yx",
    ) -> DetectionResult:
        """
        Run detection on a single image input (path / np / PIL).

        input_color:
            Only applies when `image` is a numpy array with 3 channels.
            Use "rgb" if your array is RGB; default is "bgr" (OpenCV convention).
        """
        t0_total = time.perf_counter()
        timing: Dict[str, float] = {}

        t0 = time.perf_counter()
        im0 = load_image_bgr(image, input_color=input_color) # uint8 BGR
        timing["load_image"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        det = self.predictor.predict_bgr(
            im0,
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            max_det=int(max_det),
            classes=classes,
            agnostic_nms=bool(agnostic_nms),
        )
        timing["infer"] = time.perf_counter() - t0

        t0 = time.perf_counter()

        keep_ids = _resolve_keep_ids(keep_classes, self.names)
        if keep_ids:
            det_f = filter_by_class_ids(det, set(keep_ids))
            det = det_f if (det_f is not None and len(det_f) > 0) else np.zeros((0, 6), dtype=np.float32)

        if dedup:
            det = dedup_detections(
                det,
                iou_thres=float(dedup_iou),
                merge=bool(dedup_merge),
                class_agnostic=bool(dedup_class_agnostic),
            )
            det = det if (det is not None and len(det) > 0) else np.zeros((0, 6), dtype=np.float32)

        # postprocess: area/aspect/topk/sort
        H, W = im0.shape[:2]
        det = postprocess_det_xyxy_conf_cls(
            det,
            image_shape=(H, W),
            min_area_px=min_area_px,
            min_area_frac=min_area_frac,
            max_area_px=max_area_px,
            max_area_frac=max_area_frac,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            topk=topk,
            topk_per_class=topk_per_class,
            sort=sort,
        )

        det = np.asarray(det, dtype=np.float32)
        det = det.reshape((-1, 6)) if det.size else np.zeros((0, 6), dtype=np.float32)

        H, W = im0.shape[:2]
        det_objs: List[Detection] = []
        for row in det:
            x1, y1, x2, y2, conf, cls = row.tolist()
            cls_id = int(round(float(cls)))
            name = self.names.get(cls_id, str(cls_id))
            det_objs.append(
                Detection(
                    xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    conf=float(conf),
                    class_id=cls_id,
                    class_name=name,
                )
            )

        timing["post"] = time.perf_counter() - t0
        timing["total"] = time.perf_counter() - t0_total

        res = DetectionResult(
            detections=det_objs,
            det_xyxy_conf_cls=det,
            image_shape=(int(H), int(W)),
            names=self.names,
        )

        if assign_panel_ids:
            res = res.assign_ids(order=panel_id_order)

        if return_timing:
            return res, timing
        return res

    def predict_tiled(
        self,
        image: Any,
        *,
        tile: Union[int, Tuple[int, int]] = 1024,
        overlap: float = 0.2,
        merge_iou: float = 0.6,
        merge_class_agnostic: bool = False,

        # ---- same args as predict() ----
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        keep_classes: Optional[Sequence[Union[str, int]]] = None,

        # dedup after merge (optional)
        dedup: bool = True,
        dedup_iou: float = 0.90,
        dedup_merge: bool = False,
        dedup_class_agnostic: bool = False,

        # post filters / ordering (your #4 params)
        min_area_px: Optional[float] = None,
        min_area_frac: Optional[float] = None,
        max_area_px: Optional[float] = None,
        max_area_frac: Optional[float] = None,
        min_aspect: Optional[float] = None,
        max_aspect: Optional[float] = None,
        topk: Optional[int] = None,
        topk_per_class: Optional[int] = None,
        sort: str = "conf_desc",

        input_color: str = "bgr",
    ) -> DetectionResult:
        """
        Tile-based prediction for large images.

        Flow:
        - normalize input once
        - run detection on overlapping tiles
        - shift tile detections to full-image coords
        - merge duplicates across overlaps (IoU merge_iou)
        - optional dedup + postprocess (area/aspect/topk/sort)
        """
        im0 = load_image_bgr(image, input_color=input_color)
        H, W = im0.shape[:2]

        # If image is small, fallback to normal predict
        if isinstance(tile, int):
            th, tw = tile, tile
        else:
            th, tw = int(tile[0]), int(tile[1])

        if H <= th and W <= tw:
            return self.predict(
                im0,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                classes=classes,
                agnostic_nms=agnostic_nms,
                keep_classes=keep_classes,
                dedup=dedup,
                dedup_iou=dedup_iou,
                dedup_merge=dedup_merge,
                dedup_class_agnostic=dedup_class_agnostic,
                min_area_px=min_area_px,
                min_area_frac=min_area_frac,
                max_area_px=max_area_px,
                max_area_frac=max_area_frac,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                topk=topk,
                topk_per_class=topk_per_class,
                sort=sort,
                input_color="bgr",
            )

        all_det = []
        for (x1, y1, x2, y2) in iter_tiles(H, W, tile=tile, overlap=overlap):
            tile_im = im0[y1:y2, x1:x2]
            # Use existing predict(), but take only raw array to avoid rewriting internals
            tile_res = self.predict(
                tile_im,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                classes=classes,
                agnostic_nms=agnostic_nms,
                keep_classes=keep_classes,
                dedup=False,  # merge across tiles later
                # IMPORTANT: don't apply area_frac filters per-tile (would be wrong)
                min_area_px=None,
                min_area_frac=None,
                max_area_px=None,
                max_area_frac=None,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                topk=None,
                topk_per_class=None,
                sort="conf_desc",
                input_color="bgr",
            )
            det_tile = tile_res.det_xyxy_conf_cls
            det_shifted = self._shift_det_xyxy(det_tile, dx=x1, dy=y1)
            if len(det_shifted) > 0:
                all_det.append(det_shifted)

        if not all_det:
            det_merged = np.zeros((0, 6), dtype=np.float32)
        else:
            det_merged = np.concatenate(all_det, axis=0).astype(np.float32, copy=False)

            # merge across overlaps
            # Reuse existing dedup_detections() as “merge NMS”
            det_merged = dedup_detections(
                det_merged,
                iou_thres=float(merge_iou),
                merge=False,
                class_agnostic=bool(merge_class_agnostic),
            )
            det_merged = det_merged if (det_merged is not None and len(det_merged) > 0) else np.zeros((0, 6), dtype=np.float32)

        # optional final dedup on full image
        if dedup and len(det_merged) > 0:
            det_merged = dedup_detections(
                det_merged,
                iou_thres=float(dedup_iou),
                merge=bool(dedup_merge),
                class_agnostic=bool(dedup_class_agnostic),
            )
            det_merged = det_merged if (det_merged is not None and len(det_merged) > 0) else np.zeros((0, 6), dtype=np.float32)

        # final postprocess on FULL image shape
        det_merged = postprocess_det_xyxy_conf_cls(
            det_merged,
            image_shape=(H, W),
            min_area_px=min_area_px,
            min_area_frac=min_area_frac,
            max_area_px=max_area_px,
            max_area_frac=max_area_frac,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            topk=topk,
            topk_per_class=topk_per_class,
            sort=sort,
        )

        # build DetectionResult (same way predict() does)
        detections = []
        for row in det_merged:
            x1, y1, x2, y2, conf, cls = row.tolist()
            cls_id = int(round(float(cls)))
            detections.append(
                Detection(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    conf=float(conf),
                    class_id=cls_id,
                    class_name=self.names.get(cls_id, str(cls_id)),
                )
            )

        return DetectionResult(
            detections=detections,
            det_xyxy_conf_cls=det_merged,
            image_shape=(H, W),
            names=self.names,
        )

    def predict_batch(
        self,
        images: Sequence[Any],
        *,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        keep_classes: Optional[Sequence[Union[str, int]]] = None,
        dedup: bool = False,
        dedup_iou: float = 0.90,
        dedup_merge: bool = False,
        dedup_class_agnostic: bool = False,
        input_color: str = "bgr",
        # --- post filters / ordering ---
        min_area_px: Optional[float] = None,
        min_area_frac: Optional[float] = None,
        max_area_px: Optional[float] = None,
        max_area_frac: Optional[float] = None,
        min_aspect: Optional[float] = None,
        max_aspect: Optional[float] = None,
        topk: Optional[int] = None,
        topk_per_class: Optional[int] = None,
        sort: str = "conf_desc",
        return_timing: bool = False,
    ) -> List[DetectionResult]:
        ims0 = [load_image_bgr(x, input_color=input_color) for x in images]
        dets = self.predictor.predict_batch_bgr(
            ims0,
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            max_det=int(max_det),
            classes=classes,
            agnostic_nms=bool(agnostic_nms),
        )

        keep_ids = _resolve_keep_ids(keep_classes, self.names)
        out: List[DetectionResult] = []
        for im0, det in zip(ims0, dets):
            det = np.asarray(det, dtype=np.float32)
            if keep_ids:
                det_f = filter_by_class_ids(det, set(keep_ids))
                det = det_f if (det_f is not None and len(det_f) > 0) else np.zeros((0, 6), dtype=np.float32)

            if dedup:
                det = dedup_detections(
                    det,
                    iou_thres=float(dedup_iou),
                    merge=bool(dedup_merge),
                    class_agnostic=bool(dedup_class_agnostic),
                )
                det = det if (det is not None and len(det) > 0) else np.zeros((0, 6), dtype=np.float32)

            H, W = im0.shape[:2]
            det = postprocess_det_xyxy_conf_cls(
                det,
                image_shape=(H, W),
                min_area_px=min_area_px,
                min_area_frac=min_area_frac,
                max_area_px=max_area_px,
                max_area_frac=max_area_frac,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                topk=topk,
                topk_per_class=topk_per_class,
                sort=sort,
            )
            
            H, W = im0.shape[:2]
            det_objs: List[Detection] = []
            for row in det:
                x1, y1, x2, y2, conf, cls = row.tolist()
                cls_id = int(round(float(cls)))
                name = self.names.get(cls_id, str(cls_id))
                det_objs.append(
                    Detection(
                        xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                        conf=float(conf),
                        class_id=cls_id,
                        class_name=name,
                    )
                )

            out.append(
                DetectionResult(
                    detections=det_objs,
                    det_xyxy_conf_cls=det,
                    image_shape=(int(H), int(W)),
                    names=self.names,
                )
            )
        return out

    def predict_crops(
        self,
        image: Any,
        *,
        # --- predict() args ---
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        keep_classes: Optional[Sequence[Union[str, int]]] = None,
        dedup: bool = False,
        dedup_iou: float = 0.90,
        dedup_merge: bool = False,
        dedup_class_agnostic: bool = False,
        input_color: str = "bgr",
        # --- extract_crops() args ---
        pad_px: int = 0,
        pad_pct: float = 0.0,
        expand_mode: str = "margin",  # margin|context
        context_gap_px: int = 0,
        return_format: str = "bgr",  # bgr|rgb|pil
        obstacles_xyxy: Optional[np.ndarray] = None,
    ) -> Tuple[DetectionResult, List[Crop]]:
        """
        Convenience: run predict() and extract_crops() in one call.

        Implementation detail:
        - Normalizes the input image ONCE to BGR (uint8) and reuses it for both
            detection and cropping to avoid double decoding/conversion.
        """
        # normalize once
        im0 = load_image_bgr(image, input_color=input_color)

        # predict on normalized BGR
        res = self.predict(
            im0,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            classes=classes,
            agnostic_nms=agnostic_nms,
            keep_classes=keep_classes,
            dedup=dedup,
            dedup_iou=dedup_iou,
            dedup_merge=dedup_merge,
            dedup_class_agnostic=dedup_class_agnostic,
            input_color="bgr",
        )

        # crops from same normalized BGR
        crops = self.extract_crops(
            im0,
            res,
            pad_px=pad_px,
            pad_pct=pad_pct,
            expand_mode=expand_mode,
            context_gap_px=context_gap_px,
            return_format=return_format,
            input_color="bgr",
            obstacles_xyxy=obstacles_xyxy,
        )
        return res, crops

    def save_artifacts(
        self,
        out_dir: Union[str, Path],
        image: Any,
        result: Optional[DetectionResult] = None,
        *,
        # if result is None, we can run predict() using these defaults:
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        keep_classes: Optional[Sequence[Union[str, int]]] = None,
        dedup: bool = False,
        dedup_iou: float = 0.90,
        dedup_merge: bool = False,
        dedup_class_agnostic: bool = False,
        input_color: str = "bgr",
        # what to save
        save_json: bool = True,
        include_raw: bool = False,   # include det_xyxy_conf_cls list in JSON
        save_overlay: bool = True,
        overlay_name: str = "overlay.png",
        save_crops: bool = False,
        crops_dirname: str = "crops",
        crops_meta_name: str = "crops.json",
        # crop options (used when save_crops=True)
        pad_px: int = 0,
        pad_pct: float = 0.0,
        expand_mode: str = "margin",
        context_gap_px: int = 0,
    ) -> Dict[str, Any]:
        """
        Save standard debugging artifacts to disk (opt-in).

        Outputs (if enabled):
        - detections.json
        - overlay.png
        - crops/*.png and crops.json

        Returns a dict of produced paths.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Normalize once
        im0 = load_image_bgr(image, input_color=input_color)

        # Get result if not provided
        if result is None:
            result = self.predict(
                im0,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                classes=classes,
                agnostic_nms=agnostic_nms,
                keep_classes=keep_classes,
                dedup=dedup,
                dedup_iou=dedup_iou,
                dedup_merge=dedup_merge,
                dedup_class_agnostic=dedup_class_agnostic,
                input_color="bgr",
            )

        produced: Dict[str, Any] = {}

        # 1) detections.json
        if save_json:
            det_json_path = out_dir / "detections.json"
            result.to_json(det_json_path, include_raw=include_raw)
            produced["detections_json"] = str(det_json_path)

        # 2) overlay image
        if save_overlay:
            try:
                import cv2  # type: ignore
            except Exception as e:
                raise ImportError("opencv-python is required to save overlay images.") from e

            overlay_bgr = self.visualize(im0, result, return_format="bgr", input_color="bgr")
            overlay_path = out_dir / overlay_name
            ok = cv2.imwrite(str(overlay_path), overlay_bgr)
            if not ok:
                raise RuntimeError(f"Failed to write overlay image: {overlay_path}")
            produced["overlay"] = str(overlay_path)

        # 3) crops
        if save_crops:
            try:
                import cv2  # type: ignore
            except Exception as e:
                raise ImportError("opencv-python is required to save crop images.") from e

            crops_dir = out_dir / crops_dirname
            crops_dir.mkdir(parents=True, exist_ok=True)

            crops = self.extract_crops(
                im0,
                result,
                pad_px=pad_px,
                pad_pct=pad_pct,
                expand_mode=expand_mode,
                context_gap_px=context_gap_px,
                return_format="bgr",
                input_color="bgr",
            )

            crops_meta: List[Dict[str, Any]] = []
            for idx, c in enumerate(crops):
                # safe filename
                cls_safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in c.class_name)
                stem = f"{idx:04d}_{cls_safe}_conf{c.conf:.3f}_id{c.class_id}"
                crop_path = crops_dir / f"{stem}.png"

                crop_bgr = c.image
                if crop_bgr is None:
                    continue
                ok = cv2.imwrite(str(crop_path), crop_bgr)
                if not ok:
                    raise RuntimeError(f"Failed to write crop image: {crop_path}")

                m = c.to_dict(meta_only=True)
                m["path"] = str(crop_path)
                crops_meta.append(m)

            meta_path = crops_dir / crops_meta_name
            meta_path.write_text(json.dumps(crops_meta, indent=2), encoding="utf-8")

            produced["crops_dir"] = str(crops_dir)
            produced["crops_meta"] = str(meta_path)
            produced["crops_count"] = len(crops_meta)

        return produced

    def visualize(
        self,
        image: Any,
        dets: Union[DetectionResult, np.ndarray, Sequence[Detection]],
        *,
        return_format: str = "bgr",  # bgr|rgb|pil
        input_color: str = "bgr",
        color_map: str = "",
        line_thickness: int = 2,
        hide_labels: bool = False,
        hide_conf: bool = False,
        min_font_scale: float = 0.35,
        max_font_scale: float = 0.90,
        label_max_width_ratio: float = 0.70,
        label_pad: int = 2,
        label_bg_alpha: float = 0.55,
        touch_tol: int = 0,
        label_gap_ratio_w: float = 0.05,
        label_gap_ratio_h: float = 0.90,
    ):
        """
        Draw detections on the image and return the visualization.
        """
        im0 = load_image_bgr(image, input_color=input_color)
        im_vis = im0.copy()

        det_arr = _dets_to_np(dets)

        render_detections(
            im_vis=im_vis,
            det_scaled=det_arr,
            names=self.names,
            color_map=str(color_map),
            line_thickness=int(line_thickness),
            hide_labels=bool(hide_labels),
            hide_conf=bool(hide_conf),
            min_font_scale=float(min_font_scale),
            max_font_scale=float(max_font_scale),
            label_max_width_ratio=float(label_max_width_ratio),
            label_pad=int(label_pad),
            label_bg_alpha=float(label_bg_alpha),
            touch_tol=int(touch_tol),
            label_gap_ratio_w=float(label_gap_ratio_w),
            label_gap_ratio_h=float(label_gap_ratio_h),
        )

        return _format_image(im_vis, return_format)

    def extract_crops(
        self,
        image: Any,
        dets: Union[DetectionResult, np.ndarray, Sequence[Detection]],
        *,
        pad_px: int = 0,
        pad_pct: float = 0.0,
        expand_mode: str = "margin",  # margin|context
        context_gap_px: int = 0,
        return_format: str = "bgr",  # bgr|rgb|pil
        input_color: str = "bgr",
        obstacles_xyxy: Optional[np.ndarray] = None,
    ) -> List[Crop]:
        """
        Extract cropped regions from detections.

        expand_mode="context":
            Uses nearby detections as obstacles to avoid overlapping while expanding.
            If obstacles_xyxy is None, uses the provided detections' boxes.
        """
        im0 = load_image_bgr(image, input_color=input_color)
        det_arr = _dets_to_np(dets)

        if expand_mode == "context" and obstacles_xyxy is None:
            obstacles_xyxy = det_arr[:, 0:4] if det_arr.size else None

        cfg = CropConfig(
            pad_px=int(pad_px),
            pad_pct=float(pad_pct),
            expand_mode=str(expand_mode),
            context_gap_px=int(context_gap_px),
        )

        crops_np = _extract_crops_np(
            im0_bgr=im0,
            det_xyxy_conf_cls=det_arr,
            names=self.names,
            cfg=cfg,
            obstacles_xyxy=obstacles_xyxy,
        )

        panel_id_by_det_index: Dict[int, Optional[int]] = {}
        if isinstance(dets, DetectionResult):
            for i, d in enumerate(dets.detections):
                panel_id_by_det_index[i] = getattr(d, "panel_id", None)
            
        out: List[Crop] = []
        for item in crops_np:
            crop_bgr = item["crop_bgr"]
            crop_img = _format_image(crop_bgr, return_format)
            det_index = int(item["det_index"])
            pid = panel_id_by_det_index.get(det_index, None)

            out.append(
                Crop(
                    image=crop_img,
                    xyxy=tuple(int(v) for v in item["box_xyxy"]),
                    conf=float(item["conf"]),
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    det_index=int(item["det_index"]),
                    panel_id=pid,
                )
            )
        return out

    def encode_image(self, *args, **kwargs):
        """
        Placeholder for a future embedding API.
        Panel detection models (YOLOv5) don't naturally expose a stable embedding.
        Use panel-cbir / CLIP models for embeddings instead.
        """
        raise NotImplementedError("encode_image is not available for detection models. Use panel-cbir for embeddings.")

    def runtime_info(self) -> Dict[str, Any]:
        """Small helper for debugging env/provider setup."""
        return {
            "model_onnx": self.model_onnx,
            "imgsz": self.imgsz,
            "names_source": str(self.names_source) if self.names_source is not None else None,
            "providers": getattr(self.predictor, "providers", None),
        }

    def warmup(self, *, n: int = 3, imgsz: Optional[int] = None) -> None:
        """
        Run a few dummy forwards to warm up ONNX runtime (especially GPU).
        """
        if n <= 0:
            return
        s = int(imgsz or getattr(self, "imgsz", 640))
        dummy = np.zeros((s, s, 3), dtype=np.uint8)
        for _ in range(int(n)):
            _ = self.predict(dummy)

    def profile(
        self,
        image: Any,
        *,
        n: int = 20,
        warmup_n: int = 3,
        input_color: str = "bgr",
    ) -> Dict[str, float]:
        """
        Simple performance profile on a single image.
        Returns avg/p50/p90 for total time (seconds).
        """
        if warmup_n > 0:
            self.warmup(n=warmup_n)

        times = []
        for _ in range(int(n)):
            _, t = self.predict(image, input_color=input_color, return_timing=True)
            times.append(float(t["total"]))

        arr = np.asarray(times, dtype=np.float64)
        return {
            "n": float(len(arr)),
            "avg": float(arr.mean()) if len(arr) else 0.0,
            "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
            "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
            "min": float(arr.min()) if len(arr) else 0.0,
            "max": float(arr.max()) if len(arr) else 0.0,
        }


def _default_ort_providers() -> List[str]:
    """
    Return a sensible default provider list:
    - Prefer CUDA if available, always fall back to CPU.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return ["CPUExecutionProvider"]

    avail = set(ort.get_available_providers())
    out: List[str] = []
    if "CUDAExecutionProvider" in avail:
        out.append("CUDAExecutionProvider")
    out.append("CPUExecutionProvider")
    return out


def _dets_to_np(dets: Union[DetectionResult, np.ndarray, Sequence[Detection]]) -> np.ndarray:
    if isinstance(dets, DetectionResult):
        arr = dets.det_xyxy_conf_cls
        return np.asarray(arr, dtype=np.float32).reshape((-1, 6)) if arr is not None else np.zeros((0, 6), dtype=np.float32)
    if isinstance(dets, np.ndarray):
        arr = np.asarray(dets, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ValueError(f"Expected det array Nx6, got shape {arr.shape}")
        return arr[:, :6]
    # list[Detection]
    out = []
    for d in dets:
        if not isinstance(d, Detection):
            raise TypeError(f"Expected Detection objects, got {type(d)}")
        x1, y1, x2, y2 = d.xyxy
        out.append([x1, y1, x2, y2, float(d.conf), float(d.class_id)])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 6), dtype=np.float32)


def _format_image(im_bgr: np.ndarray, return_format: str):
    rf = str(return_format).lower()
    if rf == "bgr":
        return im_bgr
    if rf == "rgb":
        return to_rgb(im_bgr)
    if rf == "pil":
        return to_pil(im_bgr)
    raise ValueError("return_format must be one of: 'bgr', 'rgb', 'pil'")
