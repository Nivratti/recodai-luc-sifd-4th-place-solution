from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception as e:  # pragma: no cover
    ort = None
    _ORT_IMPORT_ERROR = e
else:
    _ORT_IMPORT_ERROR = None

from .preprocess import bgr_to_tensor, letterbox
from .postprocess import postprocess_yolov5, scale_boxes_yolov5


@dataclass(frozen=True)
class OnnxPredictorConfig:
    imgsz: int = 640
    fp16: bool = False
    providers: Optional[List[str]] = None


class YoloOnnxPredictor:
    """YOLOv5-style ONNX inference: preprocess -> ort.run -> postprocess -> scale to original."""

    def __init__(self, model_path: str, cfg: OnnxPredictorConfig):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Install with: pip install onnxruntime"
            ) from _ORT_IMPORT_ERROR

        self.model_path = model_path
        self.cfg = cfg

        providers = cfg.providers or ort.get_available_providers()
        self.sess = ort.InferenceSession(model_path, providers=providers)

        inputs = self.sess.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")
        self.inp = inputs[0]
        self.inp_name = self.inp.name

        self._warned_batch_fallback = False

    # ---- runtime info helpers ----
    def get_providers(self) -> List[str]:
        try:
            return list(self.sess.get_providers())
        except Exception:
            return []

    def get_device(self) -> str:
        if ort is None:
            return "unknown"
        try:
            return str(ort.get_device())
        except Exception:
            return "unknown"

    # ---- batching helpers ----
    def _input_batch_dim(self):
        """
        Returns:
          - int (fixed batch)
          - None (dynamic/unknown)
          - other (string/symbolic) treated as dynamic
        """
        try:
            shp = self.inp.shape
            if not shp:
                return None
            return shp[0]
        except Exception:
            return None

    def supports_batch(self, batch_size: int) -> bool:
        if int(batch_size) <= 1:
            return True
        b0 = self._input_batch_dim()
        if b0 is None:
            return True
        if isinstance(b0, str):
            return True
        try:
            return int(b0) == int(batch_size) or int(b0) > 1
        except Exception:
            return True

    def _tensor_no_batch(self, x: np.ndarray) -> np.ndarray:
        """
        bgr_to_tensor may return (1,3,H,W) or (3,H,W).
        Normalize to (3,H,W) for stacking.
        """
        if x.ndim == 4 and x.shape[0] == 1:
            return x[0]
        return x

    def predict_bgr(
        self,
        im0_bgr: np.ndarray,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
    ) -> np.ndarray:
        """Returns Nx6 (xyxy, conf, cls) in ORIGINAL image coords."""
        out = self.predict_batch_bgr(
            [im0_bgr],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            classes=classes,
            agnostic_nms=agnostic_nms,
        )
        return out[0] if out else np.zeros((0, 6), dtype=np.float32)

    def predict_batch_bgr(
        self,
        ims0_bgr: List[np.ndarray],
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
    ) -> List[np.ndarray]:
        """
        Batched forward pass. Returns list of Nx6 (xyxy,conf,cls) per image.
        Falls back to per-image inference if batch is unsupported.
        """
        if not ims0_bgr:
            return []

        bs = len(ims0_bgr)
        if bs == 1:
            # keep original code path
            im0 = ims0_bgr[0]
            im0_shape = im0.shape[:2]

            im_lb, gain, pad = letterbox(im0, new_shape=int(self.cfg.imgsz))
            x = bgr_to_tensor(im_lb, fp16=self.cfg.fp16)

            outs = self.sess.run(None, {self.inp_name: x})
            if len(outs) == 0:
                return [np.zeros((0, 6), dtype=np.float32)]

            pred = outs[0]
            boxes_lb, scores, cls_id = postprocess_yolov5(
                pred=pred,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                classes=classes,
                agnostic_nms=agnostic_nms,
                nms_impl="opencv",
            )
            boxes = scale_boxes_yolov5(boxes_lb, im0_shape=im0_shape, pad=pad, gain=gain)
            if boxes.shape[0] == 0:
                return [np.zeros((0, 6), dtype=np.float32)]
            det = np.concatenate(
                [
                    boxes.astype(np.float32),
                    scores.reshape(-1, 1).astype(np.float32),
                    cls_id.reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            )
            return [det]

        # If model input is fixed batch=1, do per-image
        if not self.supports_batch(bs):
            if not self._warned_batch_fallback:
                print(f"[warn] model input appears to not support batch={bs}; falling back to per-image inference")
                self._warned_batch_fallback = True
            return [
                self.predict_bgr(im, conf_thres, iou_thres, max_det, classes=classes, agnostic_nms=agnostic_nms)
                for im in ims0_bgr
            ]

        # preprocess all -> stack
        im0_shapes = [im.shape[:2] for im in ims0_bgr]
        lbs = []
        gains = []
        pads = []
        for im in ims0_bgr:
            im_lb, gain, pad = letterbox(im, new_shape=int(self.cfg.imgsz))
            t = bgr_to_tensor(im_lb, fp16=self.cfg.fp16)
            lbs.append(self._tensor_no_batch(t))
            gains.append(gain)
            pads.append(pad)

        x = np.stack(lbs, axis=0)  # (bs,3,H,W)

        # forward
        try:
            outs = self.sess.run(None, {self.inp_name: x})
        except Exception as e:
            # runtime fallback if ORT/model rejects batch
            if not self._warned_batch_fallback:
                print(f"[warn] ORT batch inference failed; falling back to per-image. err={e}")
                self._warned_batch_fallback = True
            return [
                self.predict_bgr(im, conf_thres, iou_thres, max_det, classes=classes, agnostic_nms=agnostic_nms)
                for im in ims0_bgr
            ]

        if len(outs) == 0:
            return [np.zeros((0, 6), dtype=np.float32) for _ in range(bs)]

        pred = outs[0]

        # postprocess per image
        out_list: List[np.ndarray] = []
        for i in range(bs):
            # keep batch dim for compatibility (i:i+1)
            try:
                pred_i = pred[i : i + 1]
            except Exception:
                pred_i = pred

            boxes_lb, scores, cls_id = postprocess_yolov5(
                pred=pred_i,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                classes=classes,
                agnostic_nms=agnostic_nms,
                nms_impl="opencv",
            )
            boxes = scale_boxes_yolov5(boxes_lb, im0_shape=im0_shapes[i], pad=pads[i], gain=gains[i])
            if boxes.shape[0] == 0:
                out_list.append(np.zeros((0, 6), dtype=np.float32))
                continue

            det = np.concatenate(
                [
                    boxes.astype(np.float32),
                    scores.reshape(-1, 1).astype(np.float32),
                    cls_id.reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            )
            out_list.append(det)

        return out_list
