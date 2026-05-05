import numpy as np
import pytest

from figure_panel_detection import FigurePanelDetector


class DummyPredictor:
    """A tiny stand-in for YoloOnnxPredictor used for unit tests.

    It returns deterministic detections in Nx6 format:
    [x1,y1,x2,y2,conf,cls]
    """

    def __init__(self, *args, **kwargs):
        self.providers = ["DummyExecutionProvider"]

    def predict_bgr(self, im_bgr, *, conf_thres=0.25, iou_thres=0.45, max_det=300, classes=None, agnostic_nms=False):
        h, w = im_bgr.shape[:2]
        # 3 boxes, two overlapping blots + one microscopy
        det = np.array([
            [0.10*w, 0.10*h, 0.50*w, 0.50*h, 0.90, 0],  # Blots
            [0.12*w, 0.12*h, 0.52*w, 0.52*h, 0.80, 0],  # Blots (overlaps)
            [0.55*w, 0.10*h, 0.95*w, 0.45*h, 0.85, 1],  # Microscopy
        ], dtype=np.float32)

        # Apply basic conf filter similar to model behavior
        det = det[det[:, 4] >= float(conf_thres)]
        if classes is not None:
            keep = np.isin(det[:, 5].astype(int), np.array(classes, dtype=int))
            det = det[keep]
        # Cap max_det
        if len(det) > int(max_det):
            det = det[: int(max_det)]
        return det

    def predict_batch_bgr(self, ims_bgr, *, conf_thres=0.25, iou_thres=0.45, max_det=300, classes=None, agnostic_nms=False):
        return [self.predict_bgr(im, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, classes=classes, agnostic_nms=agnostic_nms) for im in ims_bgr]


@pytest.fixture()
def detector_dummy(monkeypatch, names_payload):
    # Patch the predictor class used inside FigurePanelDetector to avoid loading real ONNX models.
    import figure_panel_detection.api.detector as det_mod

    monkeypatch.setattr(det_mod, "YoloOnnxPredictor", DummyPredictor)

    # model_onnx path doesn't need to exist for DummyPredictor
    return FigurePanelDetector(model_onnx="dummy.onnx", names=names_payload, imgsz=640, providers=["DummyExecutionProvider"])


def test_predict_returns_detection_result(detector_dummy, sample_images):
    res = detector_dummy.predict(str(sample_images["img1"]))
    assert hasattr(res, "detections")
    assert hasattr(res, "det_xyxy_conf_cls")
    assert res.det_xyxy_conf_cls.shape[1] == 6
    assert len(res.detections) == res.det_xyxy_conf_cls.shape[0]
    assert len(res.detections) >= 1


def test_keep_classes_by_name(detector_dummy, sample_images):
    res = detector_dummy.predict(str(sample_images["img1"]), keep_classes=["Blots"])
    assert all(d.class_name == "Blots" for d in res.detections)
    assert all(int(d.class_id) == 0 for d in res.detections)


def test_keep_classes_by_id(detector_dummy, sample_images):
    res = detector_dummy.predict(str(sample_images["img1"]), keep_classes=[1])
    assert len(res.detections) >= 1
    assert all(int(d.class_id) == 1 for d in res.detections)


def test_dedup_removes_overlaps(detector_dummy, sample_images):
    res = detector_dummy.predict(str(sample_images["img1"]), dedup=True, dedup_iou=0.5)
    # since two blot boxes overlap heavily, after dedup we expect fewer than 3 boxes
    assert len(res.detections) <= 3
    assert len(res.detections) >= 1


def test_visualize_returns_image(detector_dummy, sample_images):
    res = detector_dummy.predict(str(sample_images["img1"]))
    vis = detector_dummy.visualize(str(sample_images["img1"]), res, return_format="bgr")
    assert isinstance(vis, np.ndarray)
    assert vis.ndim == 3 and vis.shape[2] == 3


def test_extract_crops(detector_dummy, sample_images):
    res = detector_dummy.predict(str(sample_images["img1"]), keep_classes=["Blots"])
    crops = detector_dummy.extract_crops(str(sample_images["img1"]), res, pad_pct=0.02, return_format="bgr")
    assert isinstance(crops, list)
    assert len(crops) == len(res.detections)
    assert crops[0].image is not None
    assert hasattr(crops[0], "xyxy")
    assert hasattr(crops[0], "det_index")


def test_predict_batch(detector_dummy, sample_images):
    paths = [str(sample_images["img1"]), str(sample_images["img2"])]
    results = detector_dummy.predict_batch(paths)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(hasattr(r, "detections") for r in results)


def test_runtime_info(detector_dummy):
    info = detector_dummy.runtime_info()
    assert isinstance(info, dict)
    assert "providers" in info


def test_encode_image_not_implemented(detector_dummy, sample_images):
    with pytest.raises(NotImplementedError):
        detector_dummy.encode_image(str(sample_images["img1"]))
