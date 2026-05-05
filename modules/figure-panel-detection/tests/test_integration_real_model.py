import pytest
import numpy as np

from figure_panel_detection import FigurePanelDetector


@pytest.mark.integration
def test_integration_predict_visualize_crops(onnx_model_path, sample_images):
    if onnx_model_path is None:
        pytest.skip("No ONNX model found. Set FIGURE_PANEL_DET_ONNX or place model under resources/models/yolov5-onnx/")

    # Try to load names sidecar if present; if not, tests will fail with clear error
    det = FigurePanelDetector(model_onnx=str(onnx_model_path))

    res = det.predict(str(sample_images["img2"]))  # complex compound figure
    assert res.det_xyxy_conf_cls.shape[1] == 6
    assert len(res.detections) == res.det_xyxy_conf_cls.shape[0]

    # visualize
    vis = det.visualize(str(sample_images["img2"]), res, return_format="bgr")
    assert isinstance(vis, np.ndarray)
    assert vis.ndim == 3 and vis.shape[2] == 3

    # crops
    crops = det.extract_crops(str(sample_images["img2"]), res, pad_pct=0.02, return_format="bgr")
    assert isinstance(crops, list)
    assert len(crops) == len(res.detections)

    # keep_classes should not crash (use first class name if available)
    if len(res.detections) > 0:
        cname = res.detections[0].class_name
        _ = det.predict(str(sample_images["img2"]), keep_classes=[cname])


@pytest.mark.integration
def test_integration_predict_batch(onnx_model_path, sample_images):
    if onnx_model_path is None:
        pytest.skip("No ONNX model found. Set FIGURE_PANEL_DET_ONNX or place model under resources/models/yolov5-onnx/")

    det = FigurePanelDetector(model_onnx=str(onnx_model_path))
    results = det.predict_batch([str(sample_images["img1"]), str(sample_images["img2"])])
    assert len(results) == 2
    assert all(r.det_xyxy_conf_cls.shape[1] == 6 for r in results)
