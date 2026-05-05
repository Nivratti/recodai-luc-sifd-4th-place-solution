import os
import pytest

def test_environment_has_model(onnx_model_path):
    """Fail fast if a real ONNX model is not available.

    This gives you an early signal that the detector cannot run end-to-end.
    To skip this check (unit tests only), set:

        FIGURE_PANEL_DET_SKIP_INTEGRATION=1
    """
    if os.environ.get("FIGURE_PANEL_DET_SKIP_INTEGRATION", "").strip() in ("1","true","yes","on"):
        pytest.skip("Integration/model check skipped via FIGURE_PANEL_DET_SKIP_INTEGRATION=1")

    if onnx_model_path is None:
        pytest.fail(
            "No ONNX model found for integration tests. "
            "Place it at resources/models/yolov5-onnx/model_4_class.onnx "
            "or set FIGURE_PANEL_DET_ONNX=/abs/path/to/model_4_class.onnx. "
            "To run unit tests only, set FIGURE_PANEL_DET_SKIP_INTEGRATION=1."
        )
