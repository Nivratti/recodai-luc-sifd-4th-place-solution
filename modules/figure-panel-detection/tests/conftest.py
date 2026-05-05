import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _default_model_candidates(root: Path):
    # Common local locations (repo-relative)
    yield root / "resources" / "models" / "yolov5-onnx" / "model_4_class.onnx"
    yield root / "resources" / "models" / "yolov5-onnx" / "model_5_class.onnx"


@pytest.fixture(scope="session")
def sample_images():
    data_dir = ROOT / "tests" / "data"
    return {
        "img1": data_dir / "13977.png",
        "img2": data_dir / "15257.png",
    }


@pytest.fixture(scope="session")
def onnx_model_path():
    """Return a usable ONNX model path for integration tests.

    Resolution order:
      1) FIGURE_PANEL_DET_ONNX env var (absolute or relative path)
      2) common repo-relative paths under resources/models/

    If no model is found, returns None (integration tests will skip).
    """
    env = os.environ.get("FIGURE_PANEL_DET_ONNX", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.exists():
            return p
        pytest.fail(f"FIGURE_PANEL_DET_ONNX was set but file does not exist: {p}")

    for cand in _default_model_candidates(ROOT):
        if cand.exists():
            return cand

    return None


@pytest.fixture(scope="session")
def names_payload():
    # Minimal class map used by dummy predictor/unit tests
    return {0: "Blots", 1: "Microscopy", 2: "Charts", 3: "Other"}
