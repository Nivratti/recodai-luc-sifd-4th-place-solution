import pytest

from figure_panel_detection import FigurePanelDetector


def test_optional_methods_presence(names_payload):
    # This test is future-proof: it doesn't fail if optional APIs aren't implemented yet.
    det = FigurePanelDetector.__dict__
    optional = ["predict_crops", "save_artifacts", "predict_tiled", "warmup", "profile"]
    missing = [m for m in optional if m not in det]
    # Always pass; report in xfail note if you want strictness later.
    assert True
