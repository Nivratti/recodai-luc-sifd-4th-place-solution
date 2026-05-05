from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PANEL_DETECTOR_DIR = Path("models/panel_detector")

PANEL_DETECTOR_FILES = {
    "model_4_class.onnx": "https://github.com/Nivratti/recodai-luc-sifd-4th-place-solution/releases/download/panel-detector-v1.0/model_4_class.onnx",
    "model_4_class.json": "https://github.com/Nivratti/recodai-luc-sifd-4th-place-solution/releases/download/panel-detector-v1.0/model_4_class.json",
}


def ensure_panel_detector_model(model_path: str | Path | None = None) -> Path:
    """
    Ensure the panel detector ONNX model and class JSON exist locally.

    Missing files are downloaded from the public GitHub release.

    Parameters
    ----------
    model_path:
        Optional path to the ONNX model. If None, the default repository
        location is used.

    Returns
    -------
    Path
        Local path to the ONNX panel detector model.
    """
    if model_path is None:
        model_path = PANEL_DETECTOR_DIR / "model_4_class.onnx"

    model_path = Path(model_path)
    model_dir = model_path.parent
    json_path = model_path.with_suffix(".json")

    expected_files = {
        model_path: PANEL_DETECTOR_FILES["model_4_class.onnx"],
        json_path: PANEL_DETECTOR_FILES["model_4_class.json"],
    }

    missing = [path for path in expected_files if not path.exists()]
    if not missing:
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)

    print("Panel detector model files not found locally. Downloading from GitHub release...")

    for path in missing:
        url = expected_files[path]
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        print(f"Downloading {path.name}...")
        try:
            urlretrieve(url, tmp_path)
            tmp_path.replace(path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()

            raise RuntimeError(
                "Failed to download panel detector model files.\n"
                f"File: {path.name}\n"
                f"URL: {url}\n\n"
                "Please check your internet connection or download the files manually "
                "from the GitHub release."
            ) from exc

    print(f"Panel detector model ready: {model_path}")
    return model_path