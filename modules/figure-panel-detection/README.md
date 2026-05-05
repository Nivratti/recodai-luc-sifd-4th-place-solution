# Figure Panel Detection

Detect and extract **image panels** from **scientific figures** (compound figures, multi-panel layouts, supplemental figures, etc.).

This repo supports **two ways of using the detector**:

- **Python API (importable, in-memory)**: `FigurePanelDetector` (best for orchestration across repos)
- **CLI / disk pipeline**: `panel-detect` / `scripts/panel_detect.py` (best for batch jobs that write labels/crops/visualizations to disk)

📚 Detailed docs:

- **Python API Reference**: [`docs/PYTHON_API.md`](docs/PYTHON_API.md)
- **CLI Reference**: [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md)

**Phase 1 (current focus):**
- ✅ **Microscopy panels**
- ✅ **Blot / gel panels** (e.g., western blots)

**Phase 2 (planned):**
- ⏳ Flow cytometry plots
- ⏳ Graphs / charts (bar/line/scatter, etc.)
- ⏳ Better compound-figure layout parsing (panel grids, montages, insets)

---

## What this repo does

Given an input scientific figure (often a “compound figure” containing many sub-images), this project can:
1. **Detect panel bounding boxes**
2. **Extract crops** (either in-memory via the API, or on-disk via the CLI)
3. **Render visual overlays** (either in-memory or on-disk)
4. Export results in common formats (YOLO/COCO/CSV)

Designed for downstream tasks such as:
- scientific image forensics (duplication / reuse detection / manipulation localization)
- dataset creation (panel-level datasets)
- figure understanding / layout analysis pipelines

---

## Outputs (typical)

For each input image you can produce:

- `*.json` (COCO-style detections) or `*.txt` (YOLO labels)
- `*.csv` summary (one row per detected panel)
- `*_viz.png` overlay visualization (boxes + labels)
- `/crops/` directory with panel crops (optional)

---

## YOLO (ONNX) inference

This repo runs **YOLOv5-style ONNX inference** via **ONNX Runtime**.

### Model files (required)
The detector expects an ONNX model and a sidecar JSON **next to it**:

- `model_4_class.onnx`
- `model_4_class.json` (same stem)

Example `model_4_class.json`:

```json
{
  "class_names": {
    "0": "Blots",
    "1": "Microscopy",
    "2": "Graphs",
    "3": "Flow Cytometry"
  }
}
```

> The loader auto-reads `<model_stem>.json`. If it is missing or malformed, the CLI will exit with an error.

---

## Install (recommended)

Editable install is recommended (imports work anywhere + CLI command available):

```bash
pip install -e .
```

Optional (only if you want PIL inputs/outputs in the API):

```bash
pip install pillow
```

---

## Python API (importable, in-memory)

The public API is exposed at the package root:

```python
from figure_panel_detection import FigurePanelDetector
```

### Basic usage

```python
from figure_panel_detection import FigurePanelDetector

det = FigurePanelDetector(
    model_onnx="resources/models/model_4_class.onnx",
)  # providers auto-selected (CUDA if available, else CPU)

res = det.predict(
    "/path/to/figure.png",
    conf_thres=0.4,
    iou_thres=0.4,
    keep_classes=["Blots", "Microscopy"],  # optional; names or ids
)

for d in res.detections:
    print(d.class_name, f"{d.conf:.3f}", d.xyxy)  # xyxy are int pixels in original image
```

### Input types

`predict(...)` accepts:

- `str` / `Path` (image path)
- `np.ndarray` (H×W×C)
- `PIL.Image.Image`

Notes:

- If you pass a **numpy RGB** image (common from PIL/matplotlib), set `input_color="rgb"`.
- If you pass an OpenCV image (already BGR), the default `input_color="bgr"` is correct.

```python
import numpy as np
from PIL import Image

rgb = np.zeros((256, 256, 3), dtype=np.uint8)
res1 = det.predict(rgb, input_color="rgb")

pil = Image.open("/path/to/figure.png")
res2 = det.predict(pil)
```

### Visualize (returns an image)

`visualize(...)` draws detections and returns an image (does not write to disk):

```python
vis_bgr = det.visualize("/path/to/figure.png", res, return_format="bgr")
vis_rgb = det.visualize("/path/to/figure.png", res, return_format="rgb")
vis_pil = det.visualize("/path/to/figure.png", res, return_format="pil")  # requires pillow
```

### Extract crops (in-memory)

`extract_crops(...)` returns a list of `Crop` objects. Each crop contains the cropped image plus metadata.

```python
crops = det.extract_crops(
    "/path/to/figure.png",
    res,
    pad_px=8,
    pad_pct=0.05,
    expand_mode="margin",  # or "context" (uses nearby boxes to avoid overlaps)
    return_format="pil",
)

print(len(crops))
print(crops[0].class_name, crops[0].conf, crops[0].xyxy)
crop_img = crops[0].image
```

### Predict + crops in one call

`predict_crops(...)` runs detection and returns crops in-memory in a single call.

```python
res, crops = det.predict_crops(
    "/path/to/figure.png",
    keep_classes=["Blots"],
    conf_thres=0.4,
    iou_thres=0.4,
    pad_pct=0.05,
    return_format="pil",
)

print(len(res.detections), len(crops))
print(crops[0].class_name, crops[0].conf, crops[0].xyxy)
```

### Batch predict

```python
results = det.predict_batch(
    ["/path/a.png", "/path/b.png", "/path/c.png"],
    conf_thres=0.4,
)
```

### Runtime info

```python
info = det.runtime_info()
print(info["providers"], info["model_path"])
```

### Serialize results (dict/JSON)

```python
res = det.predict("/path/to/figure.png")

# Save (metadata only by default)
res.to_json("detections.json")

# Load later
res2 = DetectionResult.from_json("detections.json")
print(len(res2.detections))
```

### Save debugging artifacts (optional)

```python
from figure_panel_detection import FigurePanelDetector

det = FigurePanelDetector(model_onnx=".../model.onnx")
paths = det.save_artifacts(
    out_dir="runs/debug/15257",
    image="/path/to/15257.png",
    keep_classes=["Blots"],
    save_overlay=True,
    save_crops=True,
    pad_pct=0.05,
)

print(paths)
```

### Filtering + stable ordering

```python
res = det.predict(
    img_path,
    keep_classes=["Blots", "Microscopy"],
    min_area_frac=0.0005,      # remove tiny noise
    max_area_frac=0.40,        # remove giant background boxes
    topk_per_class=50,         # cap per class
    sort="yx",                 # stable panel ordering (top->bottom, left->right)
)
```

### Box conversions and coordinate mapping

```python
from figure_panel_detection.api import crop_to_image_xyxy, xyxy_to_xywh

# detector boxes
xywh = xyxy_to_xywh(res.boxes_xyxy())

# map crop-local prediction back to original image
img_box = crop_to_image_xyxy(crop_xyxy=[10, 20, 110, 120], crop_offset_xyxy=crops[0].xyxy)
```

### Tiled prediction for large compound figures

```python
res = det.predict_tiled(
    img_path,
    tile=1024,
    overlap=0.2,
    merge_iou=0.6,
    keep_classes=["Blots", "Microscopy"],
    sort="yx",
)
```

### Timing and profiling

```python
res, timing = det.predict(img_path, return_timing=True)
print(timing)  # load_image / infer / post / total

det.warmup(n=3)
stats = det.profile(img_path, n=20)
print(stats)   # avg/p50/p90 total seconds
```

### Stable panel IDs

```python
res = det.predict(img_path, sort="yx")
res = res.assign_ids(order="yx")

for d in res.detections[:5]:
    print(d.panel_id, d.class_name, d.xyxy)
```

Or:

```python
res = det.predict(img_path, assign_panel_ids=True, panel_id_order="yx")
```
---

## CLI usage (disk pipeline)

### Quickstart (local testing without install)

Under `scripts/` we include `panel_detect.py` which prepends `src/` to `sys.path` so imports work.

```bash
python scripts/panel_detect.py   --model "resources/models/model_4_class.onnx"   --source "./resources/compound-figures"   --out "./runs/out/v1"   --save-vis-img
```

### Common flags

**Split outputs by detection bucket**

```bash
--split-by-detections
```

**Batch layout (useful when sorting / large jobs)**

```bash
--layout batch --layout-batch-size 100 --sort-by-objects
```

**Keep only certain classes (by name)**

```bash
--keep-classes "Blots" "Microscopy"
```

**Backup original full labels when keep-filtering is applied**

```bash
--backup-original-labels
```

**Copy input images into the output folder**

```bash
--copy-images
```

### Recommended usage (editable install)

```bash
pip install -e .
```

Then you can run:

```bash
panel-detect   --model "resources/models/model_4_class.onnx"   --source "./resources/compound-figures"   --out "./runs/out/v1"   --save-vis-img
```

> `pip install -e .` is recommended for reproducible environments and when running tools from outside the repo root.

---

## Outputs layout

By default, YOLO labels are written under:

- `runs/out/<run>/labels/**/*.txt`

If `--save-vis-img` is enabled:

- `runs/out/<run>/vis/**/*.<ext>`

If `--copy-images` is enabled:

- `runs/out/<run>/images/**/*.<ext>`

If `--split-by-detections` is enabled, outputs go into bucket subfolders:

- `runs/out/<run>/kept/...`
- `runs/out/<run>/ignored/...`
- `runs/out/<run>/no_objects/...`

Note: when `--split-by-detections` is on, visualization images are **not saved** for `no_objects`.

---

## Crops (CLI)

Enable saving cropped regions:

```bash
--save-crop
```

### Crop modes

**1) Default: class mode**  
Saves under:

- `crops/<class_name>/...`

Preserves batch folders in batch layout.

```bash
--save-crop --crop-mode class
```

**2) Image mode**  
Creates a folder per input image and saves each crop inside it.  
Also preserves batch folders when `--layout batch`.

```bash
--save-crop --crop-mode image
```

### Include ignored bucket crops (only when using keep-classes)

```bash
--crop-include-ignored
```

### Expand crop region

Two expansion mechanisms are supported (can be combined):

- fixed pixels: `--crop-pad-px 10`
- percentage of box size: `--crop-pad-pct 0.10` (10%)

```bash
--save-crop --crop-pad-px 10 --crop-pad-pct 0.05
```

### Crop image format

Default crop format is **PNG**. To save JPEG:

```bash
--save-crop --crop-ext jpg --crop-jpg-quality 95 --crop-jpg-subsampling 444
```

Crops write a provenance file:

- `crops/**/mapping.json` (contains crop path, source label/image, class, confidence, box coords, etc.)

---

## GPU vs CPU (ONNX Runtime)

Inference runs on **GPU only if ONNX Runtime has a GPU provider** available.

Check providers:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Force GPU (if available):

```bash
--providers CUDAExecutionProvider CPUExecutionProvider
```

Force CPU:

```bash
--providers CPUExecutionProvider
```

## Tests

Run the full API test suite (includes real ONNX inference):

```bash
pytest
```

The suite expects a YOLO ONNX model at:

- `resources/models/yolov5-onnx/model_4_class.onnx`

Or point to a model explicitly:

```bash
export FIGURE_PANEL_DET_ONNX=/abs/path/to/model_4_class.onnx
pytest
```

To run unit tests only (skip the real-model check):

```bash
export FIGURE_PANEL_DET_SKIP_INTEGRATION=1
pytest
```

See `tests/README.md` for details.