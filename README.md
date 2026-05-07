# Recod.ai LUC SIFD — 4th Place Kaggle Solution

This repository contains a cleaned public version of my 4th place solution for the Kaggle competition:

**Recod.ai LUC Scientific Image Forgery Detection**

The code was cleaned from the exact Kaggle submission build to make it easier to install, and run.

## What this solution does

The pipeline performs:

1. Scientific figure panel detection
2. Panel cropping and preprocessing
3. Candidate pruning using CBIR
4. Copy-move / reuse detection between panels
5. Reuse group construction
6. Competition-format `submission.csv` generation

The main runner is `scripts/main_runner.py`.

## Repository structure

```text
.
├── modules/
│   ├── figure-panel-detection/
│   ├── copy-move-det-keypoint/
│   └── panel-cbir/
├── resources/
│   └── samples/
├── scripts/
│   └── main_runner.py
├── src/
│   └── recodai_sifd/
├── requirements.txt
└── requirements-freeze.txt
└── pyproject.toml
```

## Setup

Python 3.11 is recommended. Create and activate an environment:

```bash
# conda
conda create -n recodai-sifd python=3.11 -y
conda activate recodai-sifd

# or plain venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Then install — pick **one** depending on your hardware:

**CPU**
```bash
pip install -e ".[all,cpu]"
```

**GPU (CUDA)**
```bash
pip install -e ".[all,gpu]"
```

This installs all dependencies and all local modules in one step.

## Model weights

The panel detector ONNX model is downloaded automatically on the first run from the [GitHub release](https://github.com/Nivratti/recodai-luc-sifd-4th-place-solution/releases/tag/panel-detector-v1.0). The CBIR backbone weights (`timm`/`resnet50`) are also downloaded on first run.

Internet access is required for the first run.

## Quick start

Run on the bundled sample image — no flags needed:

```bash
python scripts/main_runner.py
```

Expected output:

```text
runs/sifd/v1/submission.csv
```

## Run on your own images

```bash
python scripts/main_runner.py --input path/to/images --out runs/my_run
```

## GPU run

```bash
python scripts/main_runner.py \
  --input path/to/images \
  --out runs/my_run \
  --reuse-cbir-device cuda \
  --reuse-cbir-fp16 \
  --reuse-cbir-score-fp16
```

## Key options

| Flag                        | Default               | Description                                      |
| --------------------------- | --------------------- | ------------------------------------------------ |
| `--input`                 | `resources/samples` | Image file or folder                             |
| `--out`                   | `runs/sifd/v1`      | Output folder                                    |
| `--reuse-cbir-device`     | `cpu`               | Device for CBIR (`cpu`, `cuda`, `cuda:0`)  |
| `--reuse-cbir-topk`       | `12`                | Top-K candidates per panel from CBIR             |
| `--reuse-cbir-batch-size` | `64`                | CBIR batch size                                  |
| `--reuse-cbir-fp16`       | `False`             | fp16 embeddings — enable for GPU                |
| `--reuse-prune`           | `True`              | Candidate pruning (CBIR + geometry + grouping)   |
| `--max-images`            | `None`              | Process only first N images (useful for testing) |
| `--debug`                 | `False`             | Verbose logs and extra artifacts                 |

Run `python scripts/main_runner.py --help` for the full list.

## Documentation

| Document                                                                                | Description                                                      |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [WRITEUP.md](WRITEUP.md)                                                                   | Full solution writeup — pipeline design, decisions, and results |
| [figure-panel-detection](modules/figure-panel-detection/README.md)                         | Panel detector module — Python API and CLI reference            |
| [copy-move-det-keypoint](modules/copy-move-det-keypoint/README.md)                         | Copy-move detection module — usage and configuration            |
| [copy-move-det-keypoint Python API](modules/copy-move-det-keypoint/docs/PYTHON_API.md)     | Detailed Python API reference for the copy-move module           |
| [panel-cbir](modules/panel-cbir/README.md)                                                 | CBIR module — embedding, indexing, and ranking                  |
| [YOLOv5 ONNX model](modules/figure-panel-detection/resources/models/yolov5-onnx/README.md) | Panel detector model details and class definitions               |

## Attribution

This repository includes modified versions of the following upstream projects.

### figure-panel-detection

Based on ResearchIntegrity `panel-extractor`:
[https://github.com/researchintegrity/panel-extractor](https://github.com/researchintegrity/panel-extractor)

Modifications: ONNX Runtime inference, Python API wrappers, pipeline integration, output formatting. See `modules/figure-panel-detection/NOTICE.md`.

### copy-move-det-keypoint

Based on ResearchIntegrity `copy-move-detection-keypoint`:
[https://github.com/researchintegrity/copy-move-detection-keypoint](https://github.com/researchintegrity/copy-move-detection-keypoint)

Modifications: Python API, cross-panel pairwise matching, prepared/cached feature workflows, mask outputs. See `modules/copy-move-det-keypoint/NOTICE.md`.
