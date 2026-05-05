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

The main runner is:

```text
scripts/main_runner.py
````

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
```

## Setup

Python 3.11 is recommended.

Create and activate a conda environment:

```bash
conda create -n recodai-sifd-public python=3.11 -y
conda activate recodai-sifd-public
```

Install dependencies:

```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Install this repository and local modules:

```bash
pip install -e .
pip install -e modules/figure-panel-detection
pip install -e modules/copy-move-det-keypoint
pip install -e modules/panel-cbir
```

## Model weights

The panel detector uses an ONNX model released separately in GitHub Releases:

[https://github.com/Nivratti/recodai-luc-sifd-4th-place-solution/releases/tag/panel-detector-v1.0](https://github.com/Nivratti/recodai-luc-sifd-4th-place-solution/releases/tag/panel-detector-v1.0)

The runner automatically downloads these files if they are missing locally:

```text
models/panel_detector/model_4_class.onnx
models/panel_detector/model_4_class.json
```

The first run may also download the CBIR backbone weights used by `timm`.

Internet access is required for the first run unless the files are already cached or downloaded.

## Quick smoke test

Run this from the repository root:

```bash
python scripts/main_runner.py \
  --input resources/samples \
  --out runs/smoke \
  --max-images 1 \
  --reuse-prune \
  --reuse-cbir-topk 3 \
  --reuse-cbir-batch-size 16 \
  --reuse-cbir-device cpu
```

Expected output:

```text
runs/smoke/submission.csv
```

## Run on a folder of images

```bash
python scripts/main_runner.py \
  --input path/to/images \
  --out runs/my_run \
  --reuse-prune \
  --reuse-cbir-topk 3 \
  --reuse-cbir-batch-size 16 \
  --reuse-cbir-device cpu
```

For GPU, use:

```bash
--reuse-cbir-device cuda
```

or:

```bash
--reuse-cbir-device cuda:0
```

## Important output files

The main competition-style output is:

```text
<out>/submission.csv
```

Example:

```text
runs/smoke/submission.csv
```

## Exact package versions

`requirements.txt` contains the recommended public runtime dependencies.

`requirements-freeze.txt` records the exact package versions from the local environment used for the public smoke test.

## Attribution

This repository includes modified versions of the following upstream projects.

### figure-panel-detection

Based on ResearchIntegrity `panel-extractor`:

[https://github.com/researchintegrity/panel-extractor](https://github.com/researchintegrity/panel-extractor)

Main modifications include:

* ONNX Runtime inference support
* Python API wrappers
* Integration changes for this Kaggle solution pipeline
* Output formatting and crop handling changes

See:

```text
modules/figure-panel-detection/NOTICE.md
```

### copy-move-det-keypoint

Based on ResearchIntegrity `copy-move-detection-keypoint`:

[https://github.com/researchintegrity/copy-move-detection-keypoint](https://github.com/researchintegrity/copy-move-detection-keypoint)

Main modifications include:

* Python API for programmatic use
* Cross-panel / pairwise matching integration
* Prepared/cached feature workflows
* Return objects and mask outputs used by this solution pipeline

See:

```text
modules/copy-move-det-keypoint/NOTICE.md
```