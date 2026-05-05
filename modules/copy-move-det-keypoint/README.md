# Copy‑Move Detection (Keypoint‑based)

Keypoint-based copy‑move / overlap detection using SIFT‑style local features + geometric verification.

This repo supports:

- **Single‑image copy‑move detection** (detect duplicated regions within one image)
- **Cross‑image overlap / reuse detection** (detect duplicated content between two images)

✅ **New (Public Python API):** `copy_move_det_keypoint.api`
Returns **numpy arrays** (masks, matched keypoints, metrics), and optionally writes output files only when `save_dir` is provided.

---

## Install

Create a fresh environment and install dependencies.

Minimum requirements (typical):

- Python 3.9+
- `numpy`
- `opencv-contrib-python` (for SIFT)
- `pillow`
- `scipy`
- `scikit-learn`

Example:

```bash
pip install numpy opencv-contrib-python pillow scipy scikit-learn
```

Optional (only if you want the VLFeat descriptor option):

- `cyvlfeat` (usually easiest via conda)

---

## Quickstart: Python API

Input types: `str|Path`, `np.ndarray`, `PIL.Image.Image` (see `docs/PYTHON_API.md` for details).

### Detailed API docs

For prepared/cached features (one-to-many / all-pairs optimization) and pruning APIs, see:

- `docs/PYTHON_API.md`

### 1) Make the package importable

This repo uses a `src/` layout. Either install it (if you have packaging in your environment), or simply add `src/` to `PYTHONPATH`:

```bash
# from repo root
export PYTHONPATH="$PWD/src"
```

### 2) Run detection

```python
from copy_move_det_keypoint import detect, DetectorConfig
from copy_move_det_keypoint import DescriptorType, AlignmentStrategy, MatchingMethod

cfg = DetectorConfig(
    descriptor_type=DescriptorType.CV_RSIFT,
    alignment_strategy=AlignmentStrategy.CV_MAGSAC,
    matching_method=MatchingMethod.BF,
    check_flip=True,
    min_keypoints=20,
    min_area=0.01,
)

# Cross-image detection (overlap / reuse)
res = detect("a.png", "b.png", config=cfg)

print("is_match:", res.is_match)
print("matched_keypoints:", res.matched_keypoints)
print("shared_area_source:", res.shared_area_source)
print("shared_area_target:", res.shared_area_target)

maskA = res.mask_source          # uint8 (0/1), shape (H, W)
maskB = res.mask_target          # uint8 (0/1), shape (H, W)
kptsA = res.matched_kpts_source  # float32, shape (N, 2) -> (x, y)
kptsB = res.matched_kpts_target  # float32, shape (N, 2) -> (x, y)
```

### Single‑image mode

```python
res = detect("image.png")  # target_path=None -> single-image mode
print(res.is_match, res.matched_keypoints)
```

### Optional: save outputs (paths returned only if `save_dir` is provided)

```python
res = detect("a.png", "b.png", config=cfg, save_dir="runs/out/v1")
print(res.outputs)  # dict of file paths
```

`outputs` keys (cross-image) typically include:

- `mask_path` (combined)
- `mask_source_path`, `mask_target_path`
- `matches_path`, `clusters_path`

Single-image outputs typically include:

- `mask_path`
- `matches_path`, `clusters_path`

---

## API Reference

### Prepared / cached features (fast multi-matching)

If you need **one-to-many** or **all-pairs** matching, use the prepared/cached feature APIs:

- `FeatureSet`
- `prepare(...)` (extract once, reuse many times)
- `match_prepared(...)` (match using cached features)
- optional pruning: `match_keypoints_only(...)` + `build_masks_from_matches(...)`

See: [docs/PYTHON_API.md](docs/PYTHON_API.md)

### `DetectorConfig`

Public configuration (defaults mirror the existing CLI / detector behavior):

- `descriptor_type` (e.g., `DescriptorType.CV_RSIFT`)
- `alignment_strategy` (e.g., `AlignmentStrategy.CV_MAGSAC`)
- `matching_method` (e.g., `MatchingMethod.BF`)
- `check_flip` (bool)
- `min_keypoints` (int)
- `min_area` (float)
- `timeout` (kept for parity; currently not enforced by the core algorithm)
- `cross_kp_count` (int)
- `single_kp_count` (int)

### `DetectionResult`

Returned by `detect(...)`:

- `is_match: bool`
- `mask_source: np.ndarray` (uint8 0/1, H×W)
- `mask_target: np.ndarray` (uint8 0/1, H×W)
- `matched_kpts_source: np.ndarray` (float32 N×2, (x,y))
- `matched_kpts_target: np.ndarray` (float32 N×2, (x,y))
- `shared_area_source: float`
- `shared_area_target: float`
- `is_flipped: bool`
- `matched_keypoints: int`
- `outputs: Optional[dict]` (only present when `save_dir` is provided)

---

## CLI Usage

The original CLI remains available.

If you have a console entrypoint installed in your environment, you can continue using it.
Otherwise, from the repo root you can run:

```bash
export PYTHONPATH="$PWD/src"
python -m copy_move_det_keypoint.run_detection --help
```

Example:

```bash
python -m copy_move_det_keypoint.run_detection \
  --input resources/images/1712_000.png resources/images/1712_000.png \
  --output runs/out/v1
```

Use `--help` to see all options.

---

## Legacy code snapshot (unchanged)

A verbatim snapshot of the original implementation (pre‑Public API changes) is kept under:

- `legacy_v1/`

This is **not** imported as part of the Python package; it exists only for reference / cross-checking.

---


## Attribution

This module is based on a modified version of ResearchIntegrity `copy-move-detection-keypoint`:

https://github.com/researchintegrity/copy-move-detection-keypoint

The original project provides keypoint-based copy-move detection for identifying duplicated regions within or across images.

Main modifications in this repository include:

- Python API for programmatic use.
- Cross-panel / pairwise matching integration.
- Prepared/cached feature workflows for faster repeated matching.
- Return objects and mask outputs used by the Recod.ai LUC SIFD solution pipeline.

The original project is licensed under AGPL-3.0. See `LICENSE` and `NOTICE.md` for details.
