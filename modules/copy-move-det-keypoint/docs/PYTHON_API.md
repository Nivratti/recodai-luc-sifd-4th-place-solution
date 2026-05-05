# Python API (detailed)

This document describes the **public Python API** for `copy_move_det_keypoint`, including
the **prepared/cached features** workflow for fast one-to-many and all-pairs matching.

---

## 1) Pairwise API

### `detect(source, target=None, *, config=None, save_dir=None, assume_bgr=True) -> DetectionResult`

- If `target is None`, runs **single-image** copy-move detection.
- If `target is not None`, runs **cross-image** overlap/reuse detection.
- Returns arrays + metrics. Writes files only when `save_dir` is provided.

#### Input types
Both `source` and `target` accept:
- `str` / `pathlib.Path`: image file path
- `np.ndarray`: image array (**assumed BGR** by default, like `cv2.imread`)
- `PIL.Image.Image`: PIL image (RGB)

If your numpy array is already RGB, set `assume_bgr=False`.

#### Output: `DetectionResult`
- `is_match: bool`
- `mask_source, mask_target: np.ndarray` (uint8, H×W)
- `matched_kpts_source, matched_kpts_target: np.ndarray` (float32, N×2)
- `shared_area_source, shared_area_target: float` (0..1)
- `is_flipped: bool`
- `matched_keypoints: int`
- `outputs: Optional[dict[str,str]]` (only when `save_dir` is provided)

Example:
```python
from copy_move_det_keypoint import detect

res = detect("a.png", "b.png")
print(res.is_match, res.shared_area_source, res.shared_area_target)
```

---

## 2) Prepared / cached features (fast multi-matching)

When you have many targets (or all-pairs), feature extraction dominates runtime.
Use `prepare()` to extract features once per image/panel and reuse them.

### `FeatureSet`
A lightweight container:
- `image_id: str`
- `shape_hw: (H, W)`
- `keypoints: (N,2)`
- `descriptors: (N,D)`
- `flip_keypoints, flip_descriptors` (optional; used when `check_flip=True`)
- `image_bgr` (optional; only needed if you want to save match visualizations)

### `prepare(image, *, config=None, image_id=None, keep_image=False, assume_bgr=True, extract_flip=None, kp_count=None) -> FeatureSet`

Example:
```python
from copy_move_det_keypoint import prepare, DetectorConfig

cfg = DetectorConfig()
src = prepare("panelA.png", config=cfg, image_id="panelA")
tgt = prepare("panelB.png", config=cfg, image_id="panelB")
```

### `match_prepared(source, target, *, config=None, save_dir=None) -> DetectionResult`

- Runs matching + geometric verification using prepared features.
- Builds masks and returns a standard `DetectionResult`.
- If `save_dir` is provided:
  - Always saves mask files.
  - Saves matches/clusters visualizations only if both `FeatureSet.image_bgr` are present (created via `keep_image=True`).

Example:
```python
from copy_move_det_keypoint import match_prepared

res = match_prepared(src, tgt, config=cfg)
print(res.is_match, res.matched_keypoints)
```

---

## 3) Optional pruning API ("pure matcher")

For very large runs, you may want to avoid mask creation (and especially disk I/O)
for obvious non-matches.

### `match_keypoints_only(source, target, *, config=None) -> MatchInfo`
Returns arrays + metrics **without building masks**.

### `build_masks_from_matches(match, source_shape_hw, target_shape_hw) -> (mask_source, mask_target)`
Build convex-hull masks later, only for the pairs you decide to keep.

Example:
```python
from copy_move_det_keypoint import match_keypoints_only, build_masks_from_matches

mi = match_keypoints_only(src, tgt, config=cfg)
if mi.matched_keypoints >= 40 or max(mi.shared_area_source, mi.shared_area_target) >= 0.05:
    maskA, maskB = build_masks_from_matches(mi, src.shape_hw, tgt.shape_hw)
```

---

## 4) Recommended multi-matching patterns

### A) One source → many targets

```python
from copy_move_det_keypoint import prepare, match_keypoints_only, DetectorConfig

cfg = DetectorConfig()
src = prepare(source_img, config=cfg, image_id="src")

edges = []
for i, t in enumerate(targets):
    ft = prepare(t, config=cfg, image_id=f"t{i}")
    mi = match_keypoints_only(src, ft, config=cfg)
    edges.append({
        "source": src.image_id,
        "target": ft.image_id,
        "is_match": mi.is_match,
        "matched_keypoints": mi.matched_keypoints,
        "shared_area_source": mi.shared_area_source,
        "shared_area_target": mi.shared_area_target,
        "is_flipped": mi.is_flipped,
    })
```

### B) All-pairs (each image becomes source)

```python
from itertools import combinations
from copy_move_det_keypoint import prepare, match_keypoints_only, DetectorConfig

cfg = DetectorConfig()
features = [prepare(p, config=cfg, image_id=str(i)) for i, p in enumerate(paths)]

for i, j in combinations(range(len(features)), 2):
    mi = match_keypoints_only(features[i], features[j], config=cfg)
    # store edge, filter, etc.
```

---

## Notes on accuracy and color

- All matching/verification logic is reused from the existing implementation.
- For numpy arrays, channel order matters:
  - OpenCV arrays are typically **BGR** (default `assume_bgr=True`)
  - PIL / matplotlib arrays are typically **RGB** (`assume_bgr=False`)
