# panel-reuse-detection
Detect and localize duplicated, cropped, and overlapping regions between scientific figure panels.

panel-reuse-detection is a pairwise image-matching toolkit for scientific figures. Given two panel images at a time, it identifies panel reuse patterns—full duplicates, crop/containment matches, and partial overlaps—and outputs localized match regions (boxes/overlap areas), alignment hints (rotate/flip/scale), and reviewer-friendly JSON reports. Designed to reduce reviewer workload by quickly surfacing the most suspicious reuse candidates for deeper forensic checks.

## Why this exists

A single figure can contain 10–50+ panels. Reviewing every pair manually is slow.  
This project helps you:

- **Detect panel reuse** quickly (exact/near duplicates).
- Handle **crop/containment** (one panel is a crop/zoom-inset of the other).
- Handle **partial overlap** (shared region but neither contains the other).
- **Localize** the matching region(s) so a reviewer can verify evidence fast.
- Export **JSON reports** suitable for later analysis and model debugging.

## What it outputs (pairwise)

For each pair `(A, B)`:

- `match_type` (exactly one):
  - `FULL_DUPLICATE`
  - `FULL_OVERLAP_CROP` (containment / crop match)
  - `PARTIAL_OVERLAP`
  - `NO_MATCH`
- Localization:
  - `box_in_big` for containment (`A_IN_B` or `B_IN_A`)
  - `overlap_boxes` for partial overlap
- Alignment hints:
  - rotation / flip / scale / small shifts (best transform)
- Scores:
  - global similarity (`FULL_DUPLICATE`)
  - contained-region similarity (`FULL_OVERLAP_CROP`)
  - overlap-region similarity (`PARTIAL_OVERLAP`)
- Optional tags:
  - geometric edits, photometric edits, overlay differences, overlap pattern/strength

---
