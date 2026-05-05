## Panel Content Segmentation

**Binary content mask for scientific figure panels (content vs non-content artifacts).**

### Short description

This project generates synthetic training data and trains/evaluates models to predict a **binary content mask** for a panel crop: **true image content** vs **non-content layout artifacts** such as **border, padding, canvas/background, and gutters/separators**.

### Goals

* Produce a **high-precision binary mask**:
  **255 = content**, **0 = non-content**
* Be robust to messy real-world crops (slight misalignment, partial tight crops, uneven borders/padding).
* Support scalable synthetic dataset generation with reproducible configs (YAML) and manifests (train/val).
* Enable downstream pipelines to:

  * extract clean content ROI,
  * ignore borders/padding/background during matching/forgery checks.

### Purpose

Scientific figures often contain layout artifacts (borders, padding, separators, whitespace) that confuse:

* panel-to-panel matching,
* near-duplicate detection,
* copy-move / reuse localization,
* downstream segmentation or feature extraction.

This project provides a **content-only mask** so later stages can focus on **real visual evidence** (cells, bands, tissue, structures) rather than figure layout.

---

### YOLO panel detection crop issues (why content mask helps)

Panel crops produced by detectors (e.g., **YOLO panel detection**) are often imperfect:

* **Loose boxes** include extra surrounding regions: figure background, neighboring panel parts, gutters/separators.
* **Tight boxes** cut the panel: border/content gets truncated.
* **Boundary jitter**: box edges don’t align to true panel borders (especially with thin/low-contrast borders).
* **Grid layouts**: separators and whitespace get captured inside the crop.

These detector-induced errors introduce **non-content pixels** that degrade matching, reuse detection, and downstream modeling.
A **binary content mask** lets you *normalize imperfect detector crops* by isolating true content and ignoring border/padding/background/gutters.

---

## Terms: Content vs Non-content

### Content (positive class, 255)

Pixels that belong to the **true underlying image data** inside the panel, e.g.:

* microscopy / histology / IHC tissue structures
* western blot bands and background texture
* gel images, flow plots *if treated as image content*
* any real scene/texture captured in the panel image

**Rule:** Content pixels come from the **real input panel image** (after resize+crop while preserving aspect ratio). No synthetic filler is considered content.

### Non-content (negative class, 0)

Pixels that are **not true image content**, including:

* **Border:** outer frame lines, boxes, dashed/broken frames
* **Padding:** empty margins inside the panel boundary around content
* **Canvas / Background:** whitespace around the panel crop (figure background)
* **Gutters / Separators:** white or light strips/lines separating panels or grid regions (layout gaps)

**Rule:** Non-content = anything that exists due to layout/formatting rather than the actual image.

---
