# panel-type-classifier

Zero-shot classifier for scientific-figure **panel crops**:
- **Group**: `imaging` vs `non_imaging`
- **Subtype** (only if imaging): `blot_gel`, `microscopy`, `histology_pathology`, `flow_cytometry`, `plot_chart`, `photo_natural`, `diagram_schematic`, `radiology`, `endoscopy`, `assay_plate`, ... (see `panel_type_classifier/prompts.py`)

Under the hood it uses **BiomedCLIP** (OpenCLIP) and a prompt pack tuned for paper panels.

## Install (local / repo)
```bash
pip install -U open_clip_torch==2.23.0 pillow tqdm pandas matplotlib
# inside this repo:
pip install -e .

# or if repo in another folder
pip install -e /path/to/panel-type-classifier
```

## Quick usage (Python)
```python
from panel_type_classifier import PanelTypeClassifier

clf = PanelTypeClassifier()  # auto device
r = clf.predict("/path/to/panel.png", return_probs=False)

print(r.group_label, r.group_score)
print(r.subtype_label, r.subtype_score)

emb = clf.encode_image("/path/to/panel.png")  # unit-norm embedding (float32)
```

Supported inputs:
- image path (`str` / `Path`)
- `PIL.Image`
- `numpy.ndarray` (HxW, HxWx1, HxWx3, HxWx4)

## Batch folder run (script)
```bash
python scripts/classify_folder.py --root /path/to/panels --out out/panel_type --recursive --batch-size 32
```

Outputs:
- `predictions.csv`
- `summary.json`
- `counts_group.png`, `counts_subtype.png`
- `samples_group.png`, `samples_subtype.png`

## Kaggle tips
- Put this folder in your Kaggle working directory and add to `sys.path`, or `pip install -e .`.
- If you see `NaN` probabilities, try disabling AMP:
  ```python
  from panel_type_classifier.classifier import PanelTypeClassifier, ModelConfig
  clf = PanelTypeClassifier(config=ModelConfig(use_amp=False))
  ```

## Customizing prompts
Edit `panel_type_classifier/prompts.py` or pass your own `PromptPack` to `PanelTypeClassifier(...)`.

---

# Old Readme

**Panel Type Classifier** predicts semantic and integrity-related metadata for extracted scientific figure panels.
It classifies panels as imaging vs non-imaging, identifies imaging modality (microscopy, blot/gel, histology, other), and detects attributes such as scientific annotations and inset/inserted regions.

The goal is **panel routing**, not end-task inference — outputs are used to select the correct downstream pipelines (CBIR, reuse localization, annotation masking, manipulation analysis).

---

## What it predicts

* **Panel category**

  * Imaging / Non-imaging
* **Imaging modality** (if imaging)

  * Microscopy
  * Blot / Gel
  * Histology / Pathology
  * Other imaging
* **Panel attributes**

  * Annotations present (text, arrows, scale bars, overlays)
  * Inset / inserted regions present

---

## Why this exists

Scientific figures mix many panel types.
Running all panels through the same forensic pipeline is inefficient and error-prone.

This classifier provides **structured panel metadata** so that:

* Non-imaging panels can be skipped
* Annotation-heavy panels can be masked first
* High-risk imaging panels (e.g., blots) receive deeper analysis

---

## Output (example)

```json
{
  "panel_id": "fig3_panel_b",
  "is_imaging": true,
  "imaging_type": "microscopy",
  "has_annotation": true,
  "has_inset": false
}
```

---

## Typical usage

1. Extract panels from scientific figures
2. Run Panel Type Classifier on each panel
3. Route panels to:

   * Content-based image retrieval
   * Panel reuse / duplication detection
   * Annotation segmentation
   * Forensic integrity checks

---

## Scope

* Panel-level only (not full-figure parsing)
* Designed to work with real + synthetic biomedical panels
* Model-agnostic (supports CNNs, CLIP-style encoders, vision-LLMs)
