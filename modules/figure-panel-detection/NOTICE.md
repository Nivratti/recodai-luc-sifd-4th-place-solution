# Notice

This module contains a modified version of the ResearchIntegrity `panel-extractor` project.

Original project:

[https://github.com/researchintegrity/panel-extractor]()

The original project provides YOLO-based panel extraction for scientific images.

Main modifications in this repository:

- Added ONNX Runtime inference support for the panel detector.
- Added Python API wrappers for easier integration into the Recod.ai LUC SIFD pipeline.
- Added integration utilities for running panel detection inside the Kaggle solution pipeline.
- Added output formatting and crop handling used by the final solution workflow.
- Adjusted project structure and runtime behavior for this public solution repository.
