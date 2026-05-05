# Notice

This module contains a modified version of the ResearchIntegrity `copy-move-detection-keypoint` project.

Original project:

https://github.com/researchintegrity/copy-move-detection-keypoint

The original project provides keypoint-based copy-move detection for identifying duplicated regions within or across images.

Main modifications in this repository:

- Added Python API for programmatic use inside the Recod.ai LUC SIFD pipeline.
- Added cross-panel / pairwise matching integration.
- Added reusable prepared-feature and cached-feature style workflows.
- Added return objects and mask outputs used by the final solution pipeline.
- Adjusted project structure and runtime behavior

License:

The original project is licensed under AGPL-3.0. This modified version keeps the same license attribution. See the local LICENSE file and the original upstream repository for license details.
