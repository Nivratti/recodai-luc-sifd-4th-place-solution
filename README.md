# `recodai-sifd`

### Scientific Image Forgery Detection — wrapper workspace + utilities (Recod.ai / LUC Kaggle)

`recodai-sifd` is a **library-first workspace** for building a modular solution for the Kaggle competition **Recod.ai / LUC – Scientific Image Forgery Detection**.

This repo is intentionally designed as an **orchestration + experimentation wrapper** around multiple independent git submodules (panel detection, panel-type classification, CBIR, reuse detection, segmentation, etc.), plus shared utilities for **EDA, evaluation, and debugging**.

## Goals

* Provide a clean **workspace** to integrate multiple specialized modules.
* Make it easy to run **step-by-step experiments** in notebooks before assembling an end-to-end pipeline.
* Keep evaluation consistent with the competition metric.

## Repo philosophy

* **Library-first:** core logic should eventually live in reusable Python modules (but this repo starts in “heavy dev mode”).
* **Wrapper-oriented:** this repo connects multiple submodules and adds glue code / utilities.
* **Notebook-driven development:** each component is tested in isolation first (panel detection → reuse matching → grouping → mask merge → evaluation).
* **Evaluation-aware:** evaluation tools will be added early so changes can be validated quickly.

## Notebook policy (development workflow)
- Notebooks exist to **test and validate module functionality** (e.g., load model, run on a few images, visualize outputs, inspect failures).
- As soon as notebook code starts repeating, we promote it into **reusable utilities/helpers** so notebooks stay small.
- Notebooks should remain **thin runners**: configuration + a few function calls + visualization and saved artifacts.
- Utilities/helpers are built **incrementally (one-by-one)** to reduce notebook burden and keep the workspace maintainable.

