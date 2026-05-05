#!/usr/bin/env bash
set -euo pipefail

pip install -e .
pip install -e modules/figure-panel-detection
pip install -e modules/copy-move-det-keypoint

# later:
# pip install -e modules/panel-type-classifier
# pip install -e modules/panel-cbir
