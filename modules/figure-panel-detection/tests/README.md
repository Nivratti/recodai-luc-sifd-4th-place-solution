# Tests

Run all tests:

```bash
pytest
```

## Integration tests (real ONNX)

By default, integration tests run automatically **if** a model is found at:

- `resources/models/yolov5-onnx/model_4_class.onnx`

Or set an explicit path:

```bash
export FIGURE_PANEL_DET_ONNX=/abs/path/to/model_4_class.onnx
pytest -m integration
```

If no model is available, unit tests still run using a DummyPredictor and
integration tests are skipped.


## Unit-tests only (skip model check)

```bash
export FIGURE_PANEL_DET_SKIP_INTEGRATION=1
pytest
```
