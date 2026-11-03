# openWakeWord models

`melspectrogram.onnx`, `embedding_model.onnx` and `hey_jarvis_v0.1.onnx` come
from openWakeWord 0.4.0 (https://github.com/dscripka/openWakeWord). Not modified.

- `melspectrogram.onnx` and `embedding_model.onnx` are the feature models. The
  embedding model is Google's speech embedding, Apache-2.0.
- `hey_jarvis_v0.1.onnx` is an openWakeWord pre-trained model. openWakeWord
  licenses its pre-trained models under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  (non-commercial, share-alike), because of their training data. That license,
  not this repository's MIT license, applies to this file.

These are the exact files the second-stage tests were run with; the v0.5.1
release ships a different `embedding_model.onnx` export, so they are bundled
rather than downloaded.
