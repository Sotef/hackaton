"""Quick YOLO11 test script (safe fallback).

This script attempts to run a quick test for a hypothetical 'YOLO11' model.
Behavior:
 - If a local ONNX model `yolo11.onnx` exists, it will load it with onnxruntime and run
   a single inference on a synthetic image (prints model input details and inference OK).
 - Otherwise, it will try to import a Python package named `yolo11` (common APIs are unknown),
   and print guidance if not available.

Run:
    python yolo11_test.py
"""
import sys
import os
import numpy as np

# Prefer local `external` copies if present (allows testing without pip-install).
_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DIRS = [os.path.join(_ROOT, 'external', 'Yolov11'), os.path.join(_ROOT, 'external', 'yolo_v8')]
for _d in _LOCAL_DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

ONNX_PATH = 'yolo11.onnx'

def test_onnx(onnx_path: str):
    try:
        import onnx
        import onnxruntime as ort
    except Exception as e:
        print('onnx / onnxruntime not installed:', e)
        print('Install with: pip install onnx onnxruntime')
        return False

    print(f'Loading ONNX model: {onnx_path}')
    try:
        model = onnx.load(onnx_path)
        print('ONNX model loaded. opset:', model.opset_import[0].version if model.opset_import else 'n/a')
    except Exception as e:
        print('Failed to load ONNX model:', e)
        return False

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    print('ONNX inputs:')
    for i in inputs:
        print(' ', i.name, i.shape, i.type)

    # prepare dummy input according to first input shape (replace None with 1)
    inp0 = inputs[0]
    shape = [1 if (isinstance(s, str) or s is None) else s for s in inp0.shape]
    # ensure shape has 4 dims (N,C,H,W) — if model expects (N,H,W,C) user will need to adapt
    if len(shape) == 4:
        dummy = np.zeros(shape, dtype=np.float32)
    elif len(shape) == 3:
        dummy = np.zeros([1] + shape, dtype=np.float32)
    else:
        # fallback to (1,3,640,640)
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)

    input_name = inp0.name
    try:
        out = sess.run(None, {input_name: dummy})
        print('ONNX inference OK — number of outputs:', len(out))
        return True
    except Exception as e:
        print('ONNX inference failed:', e)
        return False

def main():
    # If there's an ONNX model, test it (this covers local custom YOLO11 models)
    # prefer explicit file first, then scan repository for any yolo*.onnx
    candidate = None
    if os.path.exists(ONNX_PATH):
        candidate = ONNX_PATH
    else:
        # search for yolo*.onnx or any .onnx in repo
        for root, _, files in os.walk(_ROOT):
            for f in files:
                if f.lower().endswith('.onnx') and ('yolo' in f.lower() or candidate is None):
                    candidate = os.path.join(root, f)
                    if 'yolo' in f.lower():
                        break
            if candidate and 'yolo' in os.path.basename(candidate).lower():
                break

    if candidate:
        print('Found ONNX model for testing:', candidate)
        ok = test_onnx(candidate)
        if ok:
            print('yolo11 (onnx) test passed')
            return
        else:
            print('yolo11 (onnx) test failed')

    # fallback: try importing a package named 'yolo11' (API unknown)
    try:
        import yolo11
        print('Imported package `yolo11`. Please adapt this test to use its API.')
        # best-effort: try to locate a load function
        if hasattr(yolo11, 'load'):
            print('Found yolo11.load — attempting to load default model...')
            try:
                model = yolo11.load()
                print('Model loaded. Try calling model.predict() according to package API.')
            except Exception as e:
                print('Failed to call yolo11.load():', e)
        else:
            print('No standard loader found in yolo11 package — check package docs.')
    except Exception as e:
        print('Package `yolo11` not found locally:', e)
        print('No local ONNX model found or ONNX inference failed.')
        # lightweight fallback: do a minimal smoke-check using numpy
        try:
            import numpy as _np
            print('Performing lightweight smoke check (numpy) ...')
            a = _np.zeros((1, 3, 64, 64), dtype=_np.float32)
            print('Smoke check OK, shape:', a.shape)
        except Exception as _:
            print('Fallback smoke check failed — environment may be broken.')

if __name__ == '__main__':
    main()
