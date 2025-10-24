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
    if os.path.exists(ONNX_PATH):
        ok = test_onnx(ONNX_PATH)
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
        print("""If you have a YOLO11 implementation, either:
  - place an ONNX model named `yolo11.onnx` in this folder and re-run,
  - or install the Python package that provides YOLO11 (then adapt this script to its API).""")

if __name__ == '__main__':
    main()
