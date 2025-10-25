"""Quick YOLOv8 test script.

This script tries to import ultralytics, create a tiny YOLOv8 segmentation/detection model
( 'yolov8n-seg.pt' or 'yolov8n.pt' will be downloaded automatically if needed), runs a single
inference on a synthetic image and prints basic results.

Run:
    python yolo8_test.py
"""
import sys
import os
import numpy as np

# Prefer local `external` copies if present (allows testing without pip-install).
_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DIRS = [os.path.join(_ROOT, 'external', 'yolo_v8'), os.path.join(_ROOT, 'external', 'Yolov11')]
for _d in _LOCAL_DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

ULTRALYTICS_AVAILABLE = True
ultralytics_import_error = None
try:
    from ultralytics import YOLO
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    ultralytics_import_error = e
    print("ultralytics package not installed or failed to import:", e)
    print("Install with: pip install ultralytics or place a local copy in external/yolo_v8/")

import cv2

def main():
    # try segmentation weight first, fallback to detection
    candidates = ['yolov8n-seg.pt', 'yolov8n.pt']
    model = None
    # Try to load via ultralytics only if import succeeded
    if ULTRALYTICS_AVAILABLE:
        for m in candidates:
            try:
                print(f"Loading model {m} ...")
                model = YOLO(m)
                break
            except Exception as e:
                print(f"Failed to load {m}: {e}")
    else:
        print('Skipping ultralytics model load due to import error:', ultralytics_import_error)

    if model is None:
        # If ultralytics couldn't load, try to find an ONNX model and run onnxruntime
        print("Could not load a YOLOv8 model automatically via ultralytics.")
        import onnxruntime as _ort
        # search for yolov8*.onnx or any .onnx
        onnx_candidate = None
        for root, _, files in os.walk(_ROOT):
            for f in files:
                if f.lower().endswith('.onnx') and ('yolov8' in f.lower() or onnx_candidate is None):
                    onnx_candidate = os.path.join(root, f)
                    if 'yolov8' in f.lower():
                        break
            if onnx_candidate and 'yolov8' in os.path.basename(onnx_candidate).lower():
                break

        if onnx_candidate:
            print('Found ONNX model, running onnxruntime inference:', onnx_candidate)
            try:
                sess = _ort.InferenceSession(onnx_candidate, providers=['CPUExecutionProvider'])
                inp = sess.get_inputs()[0]
                shape = [1 if (s is None or isinstance(s, str)) else s for s in inp.shape]
                if len(shape) == 4:
                    dummy = np.zeros(shape, dtype=np.float32)
                elif len(shape) == 3:
                    dummy = np.zeros([1] + shape, dtype=np.float32)
                else:
                    dummy = np.zeros((1, 3, 320, 320), dtype=np.float32)
                out = sess.run(None, {inp.name: dummy})
                print('ONNX inference OK — outputs:', len(out))
                return
            except Exception as e:
                print('ONNX inference failed:', e)
        print('No ONNX model found — skipping heavy ultralytics test.')
        return

    # create a synthetic image (RGB uint8)
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    # draw a white rectangle to simulate an object
    cv2.rectangle(img, (80, 80), (240, 240), (255, 255, 255), -1)

    print("Running single inference on synthetic image...")
    try:
        results = model.predict(source=img, imgsz=320, device='cpu', verbose=False)
        # results is a list-like; print summary
        if len(results) > 0:
            r = results[0]
            print("Prediction OK — attributes available:", dir(r)[:10])
            # if masks available
            if hasattr(r, 'masks') and r.masks is not None:
                print("Masks present. mask shape (approx):", getattr(r.masks, 'data', 'n/a'))
            if hasattr(r, 'boxes'):
                try:
                    print("Boxes count:", len(r.boxes))
                except Exception:
                    print("Boxes available (count unknown)")
        else:
            print("No results returned (empty list)")
    except Exception as e:
        print("Inference failed:", e)

if __name__ == '__main__':
    main()
