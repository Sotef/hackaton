"""Quick YOLOv8 test script.

This script tries to import ultralytics, create a tiny YOLOv8 segmentation/detection model
( 'yolov8n-seg.pt' or 'yolov8n.pt' will be downloaded automatically if needed), runs a single
inference on a synthetic image and prints basic results.

Run:
    python yolo8_test.py
"""
import sys
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics package not installed or failed to import:", e)
    print("Install with: pip install ultralytics")
    sys.exit(1)

import cv2

def main():
    # try segmentation weight first, fallback to detection
    candidates = ['yolov8n-seg.pt', 'yolov8n.pt']
    model = None
    for m in candidates:
        try:
            print(f"Loading model {m} ...")
            model = YOLO(m)
            break
        except Exception as e:
            print(f"Failed to load {m}: {e}")

    if model is None:
        print("Could not load a YOLOv8 model automatically. Please download weights or check network.")
        sys.exit(1)

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
