"""
YOLO11 inference + export helper
- supports selecting a YOLO11 model variant (detection / seg / pose) by name
- automatic device selection (CUDA if available and requested, otherwise CPU)
- optional export to ONNX with safe device handling
- shows frames with detections and (if available) masks

Usage examples:
  python yolo11_pipeline.py --model yolo11n.pt --source 0 --show
  python yolo11_pipeline.py --model yolo11n-seg.pt --source video.mp4 --export onnx

Requirements:
  pip install ultralytics opencv-python

Notes:
- This script assumes Ultralytics YOLO11-compatible weights; it will try to download them
  automatically if you pass a recognized model alias (like 'yolo11n').
- For exporting to ONNX we use Ultralytics export helpers; exported file will be placed
  next to the given model weights unless --out is specified.
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

import cv2
import numpy as np


DEFAULT_MODELS = {
    # these alias names are commonly used by Ultralytics; if you have custom weights, pass path
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    'yolo11m': 'yolo11m.pt',
    'yolo11n-seg': 'yolo11n-seg.pt',
}


def select_device(prefer_gpu: bool):
    """Return a device string for ultralytics: 'cpu' or 'cuda'/'0'."""
    if prefer_gpu and ULTRALYTICS_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except Exception:
            pass
    return 'cpu'


def load_model(model_spec: str, device: str):
    """Load a YOLO model using ultralytics. model_spec may be an alias or a path."""
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError('ultralytics package is not installed. pip install ultralytics')

    # map alias to official name if present
    model_path = DEFAULT_MODELS.get(model_spec, model_spec)

    # allow passing a local path
    if os.path.exists(model_path):
        y = YOLO(model_path)
    else:
        # ultralytics will download if alias is known
        y = YOLO(model_path)

    # move to device if possible (ultralytics handles device per predict call, but ensure no surprise)
    try:
        if device != 'cpu':
            y.model.to(device)
    except Exception:
        # ignore if model object doesn't expose .model or can't be moved
        pass

    return y


def draw_detection(frame, box, label, score, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    txt = f"{label} {score:.2f}"
    cv2.putText(frame, txt, (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='yolo11n', help='model alias or path (e.g. yolo11n, yolo11n-seg, /path/to/weights.pt)')
    parser.add_argument('--source', '-s', default=0, help='camera index or video file')
    parser.add_argument('--device', '-d', default='auto', choices=['auto','cpu','gpu'], help='device selection')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
    parser.add_argument('--show', action='store_true', help='display results in a window')
    parser.add_argument('--export', choices=['onnx','none'], default='none', help='export model to format')
    parser.add_argument('--out', default=None, help='output path for exported model')
    args = parser.parse_args()

    prefer_gpu = (args.device == 'gpu' or (args.device == 'auto'))
    device = select_device(prefer_gpu)
    print('Selected device:', device)

    if not ULTRALYTICS_AVAILABLE:
        print('ERROR: ultralytics not installed. Run: pip install ultralytics')
        sys.exit(1)

    print('Loading model:', args.model)
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print('Failed to load model:', e)
        sys.exit(1)

    # export if requested
    if args.export == 'onnx':
        out_path = args.out or (Path(model.pt_path).stem + '.onnx') if hasattr(model, 'pt_path') else (args.model + '.onnx')
        try:
            print('Exporting to ONNX ->', out_path)
            model.export(format='onnx', imgsz=args.imgsz, simplify=True, opset=13)
            print('Export complete')
        except Exception as e:
            print('Export failed:', e)

    # open source
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('Failed to open source:', args.source)
        sys.exit(1)

    window_name = 'YOLO11 pipeline'
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB for ultralytics if providing array
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            # ultralytics predict can accept numpy arrays
            results = model.predict(source=rgb, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=(device if device=='cpu' else '0'))
            t1 = time.time()

            r = results[0]
            # boxes
            if hasattr(r, 'boxes') and r.boxes is not None:
                # new API: r.boxes.xyxy, r.boxes.conf, r.boxes.cls
                try:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy()
                    for i in range(len(xyxy)):
                        c = int(cls[i]) if cls is not None else -1
                        score = float(confs[i]) if confs is not None else 0.0
                        label = str(c)
                        draw_detection(frame, xyxy[i], label, score)
                except Exception:
                    # fallback older interface
                    try:
                        for b in r.boxes:
                            xy = b.xyxy[0].cpu().numpy() if hasattr(b, 'xyxy') else b.xyxy
                            conf = float(b.conf) if hasattr(b, 'conf') else 0.0
                            clsid = int(b.cls) if hasattr(b, 'cls') else -1
                            draw_detection(frame, xy, str(clsid), conf)
                    except Exception:
                        pass

            # masks (if available)
            if hasattr(r, 'masks') and r.masks is not None:
                try:
                    masks = r.masks.data.cpu().numpy()  # shape: N,H,W
                    for mi in range(masks.shape[0]):
                        mask = masks[mi]
                        color = np.array([0,255,0], dtype=np.uint8)
                        colored = (np.stack([mask]*3, axis=-1) * color).astype(np.uint8)
                        alpha = (mask*0.4).astype(np.float32)
                        frame = (frame*(1-alpha[...,None]) + colored*alpha[...,None]).astype(np.uint8)
                except Exception:
                    pass

            fps = 1.0 / max(1e-6, (t1 - t0))
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if args.show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                # print basic info
                print(f"frame processed, fps={fps:.1f}")

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
