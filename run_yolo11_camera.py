#!/usr/bin/env python3
"""
Run YOLO (ONNX) on webcam using OpenCV DNN as a fallback for environments where torch/ultralytics cannot be imported.

Usage:
  python run_yolo11_camera.py --model path/to/model.onnx --max-frames 200

If --model is not provided the script searches the repository for *.onnx (yolo11/yolo* first).
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
try:
    import onnxruntime as ort
except Exception:
    ort = None


def find_onnx_model(root: Path):
    # Prefer yolo11/yolo* models, then any onnx
    patterns = ["**/yolo11*.onnx", "**/yolo*.onnx", "**/*.onnx"]
    for pat in patterns:
        files = list(root.glob(pat))
        if files:
            list(root.glob(pat))
            return str(files[0])
    return None


def load_model_cv2(onnx_path: str):
    try:
        net = cv2.dnn.readNetFromONNX(onnx_path)
        print(f"Loaded ONNX model: {onnx_path}")
        return net
    except Exception as e:
        print(f"Failed to load ONNX with cv2.dnn: {e}")
        return None


def load_model_ort(onnx_path: str):
    if ort is None:
        print("onnxruntime not available in this environment.")
        return None
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        print(f"Loaded ONNX model with onnxruntime: {onnx_path}, input: {inp.name}, shape: {inp.shape}")
        return sess
    except Exception as e:
        print(f"Failed to load ONNX with onnxruntime: {e}")
        return None


def detect_and_draw(frame, net, conf_thresh=0.25, iou_thresh=0.45):
    h, w = frame.shape[:2]
    length = max(h, w)
    # pad to square
    image = np.zeros((length, length, 3), np.uint8)
    image[0:h, 0:w] = frame
    scale = length / 640.0

    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(640, 640), swapRB=True)
    net.setInput(blob)
    outputs = net.forward()

    try:
        outputs = np.array([cv2.transpose(outputs[0])])
    except Exception:
        # Some models return shape (1,N,?) already
        outputs = np.array(outputs)

    # outputs expected: [1, N, 4+num_classes]
    if outputs.ndim < 3:
        return frame, 0

    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        row = outputs[0][i]
        if row.size <= 5:
            continue
        classes_scores = row[4:]
        # cv2.minMaxLoc expects float32 image; convert
        cs = classes_scores.astype(np.float32)
        _, maxScore, _, maxClassIndex = cv2.minMaxLoc(cs)
        if maxScore >= conf_thresh:
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x = cx - 0.5 * bw
            y = cy - 0.5 * bh
            boxes.append([x, y, float(bw), float(bh)])
            scores.append(float(maxScore))
            class_ids.append(int(maxClassIndex[1]) if isinstance(maxClassIndex, tuple) else int(maxClassIndex))

    if not boxes:
        return frame, 0

    # NMS expects boxes as [x,y,w,h]
    try:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    except Exception:
        # older opencv may return a flat list
        idxs = []

    drawn = 0
    if len(idxs) > 0:
        # idxs may be nested arrays
        try:
            iter_idx = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]
        except Exception:
            iter_idx = list(idxs)

        for j in iter_idx:
            box = boxes[j]
            score = scores[j]
            cls_id = class_ids[j] if j < len(class_ids) else 0
            x1 = int(round(box[0] * scale))
            y1 = int(round(box[1] * scale))
            x2 = int(round((box[0] + box[2]) * scale))
            y2 = int(round((box[1] + box[3]) * scale))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"id:{cls_id} {score:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            drawn += 1

    return frame, drawn


def detect_and_draw_ort(frame, sess, conf_thresh=0.25, iou_thresh=0.45):
    # Use onnxruntime session to run inference and then try to reuse the same postprocessing
    h, w = frame.shape[:2]
    length = max(h, w)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:h, 0:w] = frame
    scale = length / 640.0

    blob = cv2.resize(image, (640, 640))
    blob = blob.astype(np.float32) / 255.0
    # HWC to CHW
    input_tensor = np.transpose(blob, (2, 0, 1))[None, :]

    try:
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: input_tensor})
        # Try to coerce outputs into the same shape cv2.dnn produced
        out0 = outputs[0]
        try:
            # many YOLO ONNX exports produce shape (1, N, 85) or (N,85)
            if out0.ndim == 3:
                arr = out0
            elif out0.ndim == 2:
                arr = out0[None, :, :]
            else:
                arr = np.array(out0)
            # try to make array shaped like [1, N, C]
            outputs_for_post = np.array([np.transpose(arr[0])])
            # reuse postprocessing by calling existing function behavior
            # Build a temporary net-like object is not needed; call same processing loop here
            rows = outputs_for_post.shape[1]
            boxes = []
            scores = []
            class_ids = []
            for i in range(rows):
                row = outputs_for_post[0][i]
                if row.size <= 5:
                    continue
                classes_scores = row[4:]
                cs = classes_scores.astype(np.float32)
                _, maxScore, _, maxClassIndex = cv2.minMaxLoc(cs)
                if maxScore >= conf_thresh:
                    cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    x = cx - 0.5 * bw
                    y = cy - 0.5 * bh
                    boxes.append([x, y, float(bw), float(bh)])
                    scores.append(float(maxScore))
                    class_ids.append(int(maxClassIndex[1]) if isinstance(maxClassIndex, tuple) else int(maxClassIndex))

            drawn = 0
            if boxes:
                try:
                    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
                except Exception:
                    idxs = []
                if len(idxs) > 0:
                    try:
                        iter_idx = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]
                    except Exception:
                        iter_idx = list(idxs)
                    for j in iter_idx:
                        box = boxes[j]
                        score = scores[j]
                        cls_id = class_ids[j] if j < len(class_ids) else 0
                        x1 = int(round(box[0] * scale))
                        y1 = int(round(box[1] * scale))
                        x2 = int(round((box[0] + box[2]) * scale))
                        y2 = int(round((box[1] + box[3]) * scale))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"id:{cls_id} {score:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        drawn += 1

            return frame, drawn
        except Exception as e:
            print(f"onnxruntime postprocessing failed: {e}; outputs shapes: {[o.shape for o in outputs]}")
            return frame, 0
    except Exception as e:
        print(f"onnxruntime run failed: {e}")
        return frame, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--max-frames", type=int, default=300, help="Stop after this many frames (auto-exit)")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_path = args.model
    if model_path is None:
        found = find_onnx_model(root)
        if found:
            print(f"Found ONNX model in repo: {found}")
            model_path = found
        else:
            print("No ONNX model found in repository. Running camera without detection (you can pass --model path/to/model.onnx)")

    net = None
    if model_path:
        net = load_model_cv2(model_path)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.cam}. Exiting.")
        return

    print("Camera opened. Press 'q' in the window to quit.")
    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break

            frame_count += 1
            drawn = 0
            if net is not None:
                try:
                    out_frame, drawn = detect_and_draw(frame, net)
                except Exception as e:
                    print(f"Detection error: {e}")
                    out_frame = frame
            else:
                out_frame = frame

            # overlay fps and info
            elapsed = max(1e-6, time.time() - t0)
            fps = frame_count / elapsed
            cv2.putText(out_frame, f"FPS: {fps:.1f}  Drawn: {drawn}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            try:
                cv2.imshow("YOLO11 Camera", out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            except Exception as e:
                # Likely headless OpenCV without GUI support
                print("OpenCV GUI not available or failed to show window:", e)
                # Save a frame periodically instead
                if frame_count % 30 == 0:
                    out_path = root / f"camera_frame_{frame_count}.jpg"
                    cv2.imwrite(str(out_path), out_frame)
                    print(f"Saved frame to {out_path}")

            if args.max_frames and frame_count >= args.max_frames:
                print(f"Reached max frames={args.max_frames}. Exiting.")
                break

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
