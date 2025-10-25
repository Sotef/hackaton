'''
#!/usr/bin/env python3
"""
yolo11_seg_cam.py

Run segmentation inference with a YOLO11-seg ONNX model on webcam using onnxruntime.
- expects ONNX model that outputs (detections, mask_prototypes)
- detections shape -> (1, C, N) or (1, N, C) depending on export. Script normalizes both.
- mask_prototypes shape -> (1, M, ph, pw)

Usage:
  python yolo11_seg_cam.py --model path/to/yolo11s-seg.onnx --cam 0 --use_gpu
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


def xywh_to_xyxy(box):
    x, y, w, h = box
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_session(onnx_path: str, use_gpu: bool):
    if ort is None:
        print("[ERR] onnxruntime not available.")
        return None
    providers = None
    if use_gpu:
        # try CUDA provider first
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            print("[INFO] onnxruntime session created with providers:", sess.get_providers())
            return sess
        except Exception as e:
            print("[WARN] CUDAExecutionProvider failed, falling back to CPU:", e)
    # CPU fallback
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        print("[INFO] onnxruntime session created with CPUExecutionProvider")
        return sess
    except Exception as e:
        print("[ERR] Failed to create onnxruntime session:", e)
        return None


def postprocess_and_draw(frame, det_out, mask_proto, conf_thresh=0.25, iou_thresh=0.45, max_det=300):
    """
    det_out: numpy array with shape (N, C) where C = 4 + 1 + num_classes + mask_dim
    mask_proto: numpy array shape (mask_dim, ph, pw)
    Coordinates in det_out are in 640-space (cx, cy, w, h) as exported by Ultralytics.
    """
    h_img, w_img = frame.shape[:2]
    # model input size (export used 640)
    model_size = 640
    scale = max(h_img, w_img) / model_size
    # ensure det_out is (N, C)
    if det_out.ndim == 3:
        det = det_out[0].T  # (N, C) expected sometimes shape (1, C, N)
    else:
        det = det_out.copy()

    # get mask parameters
    if mask_proto.ndim == 4:
        # shape (1, M, ph, pw)
        mask_proto = mask_proto[0]
    # mask_proto -> (M, ph, pw)
    M, ph, pw = mask_proto.shape
    # number of channels in det
    C = det.shape[1]
    # deduce num_classes
    # C = 4 (xywh) + 1 (obj) + num_classes + mask_dim
    mask_dim = M
    num_classes = C - 5 - mask_dim
    if num_classes < 0:
        # fallback: try alternative layout
        print("[ERR] Unexpected detection channel count:", C, "mask_dim:", mask_dim)
        return frame, 0

    # collect detections
    boxes = []     # [x, y, w, h] in model 640 coords
    scores = []
    class_ids = []
    mask_coeffs_list = []

    for row in det:
        if row.size < (5 + num_classes + mask_dim):
            continue
        obj_conf = float(row[4])
        if obj_conf < conf_thresh:
            continue
        class_scores = row[5:5 + num_classes]
        cls_id = int(np.argmax(class_scores))
        cls_conf = float(class_scores[cls_id]) * obj_conf
        if cls_conf < conf_thresh:
            continue
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x = cx - bw / 2.0
        y = cy - bh / 2.0
        boxes.append([x, y, bw, bh])
        scores.append(cls_conf)
        class_ids.append(cls_id)
        coeffs = row[5 + num_classes: 5 + num_classes + mask_dim].astype(np.float32)
        mask_coeffs_list.append(coeffs)

    if not boxes:
        return frame, 0

    # convert boxes to ints for NMS and drawing: multiply by scale to fit padded square later
    # NMS expects [x, y, w, h]
    boxes_scaled = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in boxes]
    # OpenCV NMS expects floats and returns indices
    try:
        idxs = cv2.dnn.NMSBoxes(boxes_scaled, scores, conf_thresh, iou_thresh)
        if len(idxs) == 0:
            return frame, 0
        # normalize idxs to flat list
        try:
            idxs_list = [int(i[0]) if hasattr(i, "__len__") and len(i) > 0 else int(i) for i in idxs]
        except Exception:
            idxs_list = list(map(int, idxs))
    except Exception:
        # if NMSBoxes fails, just take top-K
        idxs_list = sorted(range(len(scores)), key=lambda i: -scores[i])[:max_det]

    # prepare proto flatten
    proto_flat = mask_proto.reshape(mask_dim, -1)  # (M, ph*pw)

    overlay = frame.copy().astype(np.float32) / 255.0
    alpha = 0.6
    drawn = 0

    # To avoid drawing many overlapping masks with same color, cycle colors
    rng = np.random.RandomState(0)
    class_colors = {}

    for i in idxs_list:
        coeffs = mask_coeffs_list[i]  # (M,)
        # compute mask: coeffs @ proto_flat -> (ph*pw,)
        mask_flat = coeffs.astype(np.float32) @ proto_flat  # (ph*pw,)
        mask_map = mask_flat.reshape(ph, pw)
        # activation
        mask_map = sigmoid(mask_map)
        # resize mask to model_size x model_size (proto grid -> upsample)
        mask_up = cv2.resize(mask_map, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
        # threshold (soft)
        mask_up = np.clip(mask_up, 0.0, 1.0)
        # place mask into padded square (we used top-left aligning earlier)
        # The frame was padded with zeros at right/bottom if non-square. We'll crop later.
        # Resize mask to match the padded square size (length x length) was already model_size -> then scale to padded length:
        length = max(h_img, w_img)
        if model_size != length:
            mask_up = cv2.resize(mask_up, (length, length), interpolation=cv2.INTER_LINEAR)
        # Now crop to original frame area
        mask_crop = mask_up[0:h_img, 0:w_img]
        # binarize with soft threshold
        bin_mask = (mask_crop > 0.35).astype(np.uint8)  # tweak threshold as needed

        # choose color for class
        cls = class_ids[i]
        if cls not in class_colors:
            class_colors[cls] = tuple(int(x) for x in (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        color = class_colors[cls]
        color_f = np.array(color, dtype=np.float32) / 255.0

        # compose overlay: for mask pixels, blend color
        mask_3 = np.repeat(bin_mask[:, :, None], 3, axis=2).astype(np.float32)
        overlay = (1.0 - mask_3 * alpha) * overlay + (mask_3 * alpha) * color_f

        # draw bbox (on original frame coords)
        bx, by, bw, bh = boxes[i]
        x1 = int(round(bx * scale))
        y1 = int(round(by * scale))
        x2 = int(round((bx + bw) * scale))
        y2 = int(round((by + bh) * scale))
        # clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), np.array(color_f).tolist(), 2)
        # label
        cv2.putText(overlay, f"id:{cls} {scores[i]:.2f}", (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    np.array([1.0, 1.0, 1.0]).tolist(), 1, cv2.LINE_AA)
        drawn += 1

    out_img = (overlay * 255.0).astype(np.uint8)
    return out_img, drawn


def run_cam(onnx_path: str, cam_index=0, use_gpu=False, max_frames=0):
    if ort is None:
        print("[ERR] onnxruntime not installed. Install onnxruntime or onnxruntime-gpu.")
        return

    sess = load_session(onnx_path, use_gpu)
    if sess is None:
        return

    # get input name and shapes
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    # We assume exported model uses 640x640 input; but we will resize to that
    model_size = 640

    # open capture
    cap = cv2.VideoCapture(int(cam_index) if str(cam_index).isdigit() else cam_index)
    if not cap.isOpened():
        print("[ERR] Failed to open camera:", cam_index)
        return

    print("[INFO] Camera opened. Press 'q' in window to quit (if GUI available).")
    frame_count = 0
    t0 = time.time()
    gui_ok = True
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERR] Frame read failed, exiting.")
                break
            h, w = frame.shape[:2]
            # pad to square (top-left aligned) — same as export expectation
            length = max(h, w)
            pad_img = np.zeros((length, length, 3), dtype=np.uint8)
            pad_img[0:h, 0:w] = frame
            # resize to model_size
            inp = cv2.resize(pad_img, (model_size, model_size))
            inp = inp.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, :].astype(np.float32)

            # run
            try:
                outputs = sess.run(None, {input_name: inp})
            except Exception as e:
                print("[ERR] onnxruntime run failed:", e)
                break

            # outputs: usually (det_out, mask_proto)
            # det_out shape could be (1, C, N) or (1, N, C); mask_proto (1, M, ph, pw)
            if len(outputs) < 2:
                print("[ERR] Unexpected number of outputs:", len(outputs))
                break
            det_out = outputs[0]
            mask_proto = outputs[1]

            # normalize det_out to shape (N, C) or (1, C, N) accepted by postprocess
            # Ultralytics export: det_out often shape (1, C, N)
            if det_out.ndim == 3 and det_out.shape[0] == 1 and det_out.shape[1] > det_out.shape[2]:
                # (1, C, N) -> transpose to (1, N, C) so our postprocess can handle both forms
                det_out = np.transpose(det_out, (0, 2, 1))

            out_frame, drawn = postprocess_and_draw(frame, det_out, mask_proto)
            # overlay FPS and info
            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / max(1e-6, elapsed)
            cv2.putText(out_frame, f"FPS: {fps:.1f}  Detected: {drawn}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)

            # try to show
            try:
                cv2.imshow("YOLO11-Seg", out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            except Exception as e:
                # no GUI; save frames periodically
                gui_ok = False
                if frame_count % 30 == 0:
                    out_path = Path.cwd() / f"seg_frame_{frame_count}.jpg"
                    cv2.imwrite(str(out_path), out_frame)
                    print(f"[INFO] Saved frame to {out_path}")

            if max_frames and frame_count >= max_frames:
                print("[INFO] Reached max frames. Exiting.")
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to ONNX segmentation model")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--use_gpu", action="store_true", help="Try to use CUDAExecutionProvider")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames (0=infinite)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_path = args.model
    if model_path is None:
        # search common names
        candidates = list(root.glob("**/yolo11*.onnx")) + list(root.glob("**/yolo*.onnx"))
        model_path = str(candidates[0]) if candidates else None

    if model_path is None or not Path(model_path).exists():
        print("[ERR] No ONNX model found. Use --model path/to/yolo11s-seg.onnx")
        return

    print("[INFO] Using model:", model_path)
    run_cam(model_path, cam_index=args.cam, use_gpu=args.use_gpu, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
'''

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding"""
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_segmentation(model_path, cam_index=0, use_gpu=False):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Providers: {sess.get_providers()}")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}")
        return

    print("[INFO] Press 'q' to quit")
    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img, r, dwdh = letterbox(frame, (640, 640))
        img_input = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # Inference
        outputs = sess.run(None, {input_name: img_input})
        preds, proto = outputs  # YOLOv11-seg outputs 2 tensors
        preds = preds[0]  # shape (116, 8400)
        proto = proto[0]  # shape (32, 160, 160)

        # --- POSTPROCESS ---
        num_masks = proto.shape[0]  # usually 32
        h, w = frame.shape[:2]

        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(80)]

        for det in preds.T:
            conf = det[4]
            if conf < 0.4:
                continue

            cls = int(np.argmax(det[5:85]))
            mask_coeff = det[85:]

            # Автокоррекция размерности
            if mask_coeff.shape[0] != num_masks:
                if mask_coeff.shape[0] < num_masks:
                    mask_coeff = np.pad(mask_coeff, (0, num_masks - mask_coeff.shape[0]))
                else:
                    mask_coeff = mask_coeff[:num_masks]

            cx, cy, bw, bh = det[:4]
            x1 = int((cx - bw / 2 - dwdh[0]) / r)
            y1 = int((cy - bh / 2 - dwdh[1]) / r)
            x2 = int((cx + bw / 2 - dwdh[0]) / r)
            y2 = int((cy + bh / 2 - dwdh[1]) / r)

            # Маска
            mask = np.dot(mask_coeff, proto.reshape(num_masks, -1))
            mask = sigmoid(mask).reshape(proto.shape[1:])
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = mask > 0.5

            # Морфологическая обработка
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            color = colors[cls % len(colors)]
            overlay = frame.copy()
            overlay[mask == 1] = color

            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls}:{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # FPS overlay
        frame_count += 1
        fps = frame_count / (time.time() - t0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("YOLO11 Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO11 segmentation ONNX model")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--use_gpu", action="store_true", help="Enable CUDA if available")
    args = parser.parse_args()

    run_segmentation(args.model, cam_index=args.cam, use_gpu=args.use_gpu)
