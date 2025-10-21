import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from PIL import Image, ImageSequence
import time
import os
"""
Improved background replacement / person segmentation (test2.py)

Includes:
- configuration via .env (see TEMPLATE_ENV below)
- fallback logic: try ONNX (onnxruntime-gpu) or PyTorch model if configured and available; otherwise use MediaPipe
- temporal smoothing (EMA), mask history, distance-transform feathering for clean edges
- optional guided/bilateral filtering (if opencv contrib available)
- keyboard controls to toggle GPU mode, hard/soft mask, CLAHE, gif play/pause, debug windows
- safe fallbacks so script runs even when optional libraries are missing

Save a .env next to this file (or edit TEMPLATE_ENV below)

TEMPLATE_ENV (put exactly into a file named .env in the same folder):

# ---------- .env sample ----------
# Camera index (int)
CAP_INDEX=0
# Paths to backgrounds separated by semicolon (GIFs or images allowed)
BACKGROUND_FILES=C:/Users/4ekwk/Downloads/7beN-2699045384.gif
# Mask smoothing length for history averaging (int)
MASK_HISTORY_LEN=5
# Hard threshold [0..1] for binarization
HARD_THRESH=0.5
# Apply CLAHE on V channel (True/False)
APPLY_CLAHE=True
# Morphology kernel size (odd int)
MORPH_KERNEL=3
# Use GPU if possible (True/False). This will try ONNXRuntime-GPU or PyTorch if an external model path is configured.
USE_GPU=False
# Path to optional ONNX segmentation model (if empty, MediaPipe will be used)
ONNX_MODEL_PATH=
# Use stronger temporal smoothing alpha (0..1). Lower = slower smoothing.
EMA_ALPHA=0.4
# Edge feathering radius in pixels
FEATHER_RADIUS=20
# ----------------------------------

"""

from pathlib import Path
import os
import cv2
import numpy as np
import time
from collections import deque
from PIL import Image, ImageSequence
from dotenv import load_dotenv

# --- Load configuration from .env with sensible defaults ---
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env not found at {env_path}. Using defaults; create a .env to customize.")

# helper to read env booleans and ints
def get_env(key, default=None, cast=str):
    v = os.getenv(key)
    if v is None:
        return default
    try:
        if cast is bool:
            return v.strip().lower() in ('1','true','yes','on')
        return cast(v)
    except Exception:
        return default

# configuration
CAP_INDEX = get_env('CAP_INDEX', 0, int)
BACKGROUND_FILES = [p.strip() for p in os.getenv('BACKGROUND_FILES', '').split(';') if p.strip()] or [str(Path.cwd() / 'bg_default.jpg')]
MASK_HISTORY_LEN = get_env('MASK_HISTORY_LEN', 5, int)
HARD_THRESH = get_env('HARD_THRESH', 0.5, float)
APPLY_CLAHE = get_env('APPLY_CLAHE', True, bool)
MORPH_KERNEL = (get_env('MORPH_KERNEL', 3, int),) * 2
USE_GPU = get_env('USE_GPU', False, bool)
ONNX_MODEL_PATH = os.getenv('ONNX_MODEL_PATH', '')
EMA_ALPHA = get_env('EMA_ALPHA', 0.4, float)
FEATHER_RADIUS = get_env('FEATHER_RADIUS', 20, int)

# runtime toggles
use_hard = False
draw_debug = True
show_mask = False
apply_clahe = APPLY_CLAHE

# --- Try to load optional accelerated segmentation (ONNXRuntime or PyTorch) ---
onnx_runtime = None
onnx_sess = None
torch = None
torch_model = None
use_accel = False

if USE_GPU and ONNX_MODEL_PATH:
    try:
        import onnxruntime as ort
        # try GPU provider first, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        onnx_sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=providers)
        onnx_runtime = ort
        use_accel = True
        print('Using ONNXRuntime with model', ONNX_MODEL_PATH)
    except Exception as e:
        print('ONNXRuntime GPU model load failed, will fallback to MediaPipe. Error:', e)

# Optional PyTorch model path could be added similarly; keep placeholder
if USE_GPU and not use_accel:
    try:
        import torch
        torch = torch
        # If user provided a PyTorch model load path, try to load it here (not implemented by default)
    except Exception:
        torch = None

# --- MediaPipe fallback (always available) ---
try:
    import mediapipe as mp
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    print('MediaPipe loaded as fallback segmentation')
except Exception as e:
    mp_segment = None
    print('MediaPipe not available; segmentation will fail without an external ONNX/PyTorch model. Error:', e)

# --- Utilities: background loader (supports GIF) ---

def load_bg(path):
    ext = Path(path).suffix.lower()
    if ext == '.gif':
        gif = Image.open(path)
        frames = [cv2.cvtColor(np.array(f.convert('RGB')), cv2.COLOR_RGB2BGR) for f in ImageSequence.Iterator(gif)]
        return {'type':'gif', 'frames':frames, 'pos':0, 'playing':True}
    else:
        img = cv2.imread(path)
        if img is None:
            # placeholder plain color
            img = np.full((480,640,3), (50,50,50), dtype=np.uint8)
        return {'type':'img', 'img':img}

bg_items = [load_bg(p) for p in BACKGROUND_FILES]
cur_bg = 0

# --- mask processing utilities ---

mask_history = deque(maxlen=MASK_HISTORY_LEN)
ema_mask = None

# edge feathering by distance transform
def feather_mask(mask, radius=FEATHER_RADIUS):
    # mask expected float32 0..1
    # produce soft edge by distance transform of binary mask
    b = (mask > 0.5).astype(np.uint8)
    if b.sum() == 0 or b.sum() == b.size:
        return mask
    dist_in = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    sdf = dist_out - dist_in
    # normalize and map to 0..1 with sigmoid-like curve
    soft = 1.0 / (1.0 + np.exp(-sdf / max(1.0, radius/5.0)))
    return np.clip(soft, 0.0, 1.0)

# guided filter if available
use_guided = False
try:
    from cv2.ximgproc import guidedFilter
    use_guided = True
except Exception:
    use_guided = False

# postprocess mask: morphological, temporal EMA, blur, feathering, guided/bilateral

def postprocess_mask(raw_mask, ema_mask_local=None):
    # raw_mask: float32 in [0..1], shape (h,w)
    m = raw_mask.copy()
    # morphology: open then close to remove speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    # EMA temporal smoothing (faster than naive average)
    global ema_mask
    if ema_mask_local is None:
        alpha = EMA_ALPHA
    else:
        alpha = ema_mask_local
    if ema_mask is None:
        ema_mask = m
    else:
        ema_mask = alpha * m + (1 - alpha) * ema_mask

    m = ema_mask

    # guided or bilateral smoothing to preserve edges
    if use_guided:
        # requires color image as guidance, will be applied in caller with frame
        pass
    else:
        # soft bilateral-like effect by applying small Gaussian + median
        m = cv2.GaussianBlur(m, (7,7), 0)
        m = cv2.medianBlur((m*255).astype(np.uint8), 5).astype(np.float32)/255.0

    # edge feathering
    m = feather_mask(m, radius=FEATHER_RADIUS)

    return np.clip(m, 0.0, 1.0)

# helper to run segmentation by available backend

def run_segmentation(frame_bgr):
    # return float32 mask 0..1 shape (h,w)
    h,w = frame_bgr.shape[:2]
    # 1) ONNXRuntime path (if enabled)
    if onnx_sess is not None:
        try:
            inp = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # model-specific preprocessing: assume model wants 320x320 or 256x256; try to infer
            # For robust behavior, read expected input shape from session
            inp_name = onnx_sess.get_inputs()[0].name
            shape = onnx_sess.get_inputs()[0].shape
            # fallback size
            target_h = int(shape[2]) if len(shape) >= 4 and shape[2] is not None else 256
            target_w = int(shape[3]) if len(shape) >= 4 and shape[3] is not None else 256
            small = cv2.resize(inp, (target_w, target_h))
            x = small.astype(np.float32) / 255.0
            # shape -> NCHW
            if x.ndim == 3:
                x = np.transpose(x, (2,0,1))[None, ...]
            out = onnx_sess.run(None, {inp_name: x})
            # take first output and resize back
            raw = np.squeeze(out[0])
            if raw.ndim == 3:
                raw = raw[0]
            mask = cv2.resize(raw, (w,h))
            mask = (mask - mask.min()) / max(1e-6, (mask.max()-mask.min()))
            return mask.astype(np.float32)
        except Exception as e:
            print('ONNXRuntime inference failed, falling back to MediaPipe. Error:', e)

    # 2) PyTorch path (placeholder)
    if torch is not None and torch_model is not None:
        try:
            # user-supplied torch model should return HxW mask 0..1
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320,320))
            t = torch.from_numpy(img/255.).permute(2,0,1).unsqueeze(0).float().cuda()
            with torch.no_grad():
                out = torch_model(t)
            mask = out.squeeze().cpu().numpy()
            mask = cv2.resize(mask, (w,h))
            return mask.astype(np.float32)
        except Exception as e:
            print('Torch model inference failed:', e)

    # 3) MediaPipe fallback
    if mp_segment is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = mp_segment.process(rgb)
        if res is None or res.segmentation_mask is None:
            return np.zeros((h,w), dtype=np.float32)
        mask = res.segmentation_mask.astype(np.float32)
        # MediaPipe provides smaller mask; resize if necessary
        if mask.shape != (h,w):
            mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_LINEAR)
        return mask

    # if nothing available
    return np.zeros((h,w), dtype=np.float32)

# --- Video capture loop ---
cap = cv2.VideoCapture(CAP_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Can't open camera index {CAP_INDEX}")

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

last_time = time.time()
frame_count = 0
print("Controls: 1/2/... switch backgrounds, m toggle mask soft/hard, g play/pause gif, b toggle CLAHE, p toggle show mask, v toggle GPU attempt, q/Esc quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h,w = frame.shape[:2]

    frame_proc = frame.copy()
    # optional CLAHE on V channel
    if apply_clahe:
        hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        v = clahe.apply(v)
        hsv[:,:,2] = v
        frame_proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # run segmentation
    raw_mask = run_segmentation(frame_proc)

    # postprocess (morph, EMA, blur, feather)
    processed = postprocess_mask(raw_mask)

    # if guided filter available, refine mask with color guidance
    if use_guided:
        try:
            guided = guidedFilter(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB), (processed*255).astype(np.uint8), 8, 1e-6)
            processed = guided.astype(np.float32)/255.0
        except Exception:
            pass

    # choose hard or soft
    if use_hard:
        m = (processed > HARD_THRESH).astype(np.uint8)
        m = cv2.medianBlur(m, 5)
        m = np.expand_dims(m, 2)
    else:
        m = cv2.GaussianBlur(processed, (11,11), 0)
        m = np.expand_dims(m, 2)

    # select background
    bg_item = bg_items[cur_bg]
    if bg_item['type'] == 'gif':
        frames = bg_item['frames']
        idx = bg_item['pos']
        bg_frame = frames[idx % len(frames)]
        if bg_item.get('playing', True):
            bg_item['pos'] = (bg_item['pos'] + 1) % len(frames)
    else:
        bg_frame = bg_item['img']

    bg_resized = cv2.resize(bg_frame, (w,h))

    # composite
    composed = (frame.astype(np.float32) * m + bg_resized.astype(np.float32) * (1 - m)).astype(np.uint8)

    # debug overlays
    if draw_debug:
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            last_time = now
            frame_count = 0
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = ''
        info = f"{'HARD' if use_hard else 'SOFT'} | CLAHE:{'ON' if apply_clahe else 'OFF'} | bg:{cur_bg+1}/{len(bg_items)} {fps_text} | GPU accel:{'ON' if use_accel else 'OFF'}"
        cv2.putText(composed, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Enhanced Background Replacement', composed)
    if show_mask:
        vis_mask = (processed*255).astype(np.uint8)
        vis_mask_col = cv2.applyColorMap(vis_mask, cv2.COLORMAP_JET)
        cv2.imshow('Mask (debug)', vis_mask_col)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
        idx = int(chr(key)) - 1
        if 0 <= idx < len(bg_items):
            cur_bg = idx
    elif key == ord('m'):
        use_hard = not use_hard
    elif key == ord('g'):
        if bg_items[cur_bg]['type'] == 'gif':
            bg_items[cur_bg]['playing'] = not bg_items[cur_bg].get('playing', True)
    elif key == ord('b'):
        apply_clahe = not apply_clahe
    elif key == ord('p'):
        show_mask = not show_mask
    elif key == ord('v'):
        # toggle acceleration attempt: if user toggles, we flip trying to use onnx/pytorch if configured
        use_accel = not use_accel
        print('GPU accel attempt toggled ->', use_accel)

cap.release()
cv2.destroyAllWindows()

# ---------------- Suggestions to further improve ----------------
# * Use a dedicated alpha-matting network (U^2-Net, MODNet, BGMatting / RVM) to get fine hair details.
#   Run it with Torch/CUDA or convert to ONNX+TensorRT for low-latency on GPU.
# * Use a small fast student model for real-time (MobileNet-based matting) when GPU is present.
# * Use Temporal consistency networks (e.g., optical-flow guided smoothing) to prevent mask jitter.
# * For streamer-grade background replacement, consider combining:
#     - a realtime fast segmentation net for coarse mask
#     - a matting network applied only to ROI (head/shoulders) every N frames for hair/edge detail
# * For CPU-only machines, use quantized ONNX models or OpenVINO to accelerate inference.
# * Consider exposing more tunables (feather radius, EMA alpha, bilateral params) in the UI or .env.
# * If you want virtual green-screen replacement with color spill suppression, add color transfer and edge-aware despill.

# END

# ---------- Настройки (подкорректируй пути) ----------
background_files = [
    r"C:\Users\4ekwk\Downloads\7beN-2699045384.gif"
]
cap_index = 0            # индекс камеры
mask_history_len = 5     # усреднение по N кадрам
hard_thresh = 0.5        # порог для бинаризации
apply_clahe = True       # авто улучшение освещённости
morph_kernel = (3,3)     # ядро для морфологии
draw_debug = True

# ------------------------------------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(cap_index)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

# Загрузка фонов (поддержка gif)
def load_bg(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gif",):
        gif = Image.open(path)
        frames = [cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR)
                  for f in ImageSequence.Iterator(gif)]
        return {"type":"gif", "frames": frames, "pos":0, "playing":True}
    else:
        img = cv2.imread(path)
        if img is None:
            # создаём цветной фон, если не найден
            img = np.full((480,640,3), (50,50,50), dtype=np.uint8)
        return {"type":"img", "img": img}

bg_items = [load_bg(p) for p in background_files]
cur_bg = 0

mask_history = deque(maxlen=mask_history_len)
use_hard = False

# CLAHE для V-канала
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

last_time = time.time()
frame_count = 0

print("Controls: 1/2/3 switch backgrounds, m toggle mask soft/hard, g play/pause gif, b toggle CLAHE, q/Esc quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # 1) опционально улучшаем освещённость (CLAHE на V)
    frame_proc = frame.copy()
    if apply_clahe:
        hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        v = clahe.apply(v)
        hsv[:,:,2] = v
        frame_proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    result = segment.process(rgb)
    mask = result.segmentation_mask  # float32 [0..1]

    # 2) базовая морфология (убираем шум)
    mask = cv2.erode(mask, np.ones(morph_kernel, np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

    # 3) сглаживание по времени
    mask_history.append(mask)
    avg_mask = np.mean(mask_history, axis=0)

    # 4) для вывода: либо soft (градиент), либо hard (0/1)
    if use_hard:
        m = (avg_mask > hard_thresh).astype(np.uint8)
        # можно немного расширить & затем сузить, чтобы убрать "хвосты"
        m = cv2.medianBlur(m, 5)
        m = np.expand_dims(m, axis=2)
    else:
        m = cv2.GaussianBlur(avg_mask, (15,15), 0)
        m = np.expand_dims(m, axis=2)
        m = np.clip(m, 0, 1)

    # 5) подбираем фон
    bg_item = bg_items[cur_bg]
    if bg_item["type"] == "gif":
        frames = bg_item["frames"]
        idx = bg_item["pos"]
        bg_frame = frames[idx % len(frames)]
        if bg_item.get("playing", True):
            bg_item["pos"] = (bg_item["pos"] + 1) % len(frames)
    else:
        bg_frame = bg_item["img"]

    bg_resized = cv2.resize(bg_frame, (w, h))

    # 6) комбинируем
    # m.shape == (h,w,1); frame is uint8
    # ensure float computation then back to uint8
    composed = (frame.astype(np.float32) * (1 - m) + bg_resized.astype(np.float32) * m).astype(np.uint8)

    # 7) отрисовка UI
    if draw_debug:
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            last_time = now
            frame_count = 0
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = ""
        info = f"{'HARD' if use_hard else 'SOFT'} mask | CLAHE: {'ON' if apply_clahe else 'OFF'} | bg:{cur_bg+1}/{len(bg_items)} {fps_text}"
        cv2.putText(composed, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Enhanced Background Replacement", composed)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or q
        break
    elif key in (ord('1'), ord('2'), ord('3')):
        idx = int(chr(key)) - 1
        if 0 <= idx < len(bg_items):
            cur_bg = idx
    elif key == ord('m'):
        use_hard = not use_hard
    elif key == ord('g'):
        # toggle gif play/pause if current bg is gif
        if bg_items[cur_bg]["type"] == "gif":
            bg_items[cur_bg]["playing"] = not bg_items[cur_bg].get("playing", True)
    elif key == ord('b'):
        apply_clahe = not apply_clahe

cap.release()
cv2.destroyAllWindows()
