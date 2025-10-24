"""WebSocket segmentation server for front-end integration.

Protocol (JSON, base64 frames):
 - client -> server messages (JSON):
   {"type":"start","meta":{...}} | {"type":"frame","data":"<base64 jpeg>","meta":{...}} | {"type":"stop"}
 - server -> client responses (JSON):
   {"type":"result","frame_id":"...","data":"<base64 png>","format":"png","kind":"mask|composite","meta":{...}}

Run (PowerShell):
  & .\.venv\Scripts\Activate.ps1
  uvicorn ws_server:app --host 0.0.0.0 --port 8000

Connect from browser: ws://localhost:8000/segment

Notes:
 - Uses MediaPipe selfie segmentation (lightweight, CPU-friendly).
 - Messages use JSON text with base64-encoded PNG/JPEG frames for simplicity.
 - For production/throughput, switch to binary frames (ArrayBuffer) or multipart streaming.
"""
import base64
import json
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import mediapipe as mp

app = FastAPI()

# Shared MediaPipe segmenter (reused across connections)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
SEGMENT = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# defaults
DEFAULT_MASK_HISTORY = 3
DEFAULT_HARD_THRESH = 0.5


def decode_b64_image(b64: str) -> Optional[np.ndarray]:
    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def encode_png_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode('ascii')


async def process_frame(frame: np.ndarray, state: dict) -> dict:
    """Process single BGR frame and return dict with data, kind and meta."""
    h, w = frame.shape[:2]

    # optional CLAHE
    if state.get('apply_clahe', False):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        clahe = state.setdefault('clahe', cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)))
        v = clahe.apply(v)
        hsv[:, :, 2] = v
        frame_proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        frame_proc = frame

    # segmentation
    rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    result = SEGMENT.process(rgb)
    mask = result.segmentation_mask.astype(np.float32) if (result and result.segmentation_mask is not None) else np.zeros((h, w), dtype=np.float32)

    # morphology
    morph_kernel = state.get('morph_kernel', (3, 3))
    mask = cv2.erode(mask, np.ones(morph_kernel, np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    # temporal smoothing
    mh = state.setdefault('mask_history', deque(maxlen=state.get('mask_history_len', DEFAULT_MASK_HISTORY)))
    mh.append(mask)
    avg_mask = np.mean(mh, axis=0)

    # soft or hard mask
    if state.get('use_hard', False):
        m = (avg_mask > state.get('hard_thresh', DEFAULT_HARD_THRESH)).astype(np.uint8) * 255
        m = cv2.medianBlur(m, 5)
        m_3 = np.expand_dims(m, axis=2)
        m_norm = m_3.astype(np.float32) / 255.0
    else:
        m = cv2.GaussianBlur(avg_mask, (15, 15), 0)
        m_norm = np.expand_dims(np.clip(m, 0, 1), axis=2)

    # background (solid gray by default)
    bg = state.get('bg_img')
    if bg is None:
        bg = np.full((h, w, 3), 127, dtype=np.uint8)
    else:
        bg = cv2.resize(bg, (w, h))

    invert = state.get('invert_mode', False)
    if invert:
        composed = (frame.astype(np.float32) * (1 - m_norm) + bg.astype(np.float32) * m_norm).astype(np.uint8)
    else:
        composed = (frame.astype(np.float32) * m_norm + bg.astype(np.float32) * (1 - m_norm)).astype(np.uint8)

    ret_mode = state.get('return', 'mask')
    if ret_mode == 'mask':
        mask_out = (m_norm[:, :, 0] * 255).astype(np.uint8)
        data_b64 = encode_png_b64(mask_out)
        kind = 'mask'
        fmt = 'png'
    elif ret_mode == 'composite':
        data_b64 = encode_png_b64(composed)
        kind = 'composite'
        fmt = 'png'
    elif ret_mode == 'rgba_png':
        bgr = composed
        alpha = (m_norm[:, :, 0] * 255).astype(np.uint8)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        data_b64 = encode_png_b64(rgba)
        kind = 'rgba'
        fmt = 'png'
    else:
        data_b64 = encode_png_b64(composed)
        kind = 'composite'
        fmt = 'png'

    meta = {'latency_ms': int((time.time() - state.get('_t0', time.time())) * 1000), 'orig_width': w, 'orig_height': h}
    return {'data': data_b64, 'format': fmt, 'kind': kind, 'meta': meta}


@app.websocket('/segment')
async def websocket_segment(ws: WebSocket):
    await ws.accept()
    state = {
        'mask_history_len': DEFAULT_MASK_HISTORY,
        'hard_thresh': DEFAULT_HARD_THRESH,
        'apply_clahe': False,
        'use_hard': False,
        'invert_mode': False,
        'return': 'mask',
        '_t0': time.time(),
    }

    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except Exception:
                await ws.send_text(json.dumps({'type': 'error', 'message': 'invalid json', 'code': 400}))
                continue

            mtype = msg.get('type')
            if mtype == 'ping':
                await ws.send_text(json.dumps({'type': 'status', 'clients': 1, 'fps': 0}))
                continue
            if mtype == 'config':
                cfg = msg.get('config', {}) or msg.get('meta', {})
                # apply basic keys
                for k, v in cfg.items():
                    if k in ('apply_clahe', 'use_hard', 'invert_mode'):
                        state[k] = bool(v)
                    elif k == 'return':
                        state['return'] = v
                await ws.send_text(json.dumps({'type': 'status', 'message': 'config updated', 'config': cfg}))
                continue
            if mtype == 'start':
                await ws.send_text(json.dumps({'type': 'status', 'message': 'started'}))
                continue
            if mtype == 'stop':
                await ws.send_text(json.dumps({'type': 'status', 'message': 'stopped'}))
                continue
            if mtype == 'frame':
                frame_b64 = msg.get('data')
                frame_id = msg.get('frame_id', str(time.time()))
                img = decode_b64_image(frame_b64)
                if img is None:
                    await ws.send_text(json.dumps({'type': 'error', 'message': 'invalid image data', 'code': 400}))
                    continue
                state['_t0'] = time.time()
                res = await process_frame(img, state)
                out = {'type': 'result', 'frame_id': frame_id, 'data': res['data'], 'format': res['format'], 'kind': res['kind'], 'meta': res['meta']}
                await ws.send_text(json.dumps(out))
                continue

            await ws.send_text(json.dumps({'type': 'error', 'message': 'unknown message type', 'code': 400}))

    except WebSocketDisconnect:
        return


@app.get('/')
def index():
    return HTMLResponse('<html><body><h3>WebSocket segmentation server running. Connect to ws://localhost:8000/segment</h3></body></html>')
