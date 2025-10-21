import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from PIL import Image, ImageSequence
import time
import os

# ---------- Настройки ----------
background_files = [
    r"D:\novosib_hack_t1_shabbat\hackaton\Shingeki-no-Kyojin-Anime-фэндомы-8027242.jpeg"
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
    if ext == ".gif":
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
invert_mode = False  # 🔁 False = заменяем фон, True = заменяем человека

# CLAHE для V-канала
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

last_time = time.time()
frame_count = 0

print("Controls:")
print("  1/2/3 - переключить фон")
print("  m - жесткая/мягкая маска")
print("  g - пауза/воспроизведение gif")
print("  b - включить/выключить CLAHE")
print("  r - 🔁 инвертировать режим (фон ↔ человек)")
print("  q/Esc - выйти\n")

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

    # 2) сегментация
    rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    result = segment.process(rgb)
    mask = result.segmentation_mask  # float32 [0..1]

    # 3) морфология (убираем шум)
    mask = cv2.erode(mask, np.ones(morph_kernel, np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

    # 4) сглаживание по времени
    mask_history.append(mask)
    avg_mask = np.mean(mask_history, axis=0)

    # 5) подготовка маски
    if use_hard:
        m = (avg_mask > hard_thresh).astype(np.uint8)
        m = cv2.medianBlur(m, 5)
        m = np.expand_dims(m, axis=2)
    else:
        m = cv2.GaussianBlur(avg_mask, (15,15), 0)
        m = np.expand_dims(m, axis=2)
        m = np.clip(m, 0, 1)

    # 6) выбираем фон (или gif)
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

    # 7) комбинируем
    if invert_mode:
        # 🔁 РЕЖИМ "заменяем человека" (фон оставляем)
        composed = (frame.astype(np.float32) * (1 - m) + bg_resized.astype(np.float32) * m).astype(np.uint8)
    else:
        # стандартный режим — "заменяем фон"
        composed = (frame.astype(np.float32) * m + bg_resized.astype(np.float32) * (1 - m)).astype(np.uint8)

    # 8) отрисовка UI
    if draw_debug:
        frame_count += 1
        now = time.time()
        fps_text = ""
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            last_time = now
            frame_count = 0
            fps_text = f"FPS: {fps:.1f}"

        info = f"{'HARD' if use_hard else 'SOFT'} | CLAHE:{'ON' if apply_clahe else 'OFF'} | bg:{cur_bg+1}/{len(bg_items)} | {'INVERT' if invert_mode else 'NORMAL'} {fps_text}"
        cv2.putText(composed, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Enhanced Background Replacement", composed)

    # 9) клавиши управления
    key = cv2.waitKey(30) & 0xFF
    if key in (27, ord('q')):  # Esc / q
        break
    elif key in (ord('1'), ord('2'), ord('3')):
        idx = int(chr(key)) - 1
        if 0 <= idx < len(bg_items):
            cur_bg = idx
    elif key == ord('m'):
        use_hard = not use_hard
    elif key == ord('g'):
        if bg_items[cur_bg]["type"] == "gif":
            bg_items[cur_bg]["playing"] = not bg_items[cur_bg].get("playing", True)
    elif key == ord('b'):
        apply_clahe = not apply_clahe
    elif key == ord('r'):
        invert_mode = not invert_mode
        print(f"Режим переключён: {'заменяем человека' if invert_mode else 'заменяем фон'}")

cap.release()
cv2.destroyAllWindows()
