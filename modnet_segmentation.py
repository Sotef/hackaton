import sys, os, time
sys.path.insert(0, r"C:\Users\4ekwk\Downloads\novosib_hack_t1\hackaton\MODNet\src")

import cv2
import torch
# Для CUDA 11.7: torch==2.1.0+cu117 torchvision==0.16.0+cu117 (с индексом колеса)

import numpy as np
from torchvision import transforms
from MODNet.src.models.modnet import MODNet

MODEL_PATH = r"C:\Users\4ekwk\Downloads\novosib_hack_t1\hackaton\models\modnet_photographic_portrait_matting.ckpt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] device: {DEVICE}")

modnet = MODNet(backbone_pretrained=False)
modnet.to(DEVICE)
modnet.eval()

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt
sd = {k.replace('module.', ''): v for k, v in sd.items()}
modnet.load_state_dict(sd)
print("[INFO] model loaded")

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
print("[INFO] press Q to quit")

# For one-time debug print of outputs structure
printed_structure = False

# FPS tracking
frame_count = 0
fps = 0.0
fps_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ph, pw = 512, 512
    im = cv2.resize(rgb, (pw, ph))
    im_tensor = to_tensor(im).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = modnet(im_tensor, True)

    # One-time debug print
    if not printed_structure:
        printed_structure = True
        print("DEBUG outputs type:", type(outputs))
        if isinstance(outputs, dict):
            print("DEBUG outputs keys:", list(outputs.keys()))
        elif isinstance(outputs, (list, tuple)):
            print("DEBUG outputs len:", len(outputs))

    # Extract first tensor-like output (matte)
    matte = None
    if isinstance(outputs, torch.Tensor):
        matte = outputs
    elif isinstance(outputs, (list, tuple)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                matte = o
                break
    elif isinstance(outputs, dict):
        for k in ('matte', 'alpha', 'pred_alpha', 'pha', 'pha_map'):
            if k in outputs and isinstance(outputs[k], torch.Tensor):
                matte = outputs[k]
                break
        if matte is None:
            for v in outputs.values():
                if isinstance(v, torch.Tensor):
                    matte = v
                    break

    if matte is None:
        raise RuntimeError("Model returned no tensor (None). Проверьте сигнатуру MODNet и аргументы вызова.")

    matte = matte.squeeze().cpu().numpy()
    matte = np.clip(matte, 0.0, 1.0)
    matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

    matte_3 = np.repeat(matte[:, :, None], 3, axis=2).astype(np.float32)
    frame_f = frame.astype(np.float32) / 255.0
    bg_f = np.ones_like(frame_f, dtype=np.float32)
    result = (frame_f * matte_3 + bg_f * (1 - matte_3)) * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imshow("MODNet", result)

    # FPS counting and single-line console update every second
    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = time.time()
        # print in one line, carriage return without new line
        print(f"\rFPS: {fps:.2f}", end="", flush=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()  # finish FPS line
