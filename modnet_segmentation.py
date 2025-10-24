import sys, os, time
sys.path.insert(0, r"C:\Users\4ekwk\Downloads\novosib_hack_t1\hackaton\MODNet\src")

import cv2
import torch
import numpy as np
from torchvision import transforms
from MODNet.src.models.modnet import MODNet

MODEL_PATH = r"C:\Users\4ekwk\Downloads\novosib_hack_t1\hackaton\models\modnet_photographic_portrait_matting.ckpt"
TS_PATH = r"C:\Users\4ekwk\Downloads\novosib_hack_t1\hackaton\modnet_cpu.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {DEVICE}")

# Load original model (PyTorch)
modnet = MODNet(backbone_pretrained=False)
modnet.to(DEVICE)
modnet.eval()

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
sd = {k.replace('module.', ''): v for k, v in sd.items()}
modnet.load_state_dict(sd)
print("[INFO] MODNet loaded!")

# Preprocessing
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# CLAHE + Gamma helpers
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def apply_gamma(img_rgb, gamma=1.5):
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_rgb, table)


# TorchScript Support
use_torchscript = False
ts_model = None

def ensure_torchscript(model, device, ts_path=TS_PATH):
    global ts_model, use_torchscript

    # Если файл уже есть — просто загружаем
    if os.path.exists(ts_path):
        ts_model = torch.jit.load(ts_path, map_location=device)
        ts_model.eval()
        use_torchscript = True
        print(f"[INFO] TorchScript loaded: {ts_path}")
        return ts_model

    print("[INFO] Exporting TorchScript (script mode)...")
    model_cpu = model.to('cpu').eval()
    with torch.no_grad():
        scripted = torch.jit.script(model_cpu)

    torch.jit.save(scripted, ts_path)
    print(f"[INFO] TorchScript saved: {ts_path}")

    ts_model = torch.jit.load(ts_path, map_location=device)
    ts_model.eval()
    use_torchscript = True
    model.to(device)
    return ts_model




cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("🚨 Camera not detected")

print("[INFO] Press:")
print(" Q - quit")
print(" C/c - CLAHE (once / toggle)")
print(" G/g - Gamma (once / toggle)")
print(" T - enable TorchScript (CPU)")

clahe_once = False
clahe_always = False
gamma_once = False
gamma_always = False

printed_structure = False
fps_start = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if clahe_once or clahe_always:
        rgb = apply_clahe_rgb(rgb)
    if gamma_once or gamma_always:
        rgb = apply_gamma(rgb)

    clahe_once = False
    gamma_once = False

    h, w = rgb.shape[:2]
    pw, ph = 512, 512
    resized = cv2.resize(rgb, (pw, ph))

    # ------- MODEL PREDICTION -------
    with torch.no_grad():
        if use_torchscript and ts_model is not None:
            inputs = to_tensor(resized).unsqueeze(0).to('cpu')
            try:
                outputs = ts_model(inputs, True)
            except TypeError:
                outputs = ts_model(inputs)
        else:
            inputs = to_tensor(resized).unsqueeze(0).to(DEVICE)
            outputs = modnet(inputs, True)

    # ------- SAFE MATTE EXTRACT -------
    matte = None

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        matte = outputs[2]  # ✅ MODNet pha
    elif isinstance(outputs, torch.Tensor):
        matte = outputs
    elif isinstance(outputs, dict):
        for k in ('pha', 'matte', 'pred_alpha', 'alpha'):
            if k in outputs:
                matte = outputs[k]
                break

    if matte is None:
        print("[WARN] No matte tensor detected — skipping")
        continue

    matte = matte[0, 0].cpu().numpy()
    matte = np.clip(matte, 0, 1)
    matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------- APPLY MATTING -------
    matte_3ch = np.repeat(matte[:, :, None], 3, axis=2)
    result = frame * matte_3ch + 255 * (1 - matte_3ch)
    result = result.astype(np.uint8)

    cv2.imshow("MODNet", result)

    # ------- FPS -------
    frame_count += 1
    if time.time() - fps_start >= 1:
        print(f"FPS: {frame_count}")
        frame_count = 0
        fps_start = time.time()

    # ------- KEY CONTROLS -------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        clahe_once = True
    elif key == ord('C'):
        clahe_always = not clahe_always
        print(f"CLAHE always: {clahe_always}")
    elif key == ord('g'):
        gamma_once = True
    elif key == ord('G'):
        gamma_always = not gamma_always
        print(f"Gamma always: {gamma_always}")
    elif key == ord('t'):
        ensure_torchscript(modnet, 'cpu')
        print("[INFO] TorchScript ready!")


cap.release()
cv2.destroyAllWindows()
print("\n✅ Exit complete")
