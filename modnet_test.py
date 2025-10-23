import os
import cv2
import torch
import numpy as np
from urllib.request import urlretrieve
from torch import nn

# === 1. Загрузка весов ===
MODEL_PATH = "models/modnet_photographic_portrait_matting.ckpt"
os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("[INFO] Скачиваю веса MODNet...")
    urlretrieve(
        "https://github.com/ZHKKKe/MODNet/releases/download/v1/modnet_photographic_portrait_matting.ckpt",
        MODEL_PATH
    )

# === 2. Определение модели ===
class MODNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(MODNet, self).__init__()
        from torchvision.models.mobilenetv2 import mobilenet_v2
        self.backbone = mobilenet_v2(pretrained=backbone_pretrained).features
        self.backbone[0][0].stride = (1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 320, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(320, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.conv(feat)
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear')
        return out

# === 3. Инициализация ===
device = 'cpu'
model = MODNet().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# === 4. Запуск камеры ===
cap = cv2.VideoCapture(0)
print("[INFO] Нажмите Q для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(t.to(device))
    mask = pred[0][0].cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Мягкое выделение
    result = frame.copy()
    result[:, :, 0] = result[:, :, 0] * mask
    result[:, :, 1] = result[:, :, 1] * mask
    result[:, :, 2] = result[:, :, 2] * mask

    cv2.imshow("MODNet Human Matting", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
