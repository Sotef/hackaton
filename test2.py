"""
GPU-ready demo pipeline for:
 - person segmentation (MODNet preferred, Mediapipe fallback)
 - clothes detection (YOLOv8 via ultralytics)
 - style classification (OpenCLIP / CLIP zero-shot)
 - color palette extraction (OpenCV + KMeans)
 - aesthetic scoring (optional LAION predictor if available)

This script is intended as a *local* prototype (not a web app) that runs on a machine
with a GPU and the appropriate model checkpoints installed.

How to use (example):
  python gpu_fashion_pipeline.py --source 0 --use_gpu --modnet --yolov8

Notes:
 - The script tries to import optional dependencies and will run in a degraded mode
   (mocked behavior) if they are not present. This makes it easy to test parts
   of the pipeline without all large checkpoints.
 - You must download model checkpoints manually (instructions below). The script
   expects them in ./models/ by default.

Recommended pip installs:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # match your CUDA
  pip install opencv-python-headless pillow numpy scikit-learn ultralytics open-clip-torch mediapipe
  # optionally: LAION aesthetic predictor dependencies if you want that component

Checkpoints (place into ./models/):
  - MODNet checkpoint: modnet_photographic_portrait_matting.ckpt  (from ZHKKKe/MODNet repo)
  - YOLOv8 clothes weights: yolov8n-clothes.pt (user/copies)
  - OpenCLIP: will be loaded by open_clip if available (from huggingface cache)

"""

import argparse
import os
import time
import json
from collections import deque

import cv2
import numpy as np

# Optional heavy deps
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    # ultralytics provides YOLOv8 interface
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except Exception:
    OPENCLIP_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------- Configuration -----------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Segmenter options
MODNET_CKPT = os.path.join(MODEL_DIR, "modnet_photographic_portrait_matting.ckpt")
YOLOV8_CKPT = os.path.join(MODEL_DIR, "yolov8n-clothes.pt")

# ----------------- Utilities ---------------------

def rgb_to_hex(c):
    return '#%02x%02x%02x' % (int(c[2]), int(c[1]), int(c[0]))

# ----------------- Segmentation (MODNet) ----------
class ModnetSegmenter:
    """Wrapper for MODNet. If MODNet code is not installed, the class will
    provide a fallback that uses Mediapipe (CPU) if available, or a simple
    coarse segmentation.
    """
    def __init__(self, device='cuda'):
        self.device = device if TORCH_AVAILABLE else 'cpu'
        self.ready = False
        self.model = None

        # Try to import MODNet only when requested
        try:
            from modnet import MODNet  # local modnet.py from original repo
            if TORCH_AVAILABLE and os.path.exists(MODNET_CKPT):
                self.model = MODNet(backbone_pretrained=False)
                ckpt = torch.load(MODNET_CKPT, map_location='cpu')
                self.model.load_state_dict(ckpt)
                if device.startswith('cuda') and torch.cuda.is_available():
                    self.model.to(device)
                self.model.eval()
                self.ready = True
            else:
                print('[ModnetSegmenter] MODNet checkpoint not found or torch unavailable -> fallback')
        except Exception:
            # If modnet isn't present, try Mediapipe
            if MEDIAPIPE_AVAILABLE:
                mp_selfie_seg = mp.solutions.selfie_segmentation
                self.mp_segment = mp_selfie_seg.SelfieSegmentation(model_selection=1)
                self.ready = True
                print('[ModnetSegmenter] Using Mediapipe fallback segmentation')
            else:
                print('[ModnetSegmenter] No MODNet and no Mediapipe -> will use coarse threshold fallback')

    def segment(self, frame):
        """Return float mask [0..1] same spatial size as frame."""
        h, w = frame.shape[:2]
        if self.model is not None:
            # run MODNet pipeline (expects PIL-like normalized input). We'll do a simple forward.
            import torchvision.transforms as T
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (512,512))
            im_t = T.ToTensor()(im).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # modnet returns tuple, mock usage here (real repo has specific api)
                matte = self.model(im_t, True)[0][0][0].cpu().numpy()
            matte = cv2.resize(matte, (w,h))
            matte = np.clip(matte, 0, 1)
            return matte
        elif MEDIAPIPE_AVAILABLE and hasattr(self, 'mp_segment'):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_segment.process(rgb)
            if res.segmentation_mask is None:
                return np.zeros((h,w), dtype=np.float32)
            return cv2.resize(res.segmentation_mask, (w,h))
        else:
            # coarse fallback: background subtraction-ish by brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            norm = (gray.astype(np.float32) - gray.mean()) / (gray.std() + 1e-6)
            mask = (norm < 1.0).astype(np.float32)
            return mask

# ----------------- Clothes detection (YOLOv8) -------
class ClothesDetector:
    def __init__(self, weights_path=None, device='cuda:0'):
        self.device = device
        self.model = None
        if ULTRALYTICS_AVAILABLE and weights_path and os.path.exists(weights_path):
            try:
                self.model = YOLO(weights_path)
                print('[ClothesDetector] Loaded YOLOv8 weights')
            except Exception as e:
                print('[ClothesDetector] Failed to load YOLOv8:', e)
                self.model = None
        else:
            print('[ClothesDetector] YOLOv8 not available or weights missing -> detector disabled')

    def detect(self, frame, conf=0.3):
        h, w = frame.shape[:2]
        if self.model is None:
            return []
        # ultralytics returns results object
        results = self.model.predict(source=frame, conf=conf, device=self.device, imgsz=max(h,w))[0]
        detections = []
        for r in results.boxes:
            bbox = r.xyxy.cpu().numpy().tolist()[0]
            score = float(r.conf.cpu().numpy()[0]) if hasattr(r, 'conf') else float(r.prob.cpu().numpy()[0])
            cls = int(r.cls.cpu().numpy()[0]) if hasattr(r, 'cls') else 0
            detections.append({'bbox': bbox, 'score': score, 'class_id': cls})
        return detections

# ----------------- CLIP style classification  -------
class ClipStyleClassifier:
    def __init__(self, model_name='ViT-B-32', device='cuda'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.labels = []
        if OPENCLIP_AVAILABLE and TORCH_AVAILABLE:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
                tokenizer = open_clip.get_tokenizer(model_name)
                model.to(device)
                model.eval()
                self.model = model
                self.preprocess = preprocess
                self.tokenizer = tokenizer
                print('[ClipStyleClassifier] OpenCLIP loaded')
            except Exception as e:
                print('[ClipStyleClassifier] Failed to load OpenCLIP:', e)
        else:
            print('[ClipStyleClassifier] open_clip or torch not available -> classifier disabled')

    def predict(self, pil_image, text_prompts):
        """Zero-shot scores for a list of text_prompts. Returns list of probabilities."""
        if self.model is None:
            # fallback: uniform low confidence
            return [0.0] * len(text_prompts)
        import torch
        img = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        texts = self.tokenizer(text_prompts)
        with torch.no_grad():
            image_emb = self.model.encode_image(img)
            text_emb = self.model.encode_text(texts)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
            return logits.cpu().numpy()[0].tolist()

# ----------------- Color extraction ----------------
def extract_palette(bgr_roi, n_colors=3):
    """Return top N colors as hex codes. bgr_roi is an HxWx3 numpy array."""
    if not SKLEARN_AVAILABLE:
        # fallback: compute dominant by mean per channel
        avg = bgr_roi.reshape(-1,3).mean(axis=0)
        return [rgb_to_hex(avg)]
    data = bgr_roi.reshape(-1,3).astype(np.float32)
    # sample if large
    if data.shape[0] > 20000:
        idx = np.random.choice(data.shape[0], 20000, replace=False)
        data = data[idx]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(data)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    colors = [rgb_to_hex(c[::-1]) for c in centers]  # convert BGR->RGB in hex
    return colors

# ----------------- Aesthetic (optional) -------------
class AestheticPredictor:
    def __init__(self):
        self.ready = False
        try:
            # user may implement LAION predictor here
            from aesthetic_predictor import AestheticPredictor as AP
            self.model = AP()
            self.ready = True
        except Exception:
            print('[AestheticPredictor] Not available -> skipping aesthetic scoring')

    def score(self, pil_image):
        if not self.ready:
            return None
        return float(self.model.predict(pil_image))

# ----------------- Pipeline -------------------------

def process_frame(frame, segmenter, detector, classifier, aestheticer):
    h, w = frame.shape[:2]
    # 1) person mask
    mask = segmenter.segment(frame)
    mask_3 = np.expand_dims(mask, axis=2)

    # 2) compose transparent person (crop bbox by mask)
    # compute tight bbox from mask
    ys, xs = np.where(mask > 0.1)
    if len(xs) == 0:
        person_bbox = [0,0,w,h]
        cropped = frame.copy()
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        person_bbox = [int(x1), int(y1), int(x2), int(y2)]
        cropped = frame[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]

    # 3) clothes detection
    dets = detector.detect(frame) if detector is not None else []

    # 4) palette extraction (on whole person crop)
    colors = []
    if cropped.size != 0:
        colors = extract_palette(cropped, n_colors=3)

    # 5) style classification with CLIP
    style_scores = {}
    prompts = [
        'formal clothing', 'casual clothing', 'sporty outfit', 'evening wear',
        'streetwear', 'gothic clothing', 'cosplay outfit', 'workwear uniform'
    ]
    from PIL import Image
    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    if classifier is not None:
        probs = classifier.predict(pil_crop, prompts)
        style_scores = dict(zip(prompts, probs))

    # 6) aesthetic score
    aest = aestheticer.score(pil_crop) if aestheticer is not None else None

    result = {
        'bbox_person': person_bbox,
        'colors': colors,
        'clothes_detections': dets,
        'style_scores': style_scores,
        'aesthetic': aest,
        'mask_summary': {
            'mask_mean': float(np.mean(mask)),
            'mask_sum': int(np.sum(mask>0.1))
        }
    }
    return result, mask

# ----------------- Main / CLI ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=0, help='camera index or path to video/file')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--modnet', action='store_true', help='use MODNet if available')
    parser.add_argument('--yolov8', action='store_true', help='use YOLOv8 clothes detector')
    parser.add_argument('--clip', action='store_true', help='use OpenCLIP for style classification')
    parser.add_argument('--show', action='store_true', help='show GUI window')
    args = parser.parse_args()

    device = 'cuda' if args.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    print('Device chosen:', device)

    # init modules
    segmenter = ModnetSegmenter(device=device) if args.modnet else ModnetSegmenter(device='cpu')
    detector = ClothesDetector(weights_path=YOLOV8_CKPT, device='gpu' if args.use_gpu else 'cpu') if args.yolov8 else None
    classifier = ClipStyleClassifier(device=device) if args.clip else None
    aestheticer = AestheticPredictor()

    # video source
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        print('Failed to open source:', args.source)
        return

    print('Press q or Esc to quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        out, mask = process_frame(frame, segmenter, detector, classifier, aestheticer)

        # compose debug overlay: draw bbox, colors
        vis = frame.copy()
        x1,y1,x2,y2 = out['bbox_person']
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        # draw palette
        for i,hexc in enumerate(out['colors']):
            cv2.putText(vis, hexc, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(vis, (120 + i*40, 10), (150 + i*40, 30), tuple(int(hexc.lstrip('#')[j:j+2],16) for j in (4,2,0)), -1)

        # show simple style top
        if out['style_scores']:
            # show best label
            best = max(out['style_scores'].items(), key=lambda x: x[1])
            cv2.putText(vis, f"style: {best[0]} {best[1]:.2f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if args.show:
            cv2.imshow('pipeline', vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
        else:
            # if headless, dump JSON to stdout occasionally
            print(json.dumps(out))
            time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
