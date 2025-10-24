"""
GPU-ready demo pipeline for: segmentation (MODNet / Mediapipe fallback),
YOLOv8 clothes detection, OpenCLIP zero-shot style classification,
color palette extraction, background replacement.

Place model checkpoints into ./models/:
 - modnet_photographic_portrait_matting.ckpt
 - yolov8n-clothes.pt

Example:
  python gpu_fashion_pipeline.py --source 0 --use_gpu --modnet --yolov8 --clip --show

This file contains fixes for device placement with OpenCLIP tokenizer, more
robust YOLO device handling / detection parsing, safer MODNet device usage,
and more robust background path handling.
"""
import argparse
import os
import time
import json
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageSequence

# импорт modnet
try:
    from MODNet.src.models.modnet import MODNet
except ImportError:
    from modnet import MODNet



# optional heavy deps
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
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

# model dir
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODNET_CKPT = os.path.join(MODEL_DIR, "modnet_photographic_portrait_matting.ckpt")
YOLOV8_CKPT = os.path.join(MODEL_DIR, "yolov8n-clothes.pt")
# ----------------------------------------------------

def rgb_to_hex(bgr):
    # bgr -> hex
    return '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

# ---- Segmenter (MODNet or Mediapipe or coarse) ----
class ModnetSegmenter:
    def __init__(self, device='cpu', seg_size=512, use_modnet=False):
        self.device = device if isinstance(device, str) else ('cuda' if device == 'cuda' else 'cpu')
        self.seg_size = seg_size
        self.model = None
        self.mp_segment = None
        self.ready = False

        if use_modnet and TORCH_AVAILABLE:
            try:
                # корректный импорт MODNet (установленный как пакет)
                try:
                    from models.modnet import MODNet
                except ImportError:
                    from modnet import MODNet

                if os.path.exists(MODNET_CKPT):
                    print(f"[Segmenter] Loading MODNet checkpoint from: {MODNET_CKPT}")
                    ckpt = torch.load(MODNET_CKPT, map_location='cpu')

                    # создаем модель и грузим веса
                    self.model = MODNet(backbone_pretrained=False)
                    self.model.load_state_dict(ckpt)

                    if self.device.startswith('cuda') and torch.cuda.is_available():
                        self.model.to(self.device)
                        print(f"[Segmenter] MODNet moved to GPU ({self.device})")
                    else:
                        print("[Segmenter] MODNet running on CPU")

                    self.model.eval()
                    self.ready = True
                    print("[Segmenter] MODNet loaded successfully ✅")

                else:
                    print(f"[Segmenter] ❌ MODNet checkpoint missing at {MODNET_CKPT}, falling back")

            except Exception as e:
                print(f"[Segmenter] MODNet init failed: {type(e).__name__}: {e}")

        # fallback → Mediapipe or brightness mask
        if not self.ready:
            if MEDIAPIPE_AVAILABLE:
                mp_selfie_seg = mp.solutions.selfie_segmentation
                self.mp_segment = mp_selfie_seg.SelfieSegmentation(model_selection=1)
                self.ready = True
                print("[Segmenter] Using Mediapipe SelfieSegmentation (fallback)")
            else:
                print("[Segmenter] No MODNet and no Mediapipe: using brightness fallback")
                self.ready = True


    def segment(self, frame):
        """
        Return mask as float32 HxW in [0..1].
        We process at smaller resolution (seg_size) then upscale for performance/quality balance.
        """
        h, w = frame.shape[:2]
        if self.model is not None:
            import torchvision.transforms as T
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_small = cv2.resize(im, (self.seg_size, self.seg_size))
            im_t = T.ToTensor()(im_small).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # MODNet API in repos may differ; this is a typical pattern — adapt if necessary
                matte = self.model(im_t, True)[0][0][0].cpu().numpy()
            matte = cv2.resize(matte, (w, h))
            matte = np.clip(matte, 0.0, 1.0).astype(np.float32)
            return matte
        elif self.mp_segment is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_segment.process(rgb)
            if res.segmentation_mask is None:
                return np.zeros((h, w), dtype=np.float32)
            mask = cv2.resize(res.segmentation_mask, (w, h))
            return mask.astype(np.float32)
        else:
            # simple fallback: brightness-based
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            norm = (gray - gray.mean()) / (gray.std() + 1e-6)
            mask = (norm < 1.0).astype(np.float32)
            return mask

# ---- YOLOv8 clothes detector wrapper ----
class ClothesDetector:
    def __init__(self, weights_path=None, device='cpu'):
        self.model = None
        # normalize device to ultralytics expected form: 'cpu' or '0' (for GPU 0)
        if isinstance(device, int):
            self.device = str(device)
        else:
            # allow 'cuda' -> '0' as shorthand
            if isinstance(device, str) and device.startswith('cuda'):
                self.device = '0'
            else:
                self.device = 'cpu' if device == 'cpu' else str(device)

        if ULTRALYTICS_AVAILABLE and weights_path and os.path.exists(weights_path):
            try:
                # ultralytics YOLO accepts device string 'cpu' or '0'/'cuda:0'
                self.model = YOLO(weights_path)
                print("[ClothesDetector] YOLOv8 model loaded")
            except Exception as e:
                print("[ClothesDetector] Failed to load YOLOv8:", e)
        else:
            print("[ClothesDetector] YOLOv8 not available or weights missing")

    def detect(self, frame, conf=0.3, imgsz=640):
        if self.model is None:
            return []
        # ultralytics can accept numpy frames; ensure device string
        try:
            results = self.model.predict(source=frame, conf=conf, device=self.device, imgsz=imgsz, verbose=False)
        except Exception as e:
            print("[ClothesDetector] prediction failed:", e)
            return []
        r = results[0]
        dets = []
        # more robust parsing across ultralytics versions
        try:
            if hasattr(r, 'boxes') and r.boxes is not None:
                # prefer using .xyxy, .conf, .cls if available
                if hasattr(r.boxes, 'xyxy'):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else None
                    classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else None
                    for i in range(len(xyxy)):
                        score = float(confs[i]) if confs is not None else 0.0
                        cls = int(classes[i]) if classes is not None else -1
                        dets.append({'bbox': xyxy[i].tolist(), 'score': score, 'class_id': cls})
                else:
                    # fallback older interface
                    for b in r.boxes:
                        try:
                            xy = b.xyxy.cpu().numpy().reshape(-1).tolist()
                        except Exception:
                            xy = getattr(b, 'xyxy', None)
                        try:
                            conf_score = float(b.conf.cpu().numpy()) if hasattr(b, 'conf') else float(b.conf)
                        except Exception:
                            conf_score = 0.0
                        try:
                            cls = int(b.cls.cpu().numpy()) if hasattr(b, 'cls') else int(b.cls)
                        except Exception:
                            cls = -1
                        dets.append({'bbox': xy, 'score': conf_score, 'class_id': cls})
        except Exception as e:
            print('[ClothesDetector] parsing results failed:', e)
        return dets

# ---- CLIP classifier (zero-shot) ----
class ClipStyleClassifier:
    def __init__(self, model_name='ViT-B-32', device='cpu'):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = device if isinstance(device, str) else ('cuda' if device == 'cuda' else 'cpu')
        if OPENCLIP_AVAILABLE and TORCH_AVAILABLE:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
                tokenizer = open_clip.get_tokenizer(model_name)
                model.to(self.device)
                model.eval()
                self.model = model
                self.preprocess = preprocess
                self.tokenizer = tokenizer
                print("[CLIP] OpenCLIP loaded")
            except Exception as e:
                print("[CLIP] failed to load OpenCLIP:", e)
        else:
            print("[CLIP] open_clip or torch not available")

    def predict(self, pil_image, prompts):
        """
        Returns list of probabilities aligned to prompts.
        """
        if self.model is None:
            return [0.0] * len(prompts)
        import torch
        img = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        # tokenizer may return a torch.Tensor already or another structure
        texts = self.tokenizer(prompts)
        # ensure texts are tensors and on same device as model
        try:
            if isinstance(texts, torch.Tensor):
                texts = texts.to(self.device)
            else:
                # Some tokenizers return a tuple/list of tensors or numpy; try to convert
                texts = torch.tensor(texts).to(self.device)
        except Exception:
            # fallback: try encoding each prompt separately
            toks = [self.tokenizer([p]) for p in prompts]
            try:
                texts = torch.cat([t for t in toks], dim=0).to(self.device)
            except Exception:
                # last safe fallback: raise so we return zeros
                print("[CLIP] tokenizer output couldn't be converted; returning zeros")
                return [0.0] * len(prompts)

        with torch.no_grad():
            image_embed = self.model.encode_image(img)
            texts = texts.to(self.device if self.device != 'cpu' else 'cpu')
            text_embed = self.model.encode_text(texts)
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_embed @ text_embed.T).softmax(dim=-1)
            return logits.cpu().numpy()[0].tolist()

# ---- Color palette extraction ----
def extract_palette(bgr_roi, n_colors=3):
    if bgr_roi is None or bgr_roi.size == 0:
        return []
    data = bgr_roi.reshape(-1, 3)
    # remove fully black/near-transparent pixels if any
    # sample for speed
    if data.shape[0] > 20000:
        idx = np.random.choice(data.shape[0], 20000, replace=False)
        data = data[idx]
    if SKLEARN_AVAILABLE:
        kmeans = KMeans(n_clusters=max(1, min(n_colors, len(data))), random_state=0).fit(data.astype(np.float32))
        centers = kmeans.cluster_centers_.astype(np.uint8)
    else:
        # fallback: take K most common colors by binning
        vals, counts = np.unique(data.reshape(-1,3), axis=0, return_counts=True)
        order = np.argsort(-counts)[:n_colors]
        centers = vals[order]
    colors = [rgb_to_hex(c[::-1]) for c in centers]  # bgr->hex
    return colors

# ---- aesthetic predictor stub (optional) ----
class AestheticPredictor:
    def __init__(self):
        self.ready = False
        # user may plug LAION aesthetic predictor here
        try:
            from aesthetic_predictor import AestheticPredictor as AP
            self.model = AP()
            self.ready = True
        except Exception:
            pass

    def score(self, pil_image):
        if not self.ready:
            return None
        return float(self.model.predict(pil_image))

# ---- Pipeline processing function ----
def process_frame(frame, segmenter, detector, classifier, aestheticer, seg_smooth_deque, seg_params):
    h, w = frame.shape[:2]

    mask = segmenter.segment(frame)  # float mask [0..1]

    # morphological cleanup
    k = seg_params.get('morph_kernel', (3,3))
    mask = cv2.erode(mask, np.ones(k, np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

    # temporal smoothing
    seg_smooth_deque.append(mask)
    avg_mask = np.mean(seg_smooth_deque, axis=0)

    # soft/hard
    if seg_params.get('hard', False):
        thr = seg_params.get('hard_thresh', 0.5)
        m = (avg_mask > thr).astype(np.uint8)
        m = cv2.medianBlur(m, 5)
        m = np.expand_dims(m, 2)
    else:
        m = cv2.GaussianBlur(avg_mask, (15,15), 0)
        m = np.expand_dims(m, 2)
        m = np.clip(m, 0., 1.)

    # person bbox from mask
    ys, xs = np.where(avg_mask > 0.1)
    if len(xs) == 0:
        bbox = [0,0,w,h]
        cropped = frame.copy()
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        # guard bbox
        x1c, y1c = max(0, bbox[0]), max(0, bbox[1])
        x2c, y2c = min(w, bbox[2]), min(h, bbox[3])
        cropped = frame[y1c:y2c, x1c:x2c] if (y2c>y1c and x2c>x1c) else frame.copy()

    # clothes detection (on full frame for better context)
    dets = detector.detect(frame) if detector is not None else []

    # palette from cropped region
    colors = extract_palette(cropped, n_colors=3)

    # style classification via CLIP
    style_scores = {}
    if classifier is not None and cropped.size != 0:
        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        prompts = [
            'formal clothing', 'casual clothing', 'sporty outfit', 'evening wear',
            'streetwear', 'gothic clothing', 'cosplay outfit', 'workwear uniform'
        ]
        try:
            probs = classifier.predict(pil_crop, prompts)
            style_scores = dict(zip(prompts, probs))
        except Exception as e:
            print('[Pipeline] CLIP predict failed:', e)

    # aesthetic
    aest = None
    if aestheticer is not None and cropped.size != 0:
        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        aest = aestheticer.score(pil_crop)

    out = {
        'bbox_person': bbox,
        'colors': colors,
        'clothes_detections': dets,
        'style_scores': style_scores,
        'aesthetic': aest,
        'mask_mean': float(np.mean(mask))
    }
    return out, m[...,0].astype(np.float32)  # return mask HxW

# ---- Utils for background load (supports gif) ----
def load_bg_item(path, target_size=(640, 480)):
    """
    Load image or gif for background. Supports absolute and relative paths.
    If path is empty or not found, returns a solid fallback image.
    """
    if not path:
        img = np.full((target_size[1], target_size[0], 3), (50, 50, 50), dtype=np.uint8)
        return {'type': 'img', 'img': img}

    full_path = path if os.path.isabs(path) else os.path.join(MODEL_DIR, path)

    if not os.path.exists(full_path):
        print(f"[WARN] Background not found: {full_path}")
        img = np.full((target_size[1], target_size[0], 3), (50, 50, 50), dtype=np.uint8)
        return {'type': 'img', 'img': img}

    ext = os.path.splitext(full_path)[1].lower()

    if ext == '.gif':
        gif = Image.open(full_path)
        frames = [
            cv2.resize(cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR), target_size)
            for f in ImageSequence.Iterator(gif)
        ]
        return {'type': 'gif', 'frames': frames, 'pos': 0, 'playing': True}
    else:
        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Failed to open background: {full_path}")
            img = np.full((target_size[1], target_size[0], 3), (50, 50, 50), dtype=np.uint8)
        else:
            img = cv2.resize(img, target_size)
        return {'type': 'img', 'img': img}


# ---- Main CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=0, help='camera index or video file')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--modnet', action='store_true')
    parser.add_argument('--yolov8', action='store_true')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--seg-size', type=int, default=512, help='MODNet input size (performance/quality)')
    parser.add_argument('--half', action='store_true', help='use fp16 where possible')
    parser.add_argument('--bg', nargs='*', default=[], help='background image(s) or gif(s)')
    args = parser.parse_args()

    # device selection
    device = 'cpu'
    if args.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        device = 'cuda'
    print("Device:", device)

    # init modules
    segmenter = ModnetSegmenter(device=device, seg_size=args.seg_size, use_modnet=args.modnet)
    detector = ClothesDetector(weights_path=YOLOV8_CKPT, device=('0' if device=='cuda' else 'cpu')) if args.yolov8 else None
    classifier = ClipStyleClassifier(device=device) if args.clip else None
    aestheticer = AestheticPredictor()

    # pre-load backgrounds
    if len(args.bg) == 0:
        bg_items = [load_bg_item(None)]
    else:
        bg_items = [load_bg_item(p) for p in args.bg]
    cur_bg = 0

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        print("Failed to open source:", args.source)
        return

    # smoothing deque
    seg_history_len = 5
    seg_deque = deque(maxlen=seg_history_len)

    seg_params = {'hard': False, 'hard_thresh': 0.5, 'morph_kernel': (3,3)}

    last_time = time.time()
    frame_count = 0

    print("Controls: 1/2/3 switch bg | m toggle hard | b toggle fp16 | r invert | q/ESC quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # optionally do smaller working size for models (we already do inside segmenter)
        out, mask = process_frame(frame, segmenter, detector, classifier, aestheticer, seg_deque, seg_params)

        # pick bg frame
        bg_item = bg_items[cur_bg % len(bg_items)]
        if bg_item['type'] == 'gif':
            frames = bg_item['frames']
            idx = bg_item['pos']
            bg_frame = frames[idx % len(frames)]
            if bg_item.get('playing', True):
                bg_item['pos'] = (bg_item['pos'] + 1) % len(frames)
        else:
            bg_frame = bg_item['img']
        bg_resized = cv2.resize(bg_frame, (w, h))

        # compose background replacement
        mask_3 = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
        composed = (frame.astype(np.float32) * mask_3 + bg_resized.astype(np.float32) * (1 - mask_3)).astype(np.uint8)

        # overlay bbox and info
        x1,y1,x2,y2 = out['bbox_person']
        cv2.rectangle(composed, (x1,y1), (x2,y2), (0,255,0), 2)
        # palette
        for i, hexc in enumerate(out['colors']):
            cv2.putText(composed, hexc, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # rect color conversion hex->bgr
            try:
                r = int(hexc[1:3], 16); g = int(hexc[3:5], 16); b = int(hexc[5:7], 16)
                cv2.rectangle(composed, (120 + i*40, 10), (150 + i*40, 30), (b,g,r), -1)
            except Exception:
                pass
        # style
        if out['style_scores']:
            best = max(out['style_scores'].items(), key=lambda x: x[1])
            cv2.putText(composed, f"{best[0]} {best[1]:.2f}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if args.show:
            cv2.imshow("GPU Fashion Pipeline", composed)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('1'), ord('2'), ord('3')):
                idx = int(chr(key)) - 1
                if 0 <= idx < len(bg_items):
                    cur_bg = idx
            elif key == ord('m'):
                seg_params['hard'] = not seg_params['hard']
                print("Hard mask:", seg_params['hard'])
            elif key == ord('b'):
                # toggle CLAHE or fp16? We'll toggle use of fp16 flag (not implemented heavy here)
                print("Toggle action pressed (implement as needed)")
            elif key == ord('r'):
                # invert replace mode
                pass
        else:
            print(json.dumps(out))
            time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
