import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import os

class HumanSegmentator:
    """ВСЁ для сегментации человека в одном классе"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.image_size = (256, 256)
        print(f"✅ AI Model loaded on: {self.device}")
    
    def _load_model(self):
        """Загрузка модели"""
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=1,
            activation="sigmoid"
        )
        model.to(self.device)
        model.eval()
        return model
    
    def process_image(self, image_path, background="blue"):
        """ОСНОВНОЙ МЕТОД: загрузить -> сегментировать -> поменять фон"""
        # Загрузка
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        original_size = image.shape[:2]
        
        # Сегментация
        processed = self._preprocess(image)
        with torch.no_grad():
            output = self.model(processed)
            mask = output.squeeze().cpu().numpy()
        
        # Постобработка
        final_mask = self._postprocess(mask, original_size)
        
        # Замена фона
        result = self._apply_background(image, final_mask, background)
        
        return final_mask, result
    
    def _preprocess(self, image):
        """Подготовка изображения для нейросети"""
        image_resized = cv2.resize(image, self.image_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def _postprocess(self, mask, original_size):
        """Улучшение маски"""
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Убираем шум
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    
    def _apply_background(self, image, mask, background_type):
        """Применение нового фона"""
        if background_type == "blue":
            background = np.full_like(image, [255, 0, 0])
        elif background_type == "green":
            background = np.full_like(image, [0, 255, 0])
        elif background_type == "gradient":
            background = self._create_gradient(image.shape)
        else:
            background = np.full_like(image, [0, 0, 0])
        
        return np.where(mask[:, :, np.newaxis] == 255, image, background)
    
    def _create_gradient(self, size):
        """Создание красивого градиента"""
        h, w = size[:2]
        background = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            blue = int(255 * (1 - i/h))
            red = int(255 * (i/h))
            background[i, :] = [blue, 0, red]
        
        return background

def create_demo_image():
    """Создание демо изображения с человеком"""
    img = np.ones((400, 300, 3), dtype=np.uint8) * 255
    
    # Стилизованный человек
    cv2.circle(img, (150, 80), 30, (100, 100, 100), -1)  # Голова
    cv2.rectangle(img, (135, 110), (165, 250), (100, 100, 100), -1)  # Тело
    cv2.rectangle(img, (135, 250), (145, 350), (100, 100, 100), -1)  # Нога левая
    cv2.rectangle(img, (155, 250), (165, 350), (100, 100, 100), -1)  # Нога правая
    cv2.rectangle(img, (115, 120), (135, 200), (100, 100, 100), -1)  # Рука левая
    cv2.rectangle(img, (165, 120), (185, 200), (100, 100, 100), -1)  # Рука правая
    
    return img