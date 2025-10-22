# 👗 GPU Fashion Pipeline

Интерактивный AI-пайплайн для **распознавания одежды**, **сегментации человека** и **определения стиля** с помощью GPU.  
Поддерживает работу с камерой, сегментацию фона, CLIP-классификацию и YOLOv8-детекцию.

---

## 📁 Структура проекта

hackaton/
│
├── gpu_fashion_pipeline.py # основной файл пайплайна
├── models/ # папка для весов (YOLO, MODNet, CLIP)
│ ├── yolov8n.pt
│ ├── modnet.onnx
│ └── open_clip_model.safetensors
│
├── backgrounds/ # возможные фоны для замены
├── outputs/ # сохраняемые результаты (видео, JSON, скриншоты)
├── test_t.py # тест импорта и окружения
├── requirements.txt # зависимости
├── .gitignore # исключения для git
└── README.md # документация (этот файл)

---

## ⚙️ Установка

### 1. Клонирование и создание окружения
```bash
git clone https://github.com/username/gpu_fashion_pipeline.git
cd gpu_fashion_pipeline
python -m venv venv
venv\Scripts\activate

2. Установка зависимостей
pip install -r requirements.txt
💡 Если используешь GPU, убедись, что у тебя установлен PyTorch с CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

🧩 Использование
Запуск с веб-камеры:
python gpu_fashion_pipeline.py --source 0 --use_gpu --modnet --yolov8 --clip --show

Основные флаги:
Флаг	Описание
--source <path>	Источник видео: камера (0), путь к файлу или URL
--use_gpu	Использовать CUDA (если доступна)
--modnet	Сегментация человека через MODNet
--yolov8	Детекция одежды YOLOv8
--clip	Определение стиля через CLIP
--show	Показ видео в реальном времени
--save Сохранение результата в outputs/output.mp4

🧠 Основные функции

Сегментация человека — удаление или замена фона (MODNet / Mediapipe).

Детекция одежды — YOLOv8 выделяет одежду на человеке.

Классификация стиля — OpenCLIP оценивает стиль по текстовым промптам.

GPU-ускорение — PyTorch и TensorFlow Lite на CUDA / XNNPACK.

Aesthetic scoring — (опционально) оценка визуального качества изображения.

Реальный поток — поддержка камер, RTSP, видеофайлов.

🧩 Известные недоработки

⚠️ Если opencv-python собран без GUI, опция --show вызывает ошибку (cv2.imshow не работает).
→ Решение: установить opencv-python, не opencv-python-headless.

⚠️ YOLOv8-веса должны быть вручную скачаны и помещены в /models.

⚠️ Если нет интернета, CLIP-модель не сможет загрузить веса с HuggingFace (нужно заранее скачать open_clip_model.safetensors).

⚠️ TensorFlow Lite может ругаться при отсутствии GPU Delegate — это не критично.

📌 Планы доработки

✅ Автоматическая загрузка недостающих моделей.

🧩 Поддержка мультимодального вывода (видео + JSON).

🎨 Улучшенный интерфейс отображения (streamlit/webui).

🧠 Обучение кастомных промптов для CLIP.

📦 Docker-образ для развёртывания на сервере.