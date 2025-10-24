# MODNet Segmentation — readme_ms.md

Ниже — подробный разбор и инструкция по использованию скрипта `modnet_segmentation.py`.

## Короткое описание

`modnet_segmentation.py` — демонстрационный скрипт для маттинга (отделения переднего плана от фона) в режиме реального времени с веб-камеры на основе модели MODNet. Скрипт поддерживает:

- загрузку чекпойнта PyTorch (checkpoint .ckpt)
- экспорт/загрузку TorchScript (cpu) и использование его вместо оригинальной модели
- предобработку: CLAHE (локальное выравнивание контраста) и гамма-коррекция (toggle/once)
- отображение результата в окне OpenCV и вывод FPS

## Краткий контракт

- Входы: кадры с камеры (BGR numpy-изображение), размер произвольный
- Выходы: окно OpenCV с результатом (фон заменён белым) и вывод FPS в консоль
- Ошибки: отсутствие камеры — бросает RuntimeError

## Зависимости

Установите зависимости (рекомендуется виртуальное окружение):

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
# опционально (для MODNet):
pip install -r modnet_seg_requirements.txt
```

Требуется минимум:
- Python 3.8+ (практически совместимо с 3.7+)
- torch, torchvision (при необходимости — CUDA-версия для GPU)
- opencv-python (не headless, чтобы поддерживалось `cv2.imshow`)
- numpy
- torchvision.transforms

Файлы моделей:

- По умолчанию скрипт ожидает чекпойнт по пути, заданному в `MODEL_PATH` (внутри скрипта). В репозитории есть пример пути:
  `models/modnet_photographic_portrait_matting.ckpt`
- TorchScript (выходной) файл сохраняется по `TS_PATH` (по умолчанию `modnet_cpu.pt`). Если он уже существует, скрипт загрузит его.

## Где смотреть и что изменить

- `MODEL_PATH` — путь к чекпойнту PyTorch
- `TS_PATH` — путь для сохранения/загрузки TorchScript
- `DEVICE` определяется автоматически: `cuda` если torch видит GPU, иначе `cpu`.

Если вы хотите хранить модели в папке `models/`, измените `MODEL_PATH` на `models/modnet_photographic_portrait_matting.ckpt`.

## Как работает скрипт — краткий разбор логики

1. Добавление в `sys.path` папки `MODNet/src`, импорт модели `MODNet`.
2. Инициализация модели `MODNet(backbone_pretrained=False)`, перевод на DEVICE и загрузка весов из `MODEL_PATH`.
3. Определены преобразования `to_tensor` (ToTensor + Normalize) и вспомогательные функции:
   - `apply_clahe_rgb` — CLAHE в цветовом пространстве LAB (применяется к L-каналу)
   - `apply_gamma` — гамма-коррекция через lookup table
4. Функция `ensure_torchscript(model, device, ts_path)` —
   - если `ts_path` существует, загружает TorchScript-модель и включает флаг `use_torchscript`
   - иначе экспортирует модель в TorchScript (`torch.jit.script`), сохраняет и затем загружает
5. Основной цикл:
   - Чтение кадра из `cv2.VideoCapture(0)` (камеры)
   - Применение CLAHE/гаммы (опции once/always)
   - Resize кадра до 512x512 (в коде: pw, ph = 512)
   - Прогон через модель (либо `ts_model` на CPU, либо оригинальный `modnet` на `DEVICE`)
   - Поиск тензора матте (pha/alpha) в возвращаемых output'ах (tuple/list/tensor/dict)
   - Интерполяция матте к размеру исходного кадра, применение к BGR изображению — фон заменяется белым
   - Отображение результата и показ FPS
   - Обработка клавиш:
     - `q` — выход
     - `c` — CLAHE один раз (once)
     - `C` — toggle CLAHE always
     - `g` — Gamma один раз
     - `G` — toggle Gamma always
     - `t` — экспорт/загрузка TorchScript (CPU)

## Примеры запуска (PowerShell, Windows)

Запуск модульного скрипта (по умолчанию берёт `0` camera):

```powershell
# активировать виртуальное окружение
.\.venv\Scripts\Activate
python .\modnet_segmentation.py
```

Запуск и принудительное использование CPU TorchScript (после нажатия `t` в окне или создания `modnet_cpu.pt` вручную):

```powershell
python .\modnet_segmentation.py
# затем нажмите клавишу 't' в окне для экспорта/загрузки TorchScript
```

Если вы хотите сразу использовать CPU TorchScript (без экспорта в интерактиве), можно предварительно сгенерировать `modnet_cpu.pt` через Python REPL или отдельный скрипт, вызвав `ensure_torchscript(modnet, 'cpu')`.

## Частые проблемы и рекомендации

- Камера не доступна: проверьте, что устройство подключено и не используется другим приложением. Тест: `python test_t.py` (есть в репозитории).
- `cv2.imshow` не работает: скорее всего, установлена `opencv-python-headless`. Установите `opencv-python`.
- Падает загрузка чекпойнта: убедитесь, что `MODEL_PATH` указывает на корректный файл `.ckpt` и что версия PyTorch совместима.
- Медленно на CPU: export в TorchScript может ускорить инференс на CPU, но лучшая производительность — на GPU с подходящей версией PyTorch + CUDA.
- TorchScript: иногда `torch.jit.script` не работает (например, из-за динамического кода). Скрипт уже использует `torch.jit.script`; если экспорт не проходит — оставьте `use_torchscript=False` и работайте с исходной моделью на `DEVICE`.

## Edge cases и замечания

- Пустые/коррумпированные кадры — пропускаются
- Модель может возвращать разные форматы (tuple/list/tensor/dict) — в коде есть безопасная проверка для извлечения матте
- Скрипт жестко ресайзит изображение до 512x512 перед подачей в сеть — для других размеров возможно снижение качества при ресайзе

## Что можно улучшить (next steps)

- Добавить аргументы командной строки (argparse) для настройки путей `MODEL_PATH`, `TS_PATH`, `camera_index`, размеров входа и флагов (CLAHE/Gamma/TorchScript) — сейчас они заданы в коде
- Добавить логирование вместо print
- Добавить unit-test на функцию извлечения `matte` из `outputs`

## Контакты

Если нужно — могу: добавить `argparse`, сделать автоматический поиск чекпойнтов в `models/`, или написать небольшой wrapper-скрипт для пакетного тестирования на видеозаписях.

----

Файл: `modnet_segmentation.py` — расположение и ключевые опции описаны выше. Этот `readme_ms.md` создан для быстрого старта и отладки на Windows/PowerShell.
