# ws_server quick start
````markdown
# Быстрый запуск ws_server (WebSocket-сервер сегментации)

Этот файл описывает, как запустить `ws_server.py` и как подключиться к нему из браузера для тестирования передачи кадров и получения результата (масок/композитов).

Требования
- Python 3.8+ (рекомендуется 3.10)
- Виртуальное окружение (strongly recommended)

Установка (PowerShell):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r req_serv_test3.txt
```

Запуск сервера:

```powershell
& .\.venv\Scripts\Activate.ps1
# Рекомендуется связывать на loopback для локального тестирования
uvicorn ws_server:app --host 127.0.0.1 --port 8000
```

URL WebSocket-эндпоинта
- ws://127.0.0.1:8000/segment

Краткий формат протокола
- Клиент -> Сервер: JSON с полем `type`. Для передачи кадра используйте:
	```json
	{"type":"frame","frame_id":"...","data":"<base64 jpeg/png>","meta":{...}}
	```
- Сервер -> Клиент: JSON-ответ `result` с base64 PNG в поле `data` и мета-данными:
	```json
	{"type":"result","frame_id":"...","data":"<base64 png>","format":"png","kind":"mask|composite|rgba","meta":{...}}
	```

Пример краткого клиента (JS)
- На стороне браузера: возьмите canvas.toDataURL('image/jpeg', quality).split(',')[1] и отправьте как `data` в JSON.

Рекомендации и отладка
- Для локального тестирования используйте `127.0.0.1` — в некоторых системах `localhost` может резолвиться в IPv6 или проксироваться (Docker/WSL), что даёт неожиданные 403/PermissionError.
- Если при импорте модулей (например, torch/ultralytics) вы видите ошибки ABI от NumPy, установите совместимую версию: `pip install "numpy<2"`.
- Если WebSocket handshake возвращает 403 при Python-клиенте, попробуйте:
	- подключаться к `ws://127.0.0.1:8000/segment` (вместо `localhost`);
	- отправлять заголовок Origin, похожий на браузер (например, `http://127.0.0.1:8001`).
- Для проверки, занят ли порт/кто слушает, используйте в PowerShell:
	```powershell
	netstat -aon | Select-String ":8000"
	tasklist /FI "PID eq <pid>"
	```

Полезные артефакты в этом репозитории
- `ws_test.html` — минимальная страница для браузера, которая захватывает камеру и отправляет кадры на `/segment`.
- `ws_client_headless.py` — headless Python-клиент, который эмулирует браузерный Origin, отправляет тестовый кадр и сохраняет результат в `ws_result_from_server.png`.

Примечание по производительности
- Base64/JSON удобно для прототипа и совместимости с простыми фронтендами, но накладывает дополнительное копирование и рост трафика. Для продакшена стоит перейти на бинарные WebSocket-фреймы (ArrayBuffer) или на потоковую передачу.

Если нужно — могу добавить curl/пример Python-клиента, скрипт автотестирования или инструкцию по развёртыванию через systemd/Windows service.

````
