# ws_server quick start

This file explains how to run `ws_server.py` (WebSocket segmentation server) and how to connect from a browser.

Requirements
- Python 3.8+ (3.10 recommended)
- A virtual environment (strongly recommended)

Install (PowerShell)

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r req_serv_test3.txt
```

Run server

```powershell
& .\.venv\Scripts\Activate.ps1
uvicorn ws_server:app --host 0.0.0.0 --port 8000
```

WebSocket endpoint
- ws://localhost:8000/segment

Protocol summary
- Client -> Server: JSON messages with base64 frame in `data` for `type":"frame"`.
- Server -> Client: JSON `result` with base64 PNG in `data` (fields: frame_id, data, format, kind, meta).

Example client sketch (JS): send canvas.toDataURL().split(',')[1] as `data` in JSON.

Notes
- Base64 is convenient for quick integration but less efficient than binary frames (ArrayBuffer). Switch later for performance.
- If a NumPy ABI error occurs, install a 1.x NumPy: `pip install "numpy<2"`.
