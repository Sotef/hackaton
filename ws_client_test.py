"""Simple WebSocket client test for ws_server.

Sends a single synthetic frame (white rectangle on black) as base64 JPEG and waits for server result.
Saves returned PNG to `ws_result.png`.

Run inside venv:
  & .\.venv\Scripts\Activate.ps1
  python ws_client_test.py
"""
import asyncio
import base64
import json
import time
import cv2
import numpy as np

async def run():
    import websockets
    uri = 'ws://localhost:8000/segment'
    # include Origin to mimic browser (some servers check Origin)
    async with websockets.connect(uri, origin='http://localhost:8000') as ws:
        print('connected')
        # send ping
        await ws.send(json.dumps({'type':'ping'}))
        msg = await ws.recv()
        print('recv:', msg)

        # create synthetic image
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (50,50), (270,190), (255,255,255), -1)
        # encode jpeg
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        frame_id = 'cli-' + str(int(time.time()*1000))
        payload = {'type':'frame', 'frame_id': frame_id, 'data': b64, 'meta':{'fps':1}}
        t0 = time.time()
        await ws.send(json.dumps(payload))
        print('frame sent')

        # wait for result
        res_text = await ws.recv()
        dt = (time.time() - t0) * 1000
        print('result received (ms):', int(dt))
        res = json.loads(res_text)
        if res.get('type') == 'result':
            data_b64 = res.get('data')
            img_bytes = base64.b64decode(data_b64)
            with open('ws_result.png', 'wb') as f:
                f.write(img_bytes)
            print('saved ws_result.png kind=', res.get('kind'), 'meta=', res.get('meta'))
        else:
            print('server response:', res)

if __name__ == '__main__':
    asyncio.run(run())
