import asyncio
import base64
import json
import time
import cv2
import numpy as np
import websockets


async def run_test():
    uri = 'ws://127.0.0.1:8000/segment'

    # create a simple colored test image
    img = np.full((240, 320, 3), 200, dtype=np.uint8)
    cv2.putText(img, 'TEST', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (10, 10, 200), 3)
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    try:
        # older websockets versions accept 'origin' instead of extra_headers
        async with websockets.connect(uri, origin='http://127.0.0.1:8001') as ws:
            print('Connected to', uri)
            # send start
            await ws.send(json.dumps({'type': 'start'}))
            msg = await ws.recv()
            print('->', msg)

            # send a frame
            frame_id = str(time.time())
            await ws.send(json.dumps({'type': 'frame', 'frame_id': frame_id, 'data': b64}))
            print('frame sent, waiting for result...')
            resp = await ws.recv()
            print('raw resp length', len(resp))
            try:
                j = json.loads(resp)
            except Exception as e:
                print('invalid json response', e)
                return

            if j.get('type') == 'result' and j.get('data'):
                out_b64 = j['data']
                out = base64.b64decode(out_b64)
                with open('ws_result_from_server.png', 'wb') as f:
                    f.write(out)
                print('Saved ws_result_from_server.png kind=', j.get('kind'))
            else:
                print('Server returned:', j)

    except Exception as e:
        print('WebSocket test failed:', type(e).__name__, e)


if __name__ == '__main__':
    asyncio.run(run_test())
