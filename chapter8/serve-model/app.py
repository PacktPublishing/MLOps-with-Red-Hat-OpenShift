'''
    this is a simple python http server that will accept the rquest to detect if the a 
    single frame of shape 256, 256, 3 contains a face or not.

    the server will also REDIS based atomic increment a face or not counter for each request that it receives.
    later the counter value can be displayed in the UI. there will be additional http endpoint for that.

'''

import os
import aioredis


from aiohttp import  web
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from aiohttp.web_app import Application
import tritonclient.grpc as grpcclient
import numpy as np
import base64
from aiohttp_middlewares import cors_middleware
from aiohttp_middlewares.cors import DEFAULT_ALLOW_HEADERS

# Local test
# MODEL_SERVER = os.getenv('MODEL_SERVER', 'localhost:8033')
# REDIS_SERVER = os.getenv('REDIS_SERVER', 'redis://localhost:6379')
MODEL_SERVER = os.getenv('MODEL_SERVER', 'modelmesh-serving.wines.svc.cluster.local:8033')
REDIS_SERVER = os.getenv('REDIS_SERVER', 'redis://redis.wines.svc.cluster.local:6379')

GRPC_CLIENT = "grpc_client"
REDIS_CLIENT = "redis_client"



model_name = "face-detection-ser-ov"



def create_grpc_pool(app: Application):          
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url=MODEL_SERVER,
            verbose=False,
            keepalive_options=keepalive_options)
        print(f"Connected to Triton at {triton_client}")
        app[GRPC_CLIENT] = triton_client
    except Exception as e:
        print("channel creation failed: " + str(e))    
    
    
async def create_redis_client(app: Application):
    redis = aioredis.from_url(
        REDIS_SERVER, encoding="utf-8", decode_responses=True
    )
    print(f"Connected to Redis at {redis}")
    app[REDIS_CLIENT]  = redis



async def infer(request: Request) -> Response:
    max_confidence_index = await infer_request(request)
    #2 is face
    if max_confidence_index == 2:
        await increment_face_counter()

    return web.Response(text=str(max_confidence_index))

async def increment_face_counter():
    redis = app[REDIS_CLIENT]
    await redis.incr('face-count', amount=1)


async def infer_request(request) -> bool:

    # video_frame = np.random.randn(1, 256, 256, 3).astype(np.float32)
    # print(video_frame)

    # raw_frame = await request.text()
    raw_frame = await request.read()
    decoded_frame = base64.b64decode(raw_frame).decode()
    values = [int(i) for i in decoded_frame.split(',')]
    image_width = 255 * 4
    image_frame:int = np.zeros((256, 256, 3), dtype=np.float32)
    for i in range(0, 256):
        for j in range (0, 256):
            firstitem = (i * (image_width)) + (j * 4)
            image_frame[i][j][0] = np.float32(values[firstitem])
            image_frame[i][j][1] = np.float32(values[firstitem+1])
            image_frame[i][j][2] = np.float32(values[firstitem+2])

    # video_frame = np.array(image_frame)
    # video_frame = np.array(decoded_frame.split(',').encode())
    video_frame = image_frame.reshape(1, 256, 256, 3)
    # print(rb)

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input_1', [1, 256, 256, 3], "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(video_frame)

    outputs.append(grpcclient.InferRequestedOutput('pred'))

    # Test with outputs
    results = app[GRPC_CLIENT].infer(model_name=model_name,
                                    inputs=inputs,
                                    outputs=outputs)
    # print(f"Raw results are {results}")
    # Get the output arrays from the results
    output0_data = results.as_numpy('pred')
    flatten_data = output0_data.flatten()
    # print(flatten_data[0])
    # print(flatten_data[1])
    # print(flatten_data[2])
    max_confidence_index = np.argmax(flatten_data)
    # print(max_confidence_index)
    return max_confidence_index




async def get_infer_count(request: Request) -> Response:
    redis = app[REDIS_CLIENT]    
    value = await redis.get('face-count')
    print(f"Infer Count ->{value}")
    return web.Response(text=value)




app = web.Application(client_max_size=1905280, 
                    middlewares=[cors_middleware(allow_all=True)])  

app.add_routes([web.get('/infer-count', get_infer_count),
                web.post('/infer', infer)])

#blocking connection establishment
create_grpc_pool(app)
app.on_startup.append(create_redis_client)

web.run_app(app)



