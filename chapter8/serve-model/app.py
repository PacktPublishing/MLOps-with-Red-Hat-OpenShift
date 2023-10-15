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
import json

MODEL_SERVER = os.getenv('MODEL_SERVER', 'https://fm-test-wines.apps.fmflask2.faisallabs.net') #'modelmesh-serving.wines.svc.cluster.local:8033')
REDIS_SERVER = os.getenv('REDIS_SERVER', 'redis://redis.wines.svc.cluster.local:6379')
GRPC_CLIENT = "grpc_client"
REDIS_CLIENT = "redis_client"

routes = web.RouteTableDef()

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

@routes.post('/infer') 
async def infer(request: Request) -> Response:
    response = await infer_request(request)
    face_or_not = True
    if face_or_not:
        await increment_face_counter()

    return web.Response(text=response)

async def increment_face_counter():
    async with REDIS_CLIENT as conn:
        return await conn.incr('face-count', amount=1)


async def infer_request(request) -> bool:

    video_frame = np.random.randn(1, 256, 256, 3).astype(np.float32)

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input_1', [1, 256, 256, 3], "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(video_frame)

    outputs.append(grpcclient.InferRequestedOutput('pred'))

    # Test with outputs
    results = await app[GRPC_CLIENT].infer(model_name=model_name,
                                    inputs=inputs,
                                    outputs=outputs)
    print(results)
    # Get the output arrays from the results
    output0_data = results.as_numpy('pred')
    print(json.dumps(output0_data, indent=2))
    print(len(json.loads(output0_data, indent=2)))



@routes.get('/infer-count') 
async def get_infer_count(request: Request) -> Response:
    print("Infere count")
    redis = app[REDIS_CLIENT]
    await redis.incr('face-count', amount=1)
    value = await redis.get('face-count')
    print("==" + value)
    return web.Response(text=value)


app = web.Application()  

#blocking connection establishment
create_grpc_pool(app)
app.on_startup.append(create_redis_client)

app.add_routes(routes)
web.run_app(app)