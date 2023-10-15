import tritonclient.grpc as grpcclient
import numpy as np

try:
    keepalive_options = grpcclient.KeepAliveOptions(
        keepalive_time_ms=2**31 - 1,
        keepalive_timeout_ms=20000,
        keepalive_permit_without_calls=False,
        http2_max_pings_without_data=2
    )
    triton_client = grpcclient.InferenceServerClient(
        url='modelmesh-serving.wines.svc.cluster.local:8033',
        verbose=False,
        keepalive_options=keepalive_options)
except Exception as e:
    print("channel creation failed: " + str(e))

model_name = "face-detection-ser-ov"

inputs = []
outputs = []
inputs.append(grpcclient.InferInput('input_1', [1, 256, 256, 3], "FP32"))



input0_data = np.random.randn(1, 256, 256, 3).astype(np.float32)

# Initialize the data
inputs[0].set_data_from_numpy(input0_data)

outputs.append(grpcclient.InferRequestedOutput('pred'))

# Test with outputs
results = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs)
print(results)
# Get the output arrays from the results
output0_data = results.as_numpy('pred')
print(output0_data)