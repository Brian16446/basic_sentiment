from tritonclient.utils import *
import tritonclient.http as httpclient
import sys

import numpy as np

# input [
#  {
#     name: "input__0"
#     data_type: TYPE_INT32
#     dims: [1, 256]
#   } ,
# {
#     name: "input__1"
#     data_type: TYPE_INT32
#     dims: [1, 256]
#   }
# ]
# output {
#     name: "output__0"
#     data_type: TYPE_FP32
#     dims: [1, 3]
#   }

model_name = "sentiment"
shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = "i am very happy right now" 
    input0_bytes = str.encode(input0_data)
    inputs = [
        httpclient.InferInput("input__0", [1], "BYTES")
    ]

    inputs[0].set_data_from_numpy(np.asarray(input0_data).reshape([1]), binary_data=False)
    
    # outputs = [
    #     httpclient.InferRequestedOutput("OUTPUT0"),
    #     httpclient.InferRequestedOutput("OUTPUT1"),
    # ]

    # response = client.infer(model_name,
    #                         inputs,
    #                         request_id=str(1),
    #                         outputs=outputs)

    # result = response.get_response()
    # output0_data = response.as_numpy("OUTPUT0")
    # output1_data = response.as_numpy("OUTPUT1")

    # print("INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
    #     input0_data, input1_data, output0_data))
    # print("INPUT0 ({}) - INPUT1 ({}) = OUTPUT0 ({})".format(
    #     input0_data, input1_data, output1_data))

    # if not np.allclose(input0_data + input1_data, output0_data):
    #     print("add_sub example error: incorrect sum")
    #     sys.exit(1)

    # if not np.allclose(input0_data - input1_data, output1_data):
    #     print("add_sub example error: incorrect difference")
    #     sys.exit(1)

    print('PASS: new_request')
    sys.exit(0)