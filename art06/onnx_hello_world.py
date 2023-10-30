
import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

path = "./onnx/bert_base_uncased_b1_s16.onnx"

options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
provider = "CUDAExecutionProvider"
model = InferenceSession(path, options, providers=[provider]) 

input_ids = np.array([[101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
token_type_ids = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
inputs = {
    "input_ids": input_ids, 
    "token_type_ids": token_type_ids, 
    "attention_mask": attention_mask
}

output = model.run(None, inputs)
print(output[0])


