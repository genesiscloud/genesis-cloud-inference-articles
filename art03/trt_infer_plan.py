
import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

def softmax(x):
    y = np.exp(x)
    sum = np.sum(y)
    y /= sum
    return y

def topk(x, k):
    idx = np.argsort(x)
    idx = idx[::-1][:k]
    return (idx, x[idx])

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 trt_infer_plan <plan_path> <input_path>")

    plan_path = sys.argv[1]
    input_path = sys.argv[2]

    print("Start " + plan_path)

    # read the plan
    with open(plan_path, "rb") as fp:
        plan = fp.read()

    # read the pre-processed image
    input = np.fromfile(input_path, np.float32)

    # read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # initialize the TensorRT objects
    logger = trt.Logger()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    # create device buffers and TensorRT bindings
    output = np.zeros((1000), dtype=np.float32)
    d_input = cuda.mem_alloc(input.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # copy input to device, run inference, copy output to host
    cuda.memcpy_htod(d_input, input)
    context.execute_v2(bindings=bindings)
    cuda.memcpy_dtoh(output, d_output)
    
    # apply softmax and get Top-5 results
    output = softmax(output)
    top5p, top5v = topk(output, 5)

    # print results
    print("Top-5 results")
    for ind, val in zip(top5p, top5v):
        print("  {0} {1:.2f}%".format(categories[ind], val * 100))

main()


