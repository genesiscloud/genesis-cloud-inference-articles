
import sys
from time import perf_counter
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
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 trt_bench_plan <plan_path>")

    plan_path = sys.argv[1]

    # read the plan
    with open(plan_path, "rb") as fp:
        plan = fp.read()

    # generate random input
    np.random.seed(1234)
    input = np.random.random(3 * 224 * 224)
    input = input.astype(np.float32)

    # initialize the TensorRT objects
    logger = trt.Logger()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    # create device buffers and TensorRT bindings
    stream = cuda.Stream()
    output = np.zeros((1000), dtype=np.float32)
    d_input = cuda.mem_alloc(input.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # copy input to device, run inference
    cuda.memcpy_htod(d_input, input)

    #  warm up
    for i in range(1, 10):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # benchmark
    start = perf_counter()
    for i in range(1, 100):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    end = perf_counter()
    print('Model {0}: elapsed time {1:.2f} ms'.format(plan_path, ((end - start) / 100) * 1000))

    # copy output to host
    cuda.memcpy_dtoh(output, d_output)
    
    # apply softmax and get Top-5 results
    output = softmax(output)
    top5p, top5v = topk(output, 5)

    # print results
    print("Top-5 results")
    for ind, val in zip(top5p, top5v):
        print("  {0} {1:.2f}%".format(ind, val * 100))

main()


