import argparse
import time
from contextlib import contextmanager
import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

#
#    Benchmarking utilities
#

@contextmanager
def track_infer_time(buffer):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    buffer.append(end - start)  

def generate_input(batch_size, seq_len, include_token_ids):
    shape = (batch_size, seq_len)
    inputs = {}
    inputs["input_ids"] = np.random.randint(100, size=shape, dtype=np.int64)
    if include_token_ids:
        inputs["token_type_ids"] = np.zeros(shape, dtype=np.int64)
    inputs["attention_mask"] = np.ones(shape, dtype=np.int64)
    return inputs

def generate_multiple_inputs(batch_size, seq_len, include_token_ids, nb_inputs_to_gen):
    all_inputs = []
    for _ in range(nb_inputs_to_gen):
        inputs = generate_input(batch_size, seq_len, include_token_ids)
        all_inputs.append(inputs)
    return all_inputs
 
def print_timings(name, timings):
    mean_time = 1e3 * np.mean(timings)
    std_time = 1e3 * np.std(timings)
    min_time = 1e3 * np.min(timings)
    max_time = 1e3 * np.max(timings)
    median, percent_95_time, percent_99_time = 1e3 * np.percentile(timings, [50, 95, 99])
    print(
        f"[{name}] "
        f"mean={mean_time:.2f}ms, "
        f"sd={std_time:.2f}ms, "
        f"min={min_time:.2f}ms, "
        f"max={max_time:.2f}ms, "
        f"median={median:.2f}ms, "
        f"95p={percent_95_time:.2f}ms, "
        f"99p={percent_99_time:.2f}ms"
    )

def compare_outputs(output, output_2):
    def transform(x):
        n0 = len(x)
        n1 = len(x[0])
        y = []
        for i1 in range(n1):
            base_shape = x[0][i1].shape
            dtype = x[0][i1].dtype
            shape = (n0,) + base_shape
            t = np.empty(shape, dtype)
            for i0 in range(n0):
                v = x[i0][i1]
                assert v.shape == base_shape
                assert v.dtype == dtype
                t[i0] = v
            y.append(t)
        return y

    diff = []
    for x, y in zip(transform(output), transform(output_2)):
        d = np.mean(np.abs(x - y))
        diff.append(d)
    return diff 

def check_accuracy(output, output_2):
    diff = compare_outputs(output, output_2)
    for i, d in enumerate(diff):
        print(f"Difference [{i}] {d:.5f}") 

#
#    ONNX Runtime utilities
#

def create_model(path, provider):
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    return InferenceSession(path, options, providers=[provider]) 

def validate_inputs(model):
    valid_input_names = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
    ]
    inputs = model.get_inputs()
    include_token_ids = False
    for input in inputs:
        name = input.name
        assert name in valid_input_names
        if name == "token_type_ids":
            include_token_ids = True 
    return include_token_ids

def make_infer(model):
    def infer(inputs):
        return model.run(None, inputs)
    return infer

def launch_inference(infer, inputs, nb_measures):
    assert type(inputs) == list
    assert len(inputs) > 0
    outputs = list()
    for batch_input in inputs:
        output = infer(batch_input)
        outputs.append(output)
    time_buffer = []
    for _ in range(nb_measures):
        with track_infer_time(time_buffer):
            _ = infer(inputs[0])
    return outputs, time_buffer 

#
#    Main program
#
 
def run(onnx_path, onnx_path_2, batch_size, seq_len, verbose, warmup, nb_measures, seed):
    np.random.seed(seed) 
    provider = "CUDAExecutionProvider"
    model = create_model(onnx_path, provider)
    infer = make_infer(model)
    include_token_ids = validate_inputs(model)
    inputs = generate_multiple_inputs(batch_size, seq_len, include_token_ids, warmup)  
    output, time_buffer = launch_inference(infer, inputs, nb_measures) 
    del infer, model
    if onnx_path_2 is None:
        print_timings(onnx_path, time_buffer)
    else:
        model_2 = create_model(onnx_path_2, provider)
        infer_2 = make_infer(model_2)
        include_token_ids_2 = validate_inputs(model_2)
        assert include_token_ids_2 == include_token_ids
        output_2, time_buffer_2 = launch_inference(infer_2, inputs, nb_measures) 
        del infer_2, model_2
        print_timings(onnx_path, time_buffer)
        print_timings(onnx_path_2, time_buffer_2)
        check_accuracy(output, output_2)

def parse_args(commands=None):
    parser = argparse.ArgumentParser(
        description="benchmark transformer ONNX models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", 
        "--model", 
        required=True, 
        help="path to ONNX file")
    parser.add_argument(
        "-c",
        "--compare", 
        default=None, 
        help="path to another ONNX file")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        help="batch size",
        type=int)
    parser.add_argument(
        "-s",
        "--seq-len",
        default=16,
        help="sequence length",
        type=int)
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="display detailed information")
    parser.add_argument(
        "--warmup", 
        default=10, 
        help="# of inferences to warm each model", 
        type=int)
    parser.add_argument(
        "--nb-measures", 
        default=1000, 
        help="# of inferences for benchmarks", 
        type=int)
    parser.add_argument(
        "--seed", 
        default=1234, 
        help="seed for random inputs", 
        type=int)
    args, _ = parser.parse_known_args(args=commands)
    return args 

def main():
    args = parse_args()

    onnx_path = args.model
    onnx_path_2 = args.compare
    batch_size = args.batch_size
    seq_len = args.seq_len
    verbose = args.verbose
    warmup = args.warmup
    nb_measures = args.nb_measures
    seed = args.seed

    run(onnx_path, onnx_path_2, batch_size, seq_len, verbose, warmup, nb_measures, seed)

if __name__ == "__main__":
    main()

