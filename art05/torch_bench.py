
import argparse
import time
from contextlib import contextmanager
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel 
from transformers.utils import logging

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
    device = torch.device("cuda:0")
    shape = (batch_size, seq_len)
    inputs = {}
    inputs["input_ids"] = torch.randint(100, shape, dtype=torch.int64, device=device)
    if include_token_ids:
        inputs["token_type_ids"] = torch.ones(shape, dtype=torch.int64, device=device)
    inputs["attention_mask"] = torch.ones(shape, dtype=torch.int64, device=device)
    return inputs

def generate_multiple_inputs(batch_size, seq_len, include_token_ids, nb_inputs_to_gen):
    all_inputs = []
    for _ in range(nb_inputs_to_gen):
        inputs = generate_input(batch_size, seq_len, include_token_ids)
        all_inputs.append(inputs)
    return all_inputs
 
def print_timings(name, batch_size, seq_len, timings):
    mean_time = 1e3 * np.mean(timings)
    std_time = 1e3 * np.std(timings)
    min_time = 1e3 * np.min(timings)
    max_time = 1e3 * np.max(timings)
    median, percent_95_time, percent_99_time = 1e3 * np.percentile(timings, [50, 95, 99])
    print(
        f"[{name}] "
        f"[b={batch_size} s={seq_len}] "
        f"mean={mean_time:.2f}ms, "
        f"sd={std_time:.2f}ms, "
        f"min={min_time:.2f}ms, "
        f"max={max_time:.2f}ms, "
        f"median={median:.2f}ms, "
        f"95p={percent_95_time:.2f}ms, "
        f"99p={percent_99_time:.2f}ms"
    )

#
#    Inference utilities
#

def infer(model, input):
    with torch.no_grad(): 
        output = model(**input)
    return output

def launch_inference(model, inputs, nb_measures):
    assert type(inputs) == list
    assert len(inputs) > 0
    outputs = list()
    for batch_input in inputs:
        output = infer(model, batch_input)
        outputs.append(output)
    time_buffer = []
    for _ in range(nb_measures):
        with track_infer_time(time_buffer):
            _ = infer(model, inputs[0])
    return outputs, time_buffer  

#
#    Main program
#
  
def run(model_name, batch_size, seq_len, warmup, nb_measures, seed):
    assert torch.cuda.is_available()
    logging.set_verbosity_error()
    torch.random.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_names = tokenizer.model_input_names
    include_token_ids = "token_type_ids" in input_names 
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.cuda() 
    inputs = generate_multiple_inputs(batch_size, seq_len, include_token_ids, warmup)
    output, time_buffer = launch_inference(model, inputs, nb_measures) 
    print_timings(model_name, batch_size, seq_len, time_buffer) 

def parse_args(commands=None):
    parser = argparse.ArgumentParser(
        description="benchmark transformer PyTorch models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", 
        "--model", 
        required=True, 
        help="path to model or URL to Hugging Face hub") 
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

    model_name = args.model 
    batch_size = args.batch_size
    seq_len = args.seq_len
    warmup = args.warmup
    nb_measures = args.nb_measures
    seed = args.seed

    run(model_name, batch_size, seq_len, warmup, nb_measures, seed) 

if __name__ == "__main__":
    main()


