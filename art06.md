
# Article 6. Using ONNX Runtime for deployment and optimization of transformer models

This article describes using ONNX for inference using the pre-trained transformer
models. It focuses on these topics:

* conversion of transformer models from Hugging Face Model Hub into ONNX format
* running inference for ONNX models using the ONNX Runtime
* optimization of ONNX models, including use of 16-bit floating point precision
* comparative benchmarking of the original and optimized ONNX models

We assume that you will continue using the Genesis Cloud GPU-enabled instance that
you created and configured while studying the Article 5.

Various assets (source code, shell scripts, and data files) used in this article
can be found in the supporting
[GitHub repository](https://github.com/genesiscloud/genesis-cloud-inference-articles/tree/main/art06).

To run examples described in this article we recommend cloning the entire 
[repository](https://github.com/genesiscloud/genesis-cloud-inference-articles) on your Genesis Cloud instance.
The subdirectory `art06` must be made your current directory.

## Introduction

ONNX stands for _Open Neural Network eXchange_ and represents an open, vendor- and
platform- neutral format designed to represent deep learning models. ONNX defines
a common set of operators serving as building blocks for the construction of
deep learning models. Furthermore, it specifies a common file format for storing
model descriptions. The popular deep learning frameworks usually provide tools for
exporting deep learning models in the ONNX format. A variety of tools, runtimes,
and compilers is available for processing the models expressed as ONNX.

Further details about ONNX can be found at the [project site](https://onnx.ai/).

ONNX Runtime is an open source library designed to perform deep learning across a wide range
of frameworks, operating systems, and hardware platforms. ONNX Runtime loads the deep
learning model description represented in ONNX format and executes this model on
a given platform utilizing the platform hardware capabilities in the optimal way.

ONNX Runtime supports a wide range of hardware accelerators. This support is implemented
using the interchangeable hardware-specific _execution providers_. CUDA execution provider
is designed to support NVIDIA GPUs.

ONNX Runtime provides a common Application Programming Interface (API) with bindings
for various programming languages. Code examples in this article use Python and C++
API bindings.

The abbreviation ORT is frequently used to denote the ONNX Runtime.

Further details about ONNX Runtime can be found at the [project site](https://onnxruntime.ai/).

The summary of Python bindings for ONNX Runtime API is available at the 
[API documentation site](https://onnxruntime.ai/docs/api/python/api_summary.html).


## Step 1. Install ONNX runtime for Python

Code examples in this article require two Python packages:
`onnx` and `onnxruntime`. The package `onnx` implements general support for the ONNX
format and provides common functionality for manipulating ONNX models
like creating, reading, writing, validating, or transforming.
The package `onnxruntime` implements the Python bindings for
ONNX Runtime and can be used for executing the deep learning models.

To install the ONNX Python package, use this command:

```
python3 -m pip install onnx
```

To install the GPU-enabled version of ONNX runtime for Python, use this command:

```
python3 -m pip install onnxruntime-gpu==1.12.1
```

To briefly validate the installation, run these commands in the Python interpreter:

```
python3
>>> import onnx
>>> import onnxruntime
>>> import onnxruntime.transformers
```


## Step 2. Convert a PyTorch transformer model to ONNX

The Python program `onnx_build.py` can be used to convert a PyTorch
transformer model to ONNX format.

```
import argparse
import torch
from torch.onnx import TrainingMode
from transformers import AutoTokenizer, AutoModel

def generate_input(batch_size, seq_len, include_token_ids):
    shape = (batch_size, seq_len)
    inputs = {}
    inputs["input_ids"] = torch.randint(high=100, size=shape, dtype=torch.long, device="cuda")
    if include_token_ids:
        inputs["token_type_ids"] = torch.ones(size=shape, dtype=torch.long, device="cuda")
    inputs["attention_mask"] = torch.ones(size=shape, dtype=torch.long, device="cuda")
    return inputs

def convert_to_onnx(model, output_path, inputs):
    dynamic_axis = {}
    for k in inputs.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            output_path,
            verbose=False,               # default
            training=TrainingMode.EVAL,  # default
            input_names=list(inputs.keys()),
            output_names=["output"],
            opset_version=13,
            do_constant_folding=True,    # default
            dynamic_axes=dynamic_axis)


def run(model_name, output_path, batch_size, seq_len):
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_names = tokenizer.model_input_names
    include_token_ids = "token_type_ids" in input_names 
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.cuda() 
    inputs = generate_input(batch_size, seq_len, include_token_ids)
    convert_to_onnx(model, output_path, inputs)

def parse_args(commands=None):
    parser = argparse.ArgumentParser(
        description="convert transformer models to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", 
        "--model", 
        required=True, 
        help="path to model or URL to Hugging Face hub")
    parser.add_argument(
        "-o", 
        "--output", 
        required=True, 
        help="path to output ONNX file")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        help="optimal batch size",
        type=int)
    parser.add_argument(
        "-s",
        "--seq-len",
        default=16,
        help="optimal sequence length",
        type=int)
    args, _ = parser.parse_known_args(args=commands)
    return args

def main():
    args = parse_args()

    model_name = args.model
    onnx_path = args.output
    batch_size = args.batch_size
    seq_len = args.seq_len

    run(model_name, onnx_path, batch_size, seq_len)

if __name__ == "__main__":
    main()
```

This program is designed to support the wide variety of Hugging Face transformers.
It assumes that these models share some common features. Each model is assumed
to have two or three inputs, namely:

* `input_ids` - a sequence of token indices providing numerical representations
of tokens in the input text
* `token_type_ids` - optional, used when two different sequences are to be joined
in a single `input_ids` entry, for example, for tasks like classification on pairs of 
sentences or question answering
* `attention_masks` - a binary tensor indicating to the model which tokens should be 
attended to, and which should not; typically used for padding a variable length input
sequence to represent them as tensors with fixed dimensions.

Mode detailed discussion of these inputs as well as usage examples can be found
in the [previous article](./art05.md) of this series.

There are two important commonly used parameters, the _batch size_ and the _sequence length_.

Many hardware accelerators (including most of modern GPUs) use substantially smaller computing 
time for simultaneous processing of several inputs grouped in a single batch compared
to the sequential processing of the same inputs one by one. Applications dealing with
the multiple inputs simultaneously can benefit from this feature by batching the inputs.
For example, the question answering applications processing the large text corpora
belong to this category. For practical reasons, in such cases the number of inputs in 
one batch is typically fixed and specified as a batch size parameter.

The length of the input sequence that can be processed by transformer models
is usually limited, the maximum practical length being around 512. The processing
speed substantially degrades for the higher sequence length values, therefore it
is important to accurately specify the maximum expected sequence length used for 
the given task. (The shorter input sequences will be padded to this length.)

The amount of memory on the target hardware required to hold all the model data
depends on both batch size and sequence length and is higher for the higher values
of these parameters. As the model data must fit into the available memory, for each
model and sequence length there are certain limits on the batch size.

The above program uses these command line options to specify the input transformer model, 
output ONNX file, and configurable model parameters:

* `-m` path to model or URL to Hugging Face hub
* `-o` path to output ONNX file
* `-b` optimal batch size
* `-s` optimal sequence length

Although the batch size and the sequence length in the generated ONNX model
can be dynamically set at run time, during the conversion it is recommended 
to set the "optimal" values for these parameters that are close to the expected
run time values. The "optimal" parameters are specified using the `-b` and `-s` 
command line options of the conversion program.

The program uses the built-in PyTorch functionality to convert PyTorch models to ONNX.
It performs the following steps:

* parse the command line options
* verify that the PyTorch version supports CUDA
* create a tokenizer object for the specified model name
* fetch the names of model inputs from the tokenizer object
* create a model object for the specified model name
* use the model `eval` method to configure certain layer types for inference
* use the model `cuda` method to specify model execution on the current CUDA device
* generate the random content for all model input tensors
* for each input tensor, specify the batch size and the sequence length 
(corresponding to the dimensions 0 and 1 respectively) as dynamic axes
* for the output tensor, specify the batch size (dimension 0) as a dynamic axis
* use `torch.no_grad` context manager function to disable gradient calculations
* use `torch.onnx.export` function to export the model into ONNX format

The program uses constructors for the _Auto Classes_ `AutoTokenizer` and
`AutoModel` for creation of the tokenizer and model instances.
These constructors infer the exact model architecture from the specified name or
the path of the pre-trained model. The detailed description of the Auto Classes
can be found in the [API specification](https://huggingface.co/docs/transformers/model_doc/auto).

The `torch.onnx.export` function simulates one model inference run to convert
internally to the TorchScript format that will be further exported as ONNX.
The detailed specification of this function can be found in the 
[API specification](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export).

Axes (dimensions) of the input and output tensors specified as dynamic will be set
at run time. All the other axes are fixed during the ONNX generation and are set
to match exactly the shapes of model input and output tensors. This program
thus specifies that the batch size and the sequence will be set at run time.

In the examples of this article, we will use the `bert-base-uncased` transformer model.
We will place the generated ONNX files in the `onnx` subdirectory that must be created in advance:

```
mkdir -p ./onnx
```

We will experiment with different batch sizes and sequence lengths. We start with the minimum
values for these parameters:

```
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s16.onnx -b 1 -s 16
```

We can also generate ONNX representations for different parameter values.
We will use all combinations the batch size values of 1, 8, 64 and
the sequence length values of 16, 64, and 512. For this example, we will
generate a separate ONNX representation for each combination.

The shell script `onnx_build_all.sh` will create ONNX representations for all
chosen parameter combinations:

```
#!/bin/bash

python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s16.onnx -b 1 -s 16
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s64.onnx -b 1 -s 64
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s512.onnx -b 1 -s 512

python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b8_s16.onnx -b 8 -s 16
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b8_s64.onnx -b 8 -s 64
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b8_s512.onnx -b 8 -s 512

python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b64_s16.onnx -b 64 -s 16
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b64_s64.onnx -b 64 -s 64
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b64_s512.onnx -b 64 -s 512
```

Running this script is straightforward:

```
./onnx_build_all.sh
```


## Step 3. Inference benchmarking for the ONNX transformer model using Python

The Python program `onnx_bench.py` can be used to run inference for a transformer model
in ONNX format and output the performance metrics.

```
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
```

Some of the program command line options are:

* `-m` path to the ONNX file specifying the model
* `-b` batch size
* `-s` sequence length

The program loads the model from the specified ONNX file and repeatedly performs inference
with the given batch size and sequence length. For the best results, values of these
parameters shall match the "optimal" values specified during the model conversion to ONNX.

The program starts with a number of "warmup" runs specified via the `--warmup` option
followed by a number of measure runs specified via the `--nb-measures` option.
The wall clock time required for completion of each measure run is evaluated
and the respected statistics is displayed upon the completion of all runs.
The randomly generated input is used and the option `--seed` can be specified to initialize
the random number generator.

The program allows optional specification of the second ONNX model using the `-c` command option.
If the second model is specified, the program performs inference and prints timing statistics 
for both models. Then it compares outputs of both models and prints the computed difference
for each output. It is assumed that two models implement the similar network architecture and
have compatible inputs and outputs. Typically this option is used to assess the accuracy
of the optimized model compared to its original version. 

The inference output of one model is a list of outputs for each warm up run.
The outputs of one warm up run is a list of output tensors. 
The number and shapes of tensors in this list depend on the model architecture.
The inference output of one model is therefore a list of lists of tensors.

The program performs these steps:

* parse the command line options
* initialize the random number generator
* create the model from the ONNX specification using `CUDAExecutionProvider` as execution provider
* create Python function implementing model inference
* validate model inputs
* generate model input data for the specified batch size, sequence length, and number of warm up runs
* launch inference for the given model inference function, input data, and number of runs
* if the second model for comparison is not specified, print the timing statistics

If the second model for comparison is specified, the program performs these additional steps:

* create the second model from the ONNX specification using `CUDAExecutionProvider` as execution provider
* create Python function implementing inference for the second model
* validate inputs of the second model
* validate compatibility of inputs for two models
* generate the second model input data for the specified batch size, sequence length, and number of warm up runs
* launch inference for the given second model inference function, input data, and number of runs
* print the timing statistics for the first model
* print the timing statistics for the second model
* check accuracy by comparing the outputs of two models, print the computed differences

The function `create_model` implements ONNX model creation. It performs these steps:

* create the `onnxruntime.SessionOptions` object
* enable all optimizations
* create the `onnxruntime.InferenceSession` object for the selected ONNX file, options, and execution provider

The function `validate_inputs` validates the model input names are valid for transformer models.

The function `make_infer` creates a Python function implementing one inference run.
The generated function uses the `run` method of the `InferenceSession` object to perform inference.

The function `launch_inference` implements a sequence of ONNX inference runs.
It performs these steps:

* sequentially invoke the inference function for each warm up input, collect the outputs
* create a time buffer for storing the timing statistics
* sequentially invoke the inference function for the specified number of measurement runs,
track inference time for each run

The function `print_timings` computes and prints these timing statistics for the measurement runs:

* `mean` - mean
* `sd` - standard deviation
* `min` - minimum
* `max` - maximum
* `median` - median
* `95p` - 95% percentile
* `99p` - 99% percentile

The function `check_accuracy` compares pairwise all outputs of two models
and prints the difference values.

The function `compare_outputs` compares one pair of inference outputs of two models.
It performs these steps:

* validate consistency of tensor data types and shapes in each model output
* transform each model output to the format suitable for comparison
* for each output tensor, compute mean absolute difference for all warm up runs

The program uses these ORT API classes:

* `GraphOptimizationLevel` - graph optimization level for a session
* `SessionOptions` - configuration information for a session
* `InferenceSession` - main class used to run a model

To run benchmarking for the ONNX model with the batch size of 1 and sequence length of 16, use this command:

```
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s16.onnx -b 1 -s 16
```

On completion, the program will output the performance statistics in this format:

```
[./onnx/bert_base_uncased_b1_s16.onnx] mean=4.49ms, sd=0.18ms, min=4.43ms, max=8.50ms, median=4.47ms, 95p=4.53ms, 99p=4.76ms
```

You can use the shell script `onnx_bench_all.sh` to benchmark all generated ONNX models with various parameter values:

```
#!/bin/bash

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s16.onnx -b 1 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s64.onnx -b 1 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s512.onnx -b 1 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s16.onnx -b 8 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s64.onnx -b 8 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s512.onnx -b 8 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s16.onnx -b 64 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s64.onnx -b 64 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s512.onnx -b 64 -s 512
```

Running this script is straightforward:

```
./onnx_bench_all.sh
```

The script will display results similar to these:

```
[./onnx/bert_base_uncased_b1_s16.onnx] mean=4.45ms, sd=0.17ms, min=4.40ms, max=8.06ms, median=4.43ms, 95p=4.48ms, 99p=4.52ms
[./onnx/bert_base_uncased_b1_s64.onnx] mean=5.05ms, sd=0.27ms, min=4.94ms, max=9.61ms, median=5.02ms, 95p=5.10ms, 99p=5.38ms
[./onnx/bert_base_uncased_b1_s512.onnx] mean=8.77ms, sd=0.19ms, min=8.67ms, max=11.84ms, median=8.75ms, 95p=8.81ms, 99p=9.75ms
[./onnx/bert_base_uncased_b8_s16.onnx] mean=4.16ms, sd=0.21ms, min=4.03ms, max=7.76ms, median=4.11ms, 95p=4.27ms, 99p=4.40ms
[./onnx/bert_base_uncased_b8_s64.onnx] mean=7.09ms, sd=0.16ms, min=6.96ms, max=10.22ms, median=7.08ms, 95p=7.16ms, 99p=7.36ms
[./onnx/bert_base_uncased_b8_s512.onnx] mean=55.28ms, sd=0.38ms, min=54.64ms, max=60.30ms, median=55.25ms, 95p=55.56ms, 99p=55.83ms
[./onnx/bert_base_uncased_b64_s16.onnx] mean=11.84ms, sd=0.19ms, min=11.70ms, max=15.55ms, median=11.83ms, 95p=11.98ms, 99p=12.14ms
[./onnx/bert_base_uncased_b64_s64.onnx] mean=44.33ms, sd=0.63ms, min=43.15ms, max=51.95ms, median=44.47ms, 95p=44.75ms, 99p=45.01ms
[./onnx/bert_base_uncased_b64_s512.onnx] mean=440.57ms, sd=8.99ms, min=427.67ms, max=600.28ms, median=438.13ms, 95p=452.24ms, 99p=468.81ms
```


## Step 4. Optimization of the ONNX transformer model

ONNX Runtime includes the Transformer Model Optimization Tool designed for tuning
transformer models for best performance. This tool is available after installation
of ONNX Runtime and does not require a separate installation. The respective
API is implemented via Python `onnxruntime.transformers` package.
The tool supports optimization for models of BERT and GPT-2 families.

Further information about the Transformer Model Optimization Tool can be found in the public 
[GitHub repository](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md).

Python program `onnx_optimize.py` used this tool for optimization of transformer
models of BERT family:

```
import argparse
import logging
from onnxruntime.transformers import optimizer 
from onnxruntime.transformers.fusion_options import FusionOptions
from transformers import AutoConfig

def get_model_size(model_name):
    config = AutoConfig.from_pretrained(model_name)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    return num_attention_heads, hidden_size

def optimize_onnx(
        input_path,
        output_path,
        num_attention_heads=0,
        hidden_size=0):
    optimization_options = FusionOptions("bert")
    optimization_options.enable_gelu_approximation = False  # additional optimization
    # NOTE: For 'num_heads' and 'hidden_size' automatic detection with 0
    #     may not work with opset 13 or distilbert models
    optimized_model = optimizer.optimize_model(
        input=input_path,
        model_type="bert",
        use_gpu=True,
        opt_level=1,
        num_heads=num_attention_heads,
        hidden_size=hidden_size,
        optimization_options=optimization_options)
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(output_path) 

def run(model_name, input_path, output_path):
    num_attention_heads, hidden_size = get_model_size(model_name) 
    optimize_onnx(input_path, output_path, num_attention_heads, hidden_size)

def parse_args(commands=None):
    parser = argparse.ArgumentParser(
        description="optimize transformer models in ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", 
        "--model", 
        required=True, 
        help="path to model or URL to Hugging Face hub")
    parser.add_argument(
        "-i", 
        "--input", 
        required=True, 
        help="path to input ONNX file")
    parser.add_argument(
        "-o", 
        "--output", 
        required=True, 
        help="path to output optimized ONNX file")
    args, _ = parser.parse_known_args(args=commands)
    return args

def main():
    args = parse_args()

    model_name = args.model
    input_path = args.input
    output_path = args.output

    run(model_name, input_path, output_path)

if __name__ == "__main__":
    main()
```

This program inputs the ONNX model of the specified type, applies the optimization provided by the toolkit, 
and outputs the optimized version.

The program uses these command line options:

* `-m` path to model or URL to Hugging Face hub
* `-i` path to input ONNX file
* `-o` path to output optimized ONNX file

The program performs these steps:

* parse the command line options
* query the specified model to obtain the internal parameters critical for optimization,
including number of attention heads and number of hidden nodes
* create and setup the object specifying fusing options for the model of type `"bert"`
* invoke model optimizer to create an instance of optimized model
* save the optimized model to the output file

The program uses this toolkit API class:

* `onnxruntime.transformers.fusion_options.FusionOptions` - options of fusion in graph optimization

The Auto Class `transformers.AutoConfig` is used to query the pre-trained model parameters.

To perform optimization of the previously generated ONNX model with batch size 1 and sequence length 16,
use this command:

```
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s16.onnx -o ./onnx/bert_base_uncased_b1_s16_opt.onnx
```

You can use the shell script `onnx_optimize_all.sh` to optimize all generated ONNX models with various
batch size and sequence length values:

```
#!/bin/bash

python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s16.onnx -o ./onnx/bert_base_uncased_b1_s16_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s64.onnx -o ./onnx/bert_base_uncased_b1_s64_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s512.onnx -o ./onnx/bert_base_uncased_b1_s512_opt.onnx

python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b8_s16.onnx -o ./onnx/bert_base_uncased_b8_s16_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b8_s64.onnx -o ./onnx/bert_base_uncased_b8_s64_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b8_s512.onnx -o ./onnx/bert_base_uncased_b8_s512_opt.onnx

python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b64_s16.onnx -o ./onnx/bert_base_uncased_b64_s16_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b64_s64.onnx -o ./onnx/bert_base_uncased_b64_s64_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b64_s512.onnx -o ./onnx/bert_base_uncased_b64_s512_opt.onnx
```

Running this script is straightforward:

```
./onnx_optimize_all.sh
```


## Step 5. Inference benchmarking for the optimized ONNX transformer models

Now we can use the program `onnx_bench.py` to evaluate performance of the optimized models.

The shell script `onnx_bench_opt_all.sh` can be used to obtain timing statistics for the optimized models:

```
#!/bin/bash

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s16_opt.onnx -b 1 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s64_opt.onnx -b 1 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s512_opt.onnx -b 1 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s16_opt.onnx -b 8 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s64_opt.onnx -b 8 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s512_opt.onnx -b 8 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s16_opt.onnx -b 64 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s64_opt.onnx -b 64 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s512_opt.onnx -b 64 -s 512
```

Running this script is straightforward:

```
./onnx_bench_opt_all.sh
```

The script will display results similar to these:

```
[./onnx/bert_base_uncased_b1_s16_opt.onnx] mean=2.21ms, sd=0.15ms, min=2.15ms, max=4.83ms, median=2.19ms, 95p=2.28ms, 99p=2.33ms
[./onnx/bert_base_uncased_b1_s64_opt.onnx] mean=2.38ms, sd=0.14ms, min=2.33ms, max=5.56ms, median=2.37ms, 95p=2.42ms, 99p=2.53ms
[./onnx/bert_base_uncased_b1_s512_opt.onnx] mean=7.85ms, sd=0.28ms, min=7.70ms, max=13.72ms, median=7.81ms, 95p=7.93ms, 99p=8.81ms
[./onnx/bert_base_uncased_b8_s16_opt.onnx] mean=2.41ms, sd=0.19ms, min=2.33ms, max=5.77ms, median=2.38ms, 95p=2.54ms, 99p=2.68ms
[./onnx/bert_base_uncased_b8_s64_opt.onnx] mean=6.75ms, sd=0.36ms, min=6.62ms, max=16.53ms, median=6.71ms, 95p=6.84ms, 99p=7.68ms
[./onnx/bert_base_uncased_b8_s512_opt.onnx] mean=48.30ms, sd=1.03ms, min=46.70ms, max=77.32ms, median=48.25ms, 95p=48.60ms, 99p=48.89ms
[./onnx/bert_base_uncased_b64_s16_opt.onnx] mean=11.18ms, sd=0.67ms, min=10.71ms, max=27.80ms, median=11.17ms, 95p=11.36ms, 99p=11.53ms
[./onnx/bert_base_uncased_b64_s64_opt.onnx] mean=39.27ms, sd=0.61ms, min=38.70ms, max=47.69ms, median=39.23ms, 95p=39.54ms, 99p=39.82ms
[./onnx/bert_base_uncased_b64_s512_opt.onnx] mean=391.45ms, sd=7.35ms, min=381.42ms, max=497.30ms, median=388.50ms, 95p=399.88ms, 99p=419.74ms
```

To obtain timing statistics for both original and optimized models and compare the output accuracy,
the program `onnx_bench.py` can be invoked with the `-c` command line option.

The shell script `onnx_compare_all.sh` performs this operation for all previously generated and
optimized models:

```
#!/bin/bash

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s16_opt.onnx -c ./onnx/bert_base_uncased_b1_s16.onnx -b 1 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s64_opt.onnx -c ./onnx/bert_base_uncased_b1_s64.onnx -b 1 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s512_opt.onnx -c ./onnx/bert_base_uncased_b1_s512.onnx -b 1 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s16_opt.onnx -c ./onnx/bert_base_uncased_b8_s16.onnx -b 8 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s64_opt.onnx -c ./onnx/bert_base_uncased_b8_s64.onnx -b 8 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s512_opt.onnx -c ./onnx/bert_base_uncased_b8_s512.onnx -b 8 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s16_opt.onnx -c ./onnx/bert_base_uncased_b64_s16.onnx -b 64 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s64_opt.onnx -c ./onnx/bert_base_uncased_b64_s64.onnx -b 64 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s512_opt.onnx -c ./onnx/bert_base_uncased_b64_s512.onnx -b 64 -s 512
```

Running this script is straightforward:

```
./onnx_compare_all.sh
```

The script will display results similar to these:

```
[./onnx/bert_base_uncased_b1_s16_opt.onnx] mean=2.21ms, sd=0.10ms, min=2.16ms, max=4.51ms, median=2.20ms, 95p=2.25ms, 99p=2.34ms
[./onnx/bert_base_uncased_b1_s16.onnx] mean=4.39ms, sd=0.13ms, min=4.34ms, max=7.08ms, median=4.38ms, 95p=4.42ms, 99p=4.56ms
Difference [0] 0.33973
Difference [1] 0.18862
[./onnx/bert_base_uncased_b1_s64_opt.onnx] mean=2.34ms, sd=0.15ms, min=2.27ms, max=4.96ms, median=2.32ms, 95p=2.37ms, 99p=2.60ms
[./onnx/bert_base_uncased_b1_s64.onnx] mean=4.71ms, sd=0.16ms, min=4.65ms, max=7.69ms, median=4.69ms, 95p=4.76ms, 99p=5.14ms
Difference [0] 0.29251
Difference [1] 0.11606
[./onnx/bert_base_uncased_b1_s512_opt.onnx] mean=7.78ms, sd=0.11ms, min=7.65ms, max=9.74ms, median=7.76ms, 95p=7.85ms, 99p=8.00ms
[./onnx/bert_base_uncased_b1_s512.onnx] mean=8.56ms, sd=0.12ms, min=8.46ms, max=10.08ms, median=8.53ms, 95p=8.64ms, 99p=8.96ms
Difference [0] 0.31953
Difference [1] 0.16428
[./onnx/bert_base_uncased_b8_s16_opt.onnx] mean=2.49ms, sd=0.08ms, min=2.35ms, max=3.68ms, median=2.47ms, 95p=2.67ms, 99p=2.72ms
[./onnx/bert_base_uncased_b8_s16.onnx] mean=4.52ms, sd=0.13ms, min=4.32ms, max=6.62ms, median=4.51ms, 95p=4.58ms, 99p=4.87ms
Difference [0] 0.34052
Difference [1] 0.18928
[./onnx/bert_base_uncased_b8_s64_opt.onnx] mean=6.83ms, sd=0.11ms, min=6.73ms, max=8.92ms, median=6.81ms, 95p=6.90ms, 99p=7.12ms
[./onnx/bert_base_uncased_b8_s64.onnx] mean=7.25ms, sd=0.09ms, min=7.15ms, max=8.43ms, median=7.25ms, 95p=7.31ms, 99p=7.58ms
Difference [0] 0.29276
Difference [1] 0.11330
[./onnx/bert_base_uncased_b8_s512_opt.onnx] mean=48.38ms, sd=1.00ms, min=47.33ms, max=76.33ms, median=48.34ms, 95p=48.56ms, 99p=49.90ms
[./onnx/bert_base_uncased_b8_s512.onnx] mean=55.39ms, sd=0.30ms, min=55.08ms, max=60.18ms, median=55.33ms, 95p=55.54ms, 99p=55.70ms
Difference [0] 0.32175
Difference [1] 0.16403
[./onnx/bert_base_uncased_b64_s16_opt.onnx] mean=10.88ms, sd=0.42ms, min=10.66ms, max=21.96ms, median=10.83ms, 95p=11.10ms, 99p=12.40ms
[./onnx/bert_base_uncased_b64_s16.onnx] mean=11.53ms, sd=0.15ms, min=11.36ms, max=14.23ms, median=11.51ms, 95p=11.78ms, 99p=11.98ms
Difference [0] 0.34027
Difference [1] 0.18865
[./onnx/bert_base_uncased_b64_s64_opt.onnx] mean=39.39ms, sd=0.34ms, min=38.77ms, max=44.67ms, median=39.37ms, 95p=39.66ms, 99p=39.90ms
[./onnx/bert_base_uncased_b64_s64.onnx] mean=44.59ms, sd=0.28ms, min=43.98ms, max=48.74ms, median=44.58ms, 95p=44.87ms, 99p=44.99ms
Difference [0] 0.29339
Difference [1] 0.11305
[./onnx/bert_base_uncased_b64_s512_opt.onnx] mean=386.52ms, sd=3.82ms, min=383.64ms, max=421.22ms, median=385.71ms, 95p=387.94ms, 99p=416.57ms
[./onnx/bert_base_uncased_b64_s512.onnx] mean=435.48ms, sd=4.27ms, min=432.96ms, max=512.31ms, median=434.46ms, 95p=437.02ms, 99p=452.81ms
Difference [0] 0.32153
Difference [1] 0.16355
```


## Step 6. Inference performance summary

Results of the inference performance evaluation on the NVIDIA RTX 3080 instance 
obtained during the previous steps are summarized in this table:

```
Model                Batch    Sequence    Original    Optimized
                     size      length

---------------------------------------------------------------

bert-base-uncased       1         16         4.45        2.21
                        1         64         5.05        2.38
                        1        512         8.77        7.85

                        8         16         4.16        2.41
                        8         64         7.09        6.75
                        8        512        55.28       48.30

                       64         16        11.84       11.18
                       64         64        44.33       39.27
                       64        512       440.57      391.45
```

(The numbers that you can obtain by reproducing these tests might be slightly different.)

Analysis of these results reveals that:

* Using larger values of the batch size typically provides better performance
compared to the sequential processing of the same inputs one by one;
this performance gain is most prominent for the smaller sequence lengths and
becomes modest for the larger sequence lengths.

* Model optimization provides the significant
speedup for the smaller batch sizes and sequence lengths; otherwise the speedup
is moderate.


## Step 7. Build ONNX Runtime library for C++

To explore the C++ ORT API, we will need the respective library.
At this step we will build the ONNX Runtime library for C++ from the source code.

For this purpose we will use CMake, an extensible, open-source system 
that manages the build process in an operating system and in a compiler-independent manner.
At the time of writing, ONNX Runtime required minimum CMake version 3.18 and
only version 3.16 was available for the automated installation using the `apt`
tool. We will therefore install CMake from the product
[download page](https://cmake.org/download/). We will use version 3.22.2.

We will use the directory `~/vendor` for installation of the third party tools.
Create this directory is it does not yet exist and make it the current directory:

```
mkdir -p ~/vendor
cd ~/vendor
```

Then enter the following commands:

```
wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.sh
chmod +x cmake-3.22.2-linux-x86_64.sh
./cmake-3.22.2-linux-x86_64.sh
```

These commands will download the binary installer from the official CMake site and execute it.
The CMake tool will be installed in the directory `~/vendor/cmake-3.22.2-linux-x86_64`.
Add reference to the binary executables to `PATH` environment variable:

```
export PATH=/home/ubuntu/vendor/cmake-3.22.2-linux-x86_64/bin:$PATH
```

Verify the installation as follows:

```
cmake --version
```

You can remove the script `cmake-3.22.2-linux-x86_64.sh` once the installation is completed.

Now we can proceed with building the ONNX Runtime. We will use the directory `~/factory`
as a root for building the software and place the results in `~/vendor`. 

It is assumed that you have installed CUDA and cuDNN as described in
the [previous article](./art05.md). To use the NVIDIA CUDA compiler driver `nvcc`,
update the `PATH` environment variable before starting the build procedure:

```
export PATH=/usr/local/cuda/bin:$PATH
```


Create the build directory if it does not yet exist and make it your current directory:

```
mkdir -p ~/factory
cd ~/factory
```

The GitHub repository for the ONNX Runtime is located at

```
https://github.com/Microsoft/onnxruntime
```

We will use the source code of v1.12.1 which was the most recent release at the time of writing.
Clone the respective repository branch into the current directory:

```
git clone --recursive https://github.com/Microsoft/onnxruntime --branch v1.12.1
```

This command creates a subdirectory `onnxruntime` and places the cloned source code there.
Make this directory current and start the build script:

```
cd onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr
```

Note the `--cuda_home` and `--cudnn_home` command line options that specify locations of 
previously installed CUDA and cuDNN libraries.

The shell script `build.sh` provides no option for the automatic installation of the library components,
therefore we will copy the essential library components manually to the target directory using these commands:

```
cd ~/vendor
mkdir onnxruntime
cd onnxruntime

mkdir lib
cp ~/factory/onnxruntime/build/Linux/RelWithDebInfo/*.so* lib

mkdir include
cp ~/factory/onnxruntime/include/onnxruntime/core/session/*.h include
```

These commands will create the installation directory `~/vendor/onnxruntime` and
place the essential include files and shared libraries in subdirectories
`include` and `lib` respectively.

Before running the C++ programs built using the ONNX Runtime library, add the library
path to the list specified by `LD_LIBRARY_PATH` environment variable:

```
export LD_LIBRARY_PATH=~/vendor/onnxruntime/lib:$LD_LIBRARY_PATH
```


## Step 8. Build the "Hello World" C++ program using ORT

As the first example of using ORT with C++ we will explore
the simple C++ program that runs inference for the `bert-base-uncased` model
from the previously created `bert_base_uncased_b1_s16.onnx` file
using the tokenized `"Hello World"` sequence as input.

To obtain this input sequence, use the already familiar Python techniques:

```
python3
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> input = tokenizer("Hello World")
>>> print(input)
{'input_ids': [101, 7592, 2088, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

Details of tokenization and model input are described in Step 7. of Article 5.
Note that the sequence `[101, 7592, 2088, 102]` in `input_ids` contains tokenized text
surrounded with the special tokens `CLS` and `SEP`. The inputs `token_type_ids` and
`attention_mask` are filled with the values of 0 and 1 respectively. 

Before proceeding with the C++ code, we will introduce a simple
Python program `onnx_hello_world.py` with similar functionality.
We will compare the output of this program with that of the C++ version.

```
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
```

As at this point the reader should be familiar with the Python ORT API, we skip 
the detailed description of this program. All three inputs are padded with zero 
to the optimum sequence length associated with the ONNX model.

Use this command to run the program:

```
python3 onnx_hello_world.py
```

The program will display the output similar to this:

```
[[[-0.10648529  0.0184053   0.21049182 ... -0.3729446   0.41234058
   -0.42429084]
  [-0.07861218  0.036761    0.27857646 ... -0.41379213  0.45112044
   -0.51383024]
  [ 0.01957357  0.10374497  0.33755443 ... -0.396071    0.5069147
   -0.48847422]
  ...
  [-0.13679901 -0.02539987  0.34007016 ... -0.3174727   0.46983704
   -0.44701737]
  [-0.12651716 -0.0056464   0.32455254 ... -0.31961042  0.45559156
   -0.4515508 ]
  [-0.1373698  -0.00622558  0.32804158 ... -0.32903454  0.45128483
   -0.44218844]]]
```

The C++ program `onnx_hello_world.cpp` implements the similar functionality using
the C++ ORT API:

```
#include <cstdio>
#include <cstdint>
#include <vector>

#include "onnxruntime_cxx_api.h"

Ort::Value CreateTensor(int64_t *data, size_t size, const int64_t *shape, size_t rank) {
    auto info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<int64_t>(info, data, size, shape, rank);
    return tensor;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

    const char *modelPath = "./onnx/bert_base_uncased_b1_s16.onnx";

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

    Ort::Session session(env, modelPath, sessionOptions);

    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t numInputNodes = session.GetInputCount();
    std::vector<const char *> inputNodeNames(numInputNodes);
    std::vector<int64_t> inputNodeDims;

    printf("Number of inputs = %zu\n", numInputNodes);

    // iterate over all input nodes
    for (int i = 0; i < numInputNodes; i++) {
        // print input node names
        const char *inputName = session.GetInputName(i, allocator);
        printf("Input %d: name=%s\n", i, inputName);
        inputNodeNames[i] = inputName;

        // print input node types
        Ort::TypeInfo typeInfo = session.GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        printf("Input %d: type=%d\n", i, type);

        // print input shapes/dims
        inputNodeDims = tensorInfo.GetShape();
        printf("Input %d: num_dims=%zu\n", i, inputNodeDims.size());
        for (size_t j = 0; j < inputNodeDims.size(); j++) {
            printf("Input %d: dim %zu=%jd\n", i, j, inputNodeDims[j]);
        }
    }

    // print number of model input nodes
    size_t numOutputNodes = session.GetOutputCount();
    std::vector<const char *> outputNodeNames(numOutputNodes);
    std::vector<int64_t> outputNodeDims;

    printf("Number of outputs = %zu\n", numOutputNodes);

    // iterate over all output nodes
    for (int i = 0; i < numOutputNodes; i++) {
        // print output node names
        const char *outputName = session.GetOutputName(i, allocator);
        printf("Output %d: name=%s\n", i, outputName);
        outputNodeNames[i] = outputName;

        // print output node types
        Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        printf("Output %d: type=%d\n", i, type);

        // print output shapes/dims
        outputNodeDims = tensorInfo.GetShape();
        printf("Output %d: num_dims=%zu\n", i, outputNodeDims.size());
        for (size_t j = 0; j < outputNodeDims.size(); j++) {
            printf("Output %d: dim %zu=%jd\n", i, j, outputNodeDims[j]);
        }
    }

    int64_t inputIds[16] = {101, 7592, 2088, 102};
    int64_t tokenTypeIds[16] = {0, 0, 0, 0};
    int64_t attentionMask[16] = {1, 1, 1, 1};

    // create input tensor objects from data values
    const int64_t shape[] = {1, 16};
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(CreateTensor(inputIds, 16, shape, 2));
    inputTensors.push_back(CreateTensor(tokenTypeIds, 16, shape, 2));
    inputTensors.push_back(CreateTensor(attentionMask, 16, shape, 2));

    // run inference, get back output tensors
    auto outputTensors =
        session.Run(
            Ort::RunOptions{nullptr},
            inputNodeNames.data(),
            inputTensors.data(),
            3,
            outputNodeNames.data(),
            2);

    // get pointer to output tensor float values
    float *output = outputTensors[0].GetTensorMutableData<float>();

    // print some output values; assume output shape [1, 16, 768]
    for (int i = 0; i < 16; i++) {
        int k = i * 768;
        printf("output[%d, *] %g %g %g ... %g %g %g\n",
            i,
            output[k], output[k + 1], output[k + 2],
            output[k + 765], output[k + 766], output[k + 767]);
    }

    // release buffers allocated by ORT allocator
    for(const char *nodeName: inputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }
    for(const char *nodeName: outputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }

    printf("DONE\n");
    return 0;
}
```

The program uses these principal ORT API C++ types:

* `Ort::AllocatorWithDefaultOptions` - default memory allocator
* `Ort::Env` - environment object holding the logging state used by all other objects
* `Ort::MemoryInfo` - properties of a memory buffer
* `Ort::RunOptions` - optional parameters of a particular session run
* `Ort::Session` - inference session containing the runtime representation of an ONNX model
* `Ort::SessionOptions` - options object used when creating a new Session object
* `Ort::TypeInfo` - properties of an ONNX type
* `Ort::Value` - value representing the input or output data

The full C and C++ API description can be found at the
[documentation site](https://onnxruntime.ai/docs/api/c/index.html).

The program performs these steps:

* create an environment object, set logging level
* specify the ONNX file path for the model
* create a session options object
* create an allocator for node names
* for this session object enable all optimizations and specify CUDA execution provider
* create a session object for the specified environment, ONNX model path, and session options
* fetch the number of model input modes
* for each input node, fetch and print the name, type, and shape
* fetch the number of model output modes
* for each output node, fetch and print the name, type, and shape
* specify data values for input nodes
* create input tensor objects from these data values
* use the session object to run inference, get the output tensors
* get pointer to the data of the first input tensor
* selectively print data elements using this pointer
* release buffers allocated to hold the input and output node names

The shell script `build_onnx_hello_world.sh` can be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/onnx_hello_world \
    -I ~/vendor/onnxruntime/include \
    onnx_hello_world.cpp \
    -L ~/vendor/onnxruntime/lib \
    -lonnxruntime
```

Running this script is straightforward:

```
./build_onnx_hello_world.sh
```

To run this program, use the command:

```
./bin/onnx_hello_world
```

The program will display this output:

```
Number of inputs = 3
Input 0: name=input_ids
Input 0: type=7
Input 0: num_dims=2
Input 0: dim 0=-1
Input 0: dim 1=-1
Input 1: name=token_type_ids
Input 1: type=7
Input 1: num_dims=2
Input 1: dim 0=-1
Input 1: dim 1=-1
Input 2: name=attention_mask
Input 2: type=7
Input 2: num_dims=2
Input 2: dim 0=-1
Input 2: dim 1=-1
Number of outputs = 2
Output 0: name=output
Output 0: type=1
Output 0: num_dims=3
Output 0: dim 0=-1
Output 0: dim 1=-1
Output 0: dim 2=768
Output 1: name=1792
Output 1: type=1
Output 1: num_dims=2
Output 1: dim 0=-1
Output 1: dim 1=768
output[0, *] -0.106485 0.0184053 0.210492 ... -0.372945 0.412341 -0.424291
output[1, *] -0.0786122 0.036761 0.278576 ... -0.413792 0.45112 -0.51383
output[2, *] 0.0195736 0.103745 0.337554 ... -0.396071 0.506915 -0.488474
output[3, *] -0.161659 -0.131653 0.245575 ... -0.22964 0.426617 -0.46992
output[4, *] -0.0883617 -0.0493347 0.312765 ... -0.350901 0.469369 -0.513409
output[5, *] -0.0850462 -0.0834115 0.309916 ... -0.35645 0.477627 -0.511763
output[6, *] -0.0754291 -0.0809412 0.300442 ... -0.358102 0.480565 -0.501963
output[7, *] -0.0759627 -0.0834466 0.308569 ... -0.354377 0.481741 -0.485975
output[8, *] -0.0732201 -0.0791059 0.311949 ... -0.345431 0.469186 -0.473961
output[9, *] -0.0798875 -0.0529593 0.299392 ... -0.342595 0.469766 -0.478852
output[10, *] -0.0930583 -0.0654268 0.312835 ... -0.332374 0.480679 -0.449135
output[11, *] -0.121595 -0.0477084 0.309929 ... -0.326928 0.468086 -0.457563
output[12, *] -0.144058 -0.0354681 0.323213 ... -0.318405 0.465329 -0.449185
output[13, *] -0.136799 -0.0253999 0.34007 ... -0.317473 0.469837 -0.447017
output[14, *] -0.126517 -0.0056464 0.324553 ... -0.31961 0.455592 -0.451551
output[15, *] -0.13737 -0.00622558 0.328042 ... -0.329035 0.451285 -0.442188
DONE
```

The contents of the output tensor printed by this and the previous program match well.
Note that, besides the output tensor, this program fetches and prints names, shapes, and
data types of the model input and output nodes. Dimension values of -1 correspond to
the dynamic axes.


## Step 9. Inference benchmarking for the ONNX transformer model using C++

The C++ program `onnx_bench.cpp` can be used to run inference for a transformer model
in ONNX format and output the performance metrics.

```
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <chrono>

#include "onnxruntime_cxx_api.h"

//
//    Argument parsing
//

bool Atoi(const char *s, int &v) {
    char *p;
    long t = strtol(s, &p, 10);
    if (*p != '\0') {
        return false;
    }
    int r = int(t);
    if (long(r) != t) {
        return false;
    }
    v = r;
    return true;
}

//
//    Timer
//

class Timer {
public:
    Timer();
    ~Timer();
public:
    void Reset();
    void Start();
    void Stop();
    float Elapsed();
private:
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    float elapsed;
};

// construction/destruction

Timer::Timer(): elapsed(0.0f) { }

Timer::~Timer() { }

// interface

void Timer::Reset() {
    elapsed = 0.0f;
}

void Timer::Start() {
    start = std::chrono::steady_clock::now();
}

void Timer::Stop() {
    end = std::chrono::steady_clock::now();
    elapsed +=
        std::chrono::duration_cast<
            std::chrono::duration<float, std::milli>>(end - start).count();
}

float Timer::Elapsed() {
    return elapsed;
}

//
//    Input generator
//

void GenerateInput(int64_t *input, int volume) {
    for (int i = 0; i < volume; i++) {
        float value = static_cast<double>(std::rand()) / RAND_MAX;
        input[i] = static_cast<int64_t>(100.0 * value);
    }
}

//
//    ORT functions
//

Ort::Value CreateTensor(int64_t *data, size_t size, const int64_t *shape, size_t rank) {
    auto info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<int64_t>(info, data, size, shape, rank);
    return tensor;
}

void RunInference(
        Ort::Session &session,
        int numInputNodes,
        const char **inputNodeNames, 
        Ort::Value *inputTensors,
        int numOutputNodes,
        const char **outputNodeNames) {
    auto outputTensors = 
        session.Run(
            Ort::RunOptions{nullptr}, 
            inputNodeNames, 
            inputTensors, 
            numInputNodes, 
            outputNodeNames, 
            numOutputNodes);
}

//
//    Main program
//

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: onnx_bench <batch_size> <seq_len> [opt]\n");
        return 1;
    }

    // parse command line arguments

    int batchSize;
    if (!Atoi(argv[1], batchSize)) {
        fprintf(stderr, "Invalid batch size: '%s'\n", argv[1]);
        return 1;
    }
    int seqLen;
    if (!Atoi(argv[2], seqLen)) {
        fprintf(stderr, "Invalid sequence length: '%s'\n", argv[2]);
        return 1;
    }

    const char *optMode = nullptr;
    if (argc == 4) {
        optMode = argv[3];
        if (!strstr(optMode, "opt")) {
            fprintf(stderr, "Invalid optimization mode: '%s'\n", argv[3]);
            return 1;
        }
    }

    int warmup = 10;
    int measures = 100;

    char modelPath[128];
    if (optMode != nullptr) {
        sprintf(modelPath, "./onnx/bert_base_uncased_b%d_s%d_%s.onnx", batchSize, seqLen, optMode);
    } else {
        sprintf(modelPath, "./onnx/bert_base_uncased_b%d_s%d.onnx", batchSize, seqLen);
    }

    // create ORT session

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

    Ort::Session session(env, modelPath, sessionOptions);

    // fetch names of input and output nodes

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    assert(numInputNodes == 3);
    std::vector<const char *> inputNodeNames(numInputNodes);
    for (int i = 0; i < numInputNodes; i++) {
        const char *inputName = session.GetInputName(i, allocator);
        inputNodeNames[i] = inputName;
    }

    size_t numOutputNodes = session.GetOutputCount();
    assert(numOutputNodes == 2);
    std::vector<const char *> outputNodeNames(numOutputNodes);
    for (int i = 0; i < numOutputNodes; i++) {
        const char *outputName = session.GetOutputName(i, allocator);
        outputNodeNames[i] = outputName;
    }

    // set input data values

    int volume = batchSize * seqLen;
    std::vector<int64_t> inputIds(volume);
    std::vector<int64_t> tokenTypeIds(volume, 0);
    std::vector<int64_t> attentionMask(volume, 1);

    std::srand(1234);
    GenerateInput(inputIds.data(), volume);

    // create input tensor objects from data values

    const int64_t shape[] = {batchSize, seqLen};
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(CreateTensor(inputIds.data(), volume, shape, 2));
    inputTensors.push_back(CreateTensor(tokenTypeIds.data(), volume, shape, 2));
    inputTensors.push_back(CreateTensor(attentionMask.data(), volume, shape, 2));

    // warm up runs

    for (int i = 0; i < warmup; i++) {
        RunInference(
            session,
            numInputNodes, 
            inputNodeNames.data(), 
            inputTensors.data(), 
            numOutputNodes, 
            outputNodeNames.data());
    }

    // measured runs

    Timer timer;
    timer.Start();
    for (int i = 0; i < measures; i++) {
        RunInference(
            session,
            numInputNodes, 
            inputNodeNames.data(), 
            inputTensors.data(), 
            numOutputNodes, 
            outputNodeNames.data());
    }
    timer.Stop();
    float t = timer.Elapsed();
    printf("Model %s: elapsed time %f ms / %d = %f\n", modelPath, t, measures, t / float(measures));
    // record for automated extraction
    printf("#%s;%f\n", modelPath, t / float(measures)); 

    // release buffers allocated by ORT allocator
 
    for(const char *nodeName: inputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }
    for(const char *nodeName: outputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }

    return 0;
}
```

The program performs these steps:

* parse command line arguments, fetch batch size, sequence length, and optional optimization mode
* determine model path based on the command line arguments
* create an environment object, set logging level
* create a session options object
* for this session object enable all optimizations and specify CUDA execution provider
* create a session object for the specified environment, ONNX model path, and session options
* create an allocator for node names
* fetch the number of model input modes
* for each input node, fetch the name
* fetch the number of model output modes
* for each output node, fetch the name
* specify data values for input nodes, generate random input sequence of tokens
* create input tensor objects from these data values
* sequentially invoke the inference function for the specified number of warm up runs
* create a timer object
* sequentially invoke the inference function for the specified number of measurement runs,
track inference time for each run
* print the elapsed time and the mean time for one run
* release buffers allocated to hold the input and output node names

The shell script `build_onnx_bench.sh` can be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -O3 -o ./bin/onnx_bench \
    -I ~/vendor/onnxruntime/include \
    onnx_bench.cpp \
    -L ~/vendor/onnxruntime/lib \
    -lonnxruntime
```

Running this script is straightforward:

```
./build_onnx_bench.sh
```

The general usage pattern for this program is:

```
./bin/onnx_bench <batch_size> <seq_len> [opt]
```

where

* `<batch_size>` is the model optimal batch size
* `<seq_len>` is the model optimal sequence length
* the optional third argument indicates the optimization mode

For example, to run this program for the optimized model with batch size 1 and 
sequence length 16, use the command:

```
./bin/onnx_bench 1 16 opt
```

The program will display the output similar to this:

```
Model ./onnx/bert_base_uncased_b1_s16_opt.onnx: elapsed time 219.931152 ms / 100 = 2.199311
#./onnx/bert_base_uncased_b1_s16_opt.onnx;2.199311
```

The shell script `onnx_bench_cpp_all.sh` performs benchmarking of all previously
generated ONNX models:

```
#!/bin/bash

echo "#head;ORT (C++)"

./bin/onnx_bench 1 16
./bin/onnx_bench 1 64
./bin/onnx_bench 1 512
./bin/onnx_bench 8 16
./bin/onnx_bench 8 64
./bin/onnx_bench 8 512
./bin/onnx_bench 64 16
./bin/onnx_bench 64 64
./bin/onnx_bench 64 512

./bin/onnx_bench 1 16 opt
./bin/onnx_bench 1 64 opt
./bin/onnx_bench 1 512 opt
./bin/onnx_bench 8 16 opt
./bin/onnx_bench 8 64 opt
./bin/onnx_bench 8 512 opt
./bin/onnx_bench 64 16 opt
./bin/onnx_bench 64 64 opt
./bin/onnx_bench 64 512 opt
```

Running this script is straightforward:

```
./onnx_bench_cpp_all.sh >onnx_bench_cpp.log
```

The benchmarking log will be saved in `onnx_bench_cpp.log` that later will be
used for performance comparison of various deployment methods.


