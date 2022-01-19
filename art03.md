
# Article 3. Deployment techniques for PyTorch models using TensorRT

This article covers using TensorRT for deployment of PyTorch models.

NVIDIA TensorRT is an SDK for high-performance deep learning inference on
NVIDIA GPU devices. It includes the inference engine and parsers for
handling various input network specification formats. TensorRT provides
application programmer interfaces (API) for C++ and Python. This article
will present example programs using both these languages.

To deploy PyTorch models using TensorRT, we will export them in ONNX format.
ONNX stands for Open Neural Network Exchange and is an open format built
to represent deep learning models in a framework-agnostic way. TensorRT
provides a specialized parser for importing ONNX models.

We assume that you will continue using the Genesis Cloud GPU-enabled instance that
you created and configured while studying the Article 1.

In particular, the following software must be installed and configured
as described in that article:

* CUDA 11.3.1
* cuDNN 8.2.1
* Python 3.x interpreter and package installer `pip`
* PyTorch 1.10.1 with torchvision 0.11.2

Various assets (source code, shell scripts, and data files) used in this article
can be found in the supporting
[GitHub repository](https://github.com/lxgo/genesis-kbase/tree/main/art03).


## Step 1. Install TensorRT

The version of TensorRT must be compatible with the chosen versions of CUDA and cuDNN.
For our choice of CUDA 11.3.1 and cuDNN 8.2.1 we will need TensorRT 8.0.3.
(The actual support matrix for TensorRT 8.x is available
[here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-822/support-matrix/index.html).)

To access TensorRT, you should register as a member of the 
[NVIDIA Developer Program](https://developer.nvidia.com/developer-program).

To download the TensorRT distribution, visit the official
[download site](https://developer.nvidia.com/nvidia-tensorrt-download).
Choose "TensorRT 8", then agree to the "NVIDIA TensorRT License Agreement"
and choose "TensorRT 8.0 GA Update 1" ("GA" stands for "General Availability").
Select and download "TensorRT 8.0.3 GA for Ubuntu 20.04 and CUDA 11.3 DEB local repo package".
You will get a DEB repo file; at the time of writing this article its name was:

```
nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831_1-1_amd64.deb
```

Place it in a scratch directory on you instance (we use `~/transit` in this series of articles),
then proceed with installation by entering these commands:

```
sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
```

Then install Python bindings for TensorRT API:

```
python3 -m pip install numpy
sudo apt-get install python3-libnvinfer-dev
```

Verify the installation using the command:

```
dpkg -l | grep TensorRT
```

Detailed installation instructions can be found on the official
["Installing TensorRT"](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian) page.


## Step 2. Install PyCUDA package

PyCUDA is a Python package implementing access to the CUDA API from Python.
The Python programs described in this article require PyCUDA for
accessing the basic CUDA functionality like managing CUDA device
memory buffers.

Before starting the PyCUDA installation make sure that the NVIDIA CUDA compiler
driver `nvcc` is accessible by entering the command:

```
nvcc --version
```

If this command files, update the `PATH` environment variable:

```
export PATH=/usr/local/cuda/bin:$PATH
```

To install PyCUDA, enter the command:

```
python3 -m pip install pycuda
```


## Step 3. Convert a PyTorch model to ONNX

We will continue using the torchvision image classification models for our examples.
As the first step, we will demonstrate conversion of the already familiar ResNet50 model
to ONNX format.

The Python program `generate_onnx_resnet50.py` serves this purpose.

```
import torch
import torchvision.models as models

input = torch.rand(1, 3, 224, 224)

model = models.resnet50(pretrained=True)
model.eval()
output = model(input)
torch.onnx.export(model, input, "./onnx/resnet50.onnx", export_params=True)
```

This program:

* creates a dummy input tensor
* creates a pretrained ResNet50 model
* sets the model in evaluation (inference) mode
* runs dummy inference for the model
* exports model to ONNX format and saves result in a file

We store generated ONNX files in the subdirectory `onnx` which
must be created before running the program:

```
mkdir -p onnx
```

To run this program, use the command:

```
python3 infer_generate_onnx_resnet50.py
```

The program will produce a file `resnet50.onnx` containing the ONNX model representation.

---

The Python program `generate_onnx_all.py` can be used to produce ONNX descriptions
for all considered torchvision image classification models.

```
import torch
import torchvision.models as models

MODELS = [
    ('alexnet', models.alexnet),

    ('densenet121', models.densenet121),
    ('densenet161', models.densenet161),
    ('densenet169', models.densenet169),
    ('densenet201', models.densenet201),

    ('mnasnet0_5', models.mnasnet0_5),
    ('mnasnet1_0', models.mnasnet1_0),

    ('mobilenet_v2', models.mobilenet_v2),
    ('mobilenet_v3_large', models.mobilenet_v3_large),
    ('mobilenet_v3_small', models.mobilenet_v3_small),

    ('resnet18', models.resnet18),
    ('resnet34', models.resnet34),
    ('resnet50', models.resnet50),
    ('resnet101', models.resnet101),
    ('resnet152', models.resnet152),

    ('resnext50_32x4d', models.resnext50_32x4d),
    ('resnext101_32x8d', models.resnext101_32x8d),

    ('shufflenet_v2_x0_5', models.shufflenet_v2_x0_5),
    ('shufflenet_v2_x1_0', models.shufflenet_v2_x1_0),

    ('squeezenet1_0', models.squeezenet1_0),
    ('squeezenet1_1', models.squeezenet1_1),

    ('vgg11', models.vgg11),
    ('vgg11_bn', models.vgg11_bn),
    ('vgg13', models.vgg13),
    ('vgg13_bn', models.vgg13_bn),
    ('vgg16', models.vgg16),
    ('vgg16_bn', models.vgg16_bn),
    ('vgg19', models.vgg19),
    ('vgg19_bn', models.vgg19_bn),

    ('wide_resnet50_2', models.wide_resnet50_2),
    ('wide_resnet101_2', models.wide_resnet101_2),
]

def generate_model(name, builder):
    print('Generate', name)
    input = torch.rand(1, 3, 224, 224)
    model = builder(pretrained=True)
    model.eval()
    output = model(input)
    onnx_path = './onnx/' + name + '.onnx'
    torch.onnx.export(model, input, onnx_path, export_params=True) 

for name, model in MODELS:
    generate_model(name, model)  
```

To run this program, enter the following commands:

```
mkdir -p onnx
python3 generate_onnx_all.py
```


## Step 4. Convert ONNX format to TensorRT plan using Python

To perform inference of the ONNX model using TensorRT, it must be pre-processed using
the TensorRT ONNX parser. We will start with conversion of the ONNX representation to
the TensorRT plan. The TensorRT plan is a serialized form of a TensorRT engine.
The TensorRT engine represents the model optimized for execution on a chosen
CUDA device.

The Python program `trt_onnx_parser.py` serves this purpose.

```
import sys
import tensorrt as trt

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 trt_onnx_parser.py <input_onnx_path> <output_plan_path>")

    onnx_path = sys.argv[1]
    plan_path = sys.argv[2]

    logger = trt.Logger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * 1024 * 1024
    # setup timimg cache just to avoid warnings at runtime
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, False)

    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        sys.exit("ONNX parse error")

    plan = builder.build_serialized_network(network, config)
    with open(plan_path, "wb") as fp:
        fp.write(plan)

    print("DONE")

main()
```

The Python package `tensorrt` implements TensorRT Python API and
provides a collection of Python object classes used to handle
various aspects of TensorRT inference and model parsing.

This program uses the following TensorRT API object classes:

* `Logger` - logger used by several other object classes
* `Builder` - a factory used to create several other classes
* `INetworkDefinition` - representation of TensorRT networks (models)
* `IBuilderConfig` - a class used to hold configuration parameters for `Builder`
* `ITimingCahche` - a class handling timing information
* `OnnxParser` - a class used for parsing ONNX models into TensorRT network definitions
* `IHostMemory` - representation of buffers in a host memory

The program performs the following steps:

* creates `logger: Logger` representing a logger instance
* creates `builder: Builder` representing a builder instance
* uses `builder` to create `network: INetworkDefinition` representing 
an empty network instance
* uses `builder` to create `config: IBuilderConfig` representing 
a builder configuration instance
* sets the `max_workspace_size` configuration parameter representing
the maximum workspace size that can be used by inference algorithms
* sets up timing cache `cache: ITimingCache`
* creates `parser: OnnxParser` representing an ONNX parser instance;
reference to the previously created empty network definition is attached 
to the parser
* uses `parser` to parse the input ONNX file and convert it to
the TensorRT network definition; assigns the parsing result
the attached network definition object
* uses `builder` to create `plan: IHostMemory` representing
a serialized network (plan) stored in a host memory buffer
* saves the plan in the output file

Note that in this example the timing cache is set up only to avoid
warning messages at runtime.

The program has two command line arguments: a path to the input ONNX file and
a path to the output TensorRT plan file.

We store generated plan files in the subdirectory `plan` which
must be created before running the program:

```
mkdir -p plan
```

To run this program for conversion of ResNet50 ONNX representation, use the command:

```
python3 trt_onnx_parser.py ./onnx/resnet50.onnx ./plan/resnet50.plan
```

The Python program `trt_onnx_parser_all.py` can be used to produce ONNX representations
for all considered torchvision image classification models.

```
import sys
import tensorrt as trt

MODELS = [
    'alexnet',

    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',

    'mnasnet0_5',
    'mnasnet1_0',

    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',

    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',

    'resnext50_32x4d',
    'resnext101_32x8d',

    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',

    'squeezenet1_0',
    'squeezenet1_1',

    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',

    'wide_resnet50_2',
    'wide_resnet101_2',
]


def setup_builder():
    logger = trt.Logger()
    builder = trt.Builder(logger)
    return (logger, builder)

def generate_plan(logger, builder, name):
    print('Generate TensorRT plan for ' + name)

    onnx_path = './onnx/' + name + '.onnx'
    plan_path = './plan/' + name + '.plan'

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * 1024 * 1024
    # setup timing cache just to avoid warnings at runtime
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, False)

    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        sys.exit('ONNX parse error')

    plan = builder.build_serialized_network(network, config)
    with open(plan_path, "wb") as fp:
        fp.write(plan)

def main():
    logger, builder = setup_builder()
    for name in MODELS:
        generate_plan(logger, builder, name)
    print('DONE')

main() 
```

To run this program, enter the following commands:

```
mkdir -p onnx
python3 trt_onnx_parser_all.py
```


## Step 5. Convert ONNX format to TensorRT plan using C++

Conversion of the ONNX representation to TensorRT plan can be also
implemented using the TensorRT C++ API.

The C++ program `trt_onnx_parser.cpp` serves this purpose.

```
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "common.h"

// wrapper class for ONNX parser

class OnnxParser {
public:
    OnnxParser();
    ~OnnxParser();
public:
    void Init();
    void Parse(const char *onnxPath, const char *planPath);
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IBuilder> m_builder;
    UniquePtr<nvinfer1::INetworkDefinition> m_network;
    UniquePtr<nvinfer1::IBuilderConfig> m_config;
    UniquePtr<nvinfer1::ITimingCache> m_cache;
    UniquePtr<nvonnxparser::IParser> m_parser;
};

OnnxParser::OnnxParser(): m_active(false) { }

OnnxParser::~OnnxParser() { }

void OnnxParser::Init() {
    assert(!m_active);
    m_builder.reset(nvinfer1::createInferBuilder(m_logger));
    if (m_builder == nullptr) {
        Error("Error creating infer builder");
    }
    auto networkFlags = 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_network.reset(m_builder->createNetworkV2(networkFlags));
    if (m_network == nullptr) {
        Error("Error creating network");
    }
    m_config.reset(m_builder->createBuilderConfig());
    if (m_config == nullptr) {
        Error("Error creating builder config");
    }
    m_config->setMaxWorkspaceSize(256 * 1024 * 1024);
    m_parser.reset(nvonnxparser::createParser(*m_network, m_logger));
    if (m_parser == nullptr) {
        Error("Error creating ONNX parser");
    }
    // setup timing cache just to avoid warnings at runtime
    m_cache.reset(m_config->createTimingCache(nullptr, 0));
    if (m_cache == nullptr) {
        Error("Error creating timing cache");
    }
    bool ok = m_config->setTimingCache(*m_cache, false);
    if (!ok) {
        Error("Error setting timing cache");
    }
}

void OnnxParser::Parse(const char *onnxPath, const char *planPath) {
    bool ok = m_parser->parseFromFile(onnxPath, static_cast<int>(m_logger.SeverityLevel()));
    if (!ok) {
        Error("ONNX parse error");
    }
    UniquePtr<nvinfer1::IHostMemory> plan(m_builder->buildSerializedNetwork(*m_network, *m_config));
    if (plan == nullptr) {
        Error("Network serialization error");
    }
    const void *data = plan->data();
    size_t size = plan->size();
    FILE *fp = fopen(planPath, "wb");
    if (fp == nullptr) {
        Error("Failed to create file %s", planPath);
    }
    fwrite(data, 1, size, fp);
    fclose(fp);
}

// main program

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: trt_onnx_parser <input_onnx_path> <output_plan_path>\n");
        return 1;
    }
    const char *onnxPath = argv[1];
    const char *planPath = argv[2];
    printf("Generate TensorRT plan for %s\n", onnxPath);
    OnnxParser parser;
    parser.Init();
    parser.Parse(onnxPath, planPath);   
    return 0;
}
```

The program is functionally similar to previously described Python program `trt_onnx_parser.py`.
Plans generated using the Python and C++ program versions are interchangeable; each plan
can be used for the subsequent inference with Python and C++ programs described in
this article.

The program uses the TensorRT C++ API specified in two header files:

* `NvInfer.h` defines interface to the TensorRT inference engine encapsulated
in the `nvinfer1` namespace
* `NvOnnxParser.h` defines interface to the TensorRT ONNX parser encapsulated
in the `nvonnxparser` namespace

This program uses the following TensorRT API object classes:

* `nvinfer1::ILogger` - logger used by several other object classes
* `nvinfer1::IBuilder` - a factory used to create several other classes
* `nvinfer1::INetworkDefinition` - representation of TensorRT networks (models)
* `nvinfer1::IBuilderConfig` - a class used to hold configuration parameters for `IBuilder`
* `nvinfer1::ITimingCahche` - a class handling timing information
* `nvonnxparser::IParser` - a class used for parsing ONNX models into TensorRT network definitions
* `nvinfer1::IHostMemory` - representation of buffers in a host memory

Class `OnnxParser` holds smart pointers to instances of these objects.
It exposes two principal public methods: `Init` and `Parse`.

The `Init` method performs the following steps:

* creates `m_builder` representing a builder instance
* uses `m_builder` to create `m_network` representing 
an empty network instance
* uses `m_builder` to create `m_config` representing 
a builder configuration instance
* sets the `maxWorkspaceSize` configuration parameter representing
the maximum workspace size that can be used by inference algorithms
* sets up timing cache `cache`
* creates `m_parser` representing an ONNX parser instance;
reference to the previously created empty network definition is attached 
to the parser

Note that in this example the timing cache is set up only to avoid
warning messages at runtime.

The `Parse` method performs the following steps:

* uses `m_parser` to parse the input ONNX file and convert it to
the TensorRT network definition; assigns the parsing result
the attached network definition object
* uses `m_builder` to create `plan` representing
a serialized network (plan) stored in a host memory buffer
* saves the plan in the output file

The shell script `build_trt_onnx_parser.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/trt_onnx_parser \
    -I /usr/local/cuda/include \
    trt_onnx_parser.cpp common.cpp \
    -L /usr/local/cuda/lib64 -lnvonnxparser -lnvinfer -lcudart 
```

Running this script is straightforward:

```
./build_trt_onnx_parser.sh
```

The program has two command line arguments: a path to the input ONNX file and
a path to the output TensorRT plan file.

To run this program for conversion of ResNet50 ONNX representation, use the command:

```
./bin/trt_onnx_parser ./onnx/resnet50.onnx ./plan/resnet50.plan
```

The shell script `trt_onnx_parser_all.sh` uses the C++ program to generate 
ONNX representations for all considered torchvision image classification files:

```
#!/bin/bash

./bin/trt_onnx_parser ./onnx/alexnet.onnx ./plan/alexnet.plan
./bin/trt_onnx_parser ./onnx/densenet121.onnx ./plan/densenet121.plan
./bin/trt_onnx_parser ./onnx/densenet161.onnx ./plan/densenet161.plan
./bin/trt_onnx_parser ./onnx/densenet169.onnx ./plan/densenet169.plan
./bin/trt_onnx_parser ./onnx/densenet201.onnx ./plan/densenet201.plan
./bin/trt_onnx_parser ./onnx/mnasnet0_5.onnx ./plan/mnasnet0_5.plan
./bin/trt_onnx_parser ./onnx/mnasnet1_0.onnx ./plan/mnasnet1_0.plan
./bin/trt_onnx_parser ./onnx/mobilenet_v2.onnx ./plan/mobilenet_v2.plan
./bin/trt_onnx_parser ./onnx/mobilenet_v3_large.onnx ./plan/mobilenet_v3_large.plan
./bin/trt_onnx_parser ./onnx/mobilenet_v3_small.onnx ./plan/mobilenet_v3_small.plan
./bin/trt_onnx_parser ./onnx/resnet101.onnx ./plan/resnet101.plan
./bin/trt_onnx_parser ./onnx/resnet152.onnx ./plan/resnet152.plan
./bin/trt_onnx_parser ./onnx/resnet18.onnx ./plan/resnet18.plan
./bin/trt_onnx_parser ./onnx/resnet34.onnx ./plan/resnet34.plan
./bin/trt_onnx_parser ./onnx/resnet50.onnx ./plan/resnet50.plan
./bin/trt_onnx_parser ./onnx/resnext101_32x8d.onnx ./plan/resnext101_32x8d.plan
./bin/trt_onnx_parser ./onnx/resnext50_32x4d.onnx ./plan/resnext50_32x4d.plan
./bin/trt_onnx_parser ./onnx/shufflenet_v2_x0_5.onnx ./plan/shufflenet_v2_x0_5.plan
./bin/trt_onnx_parser ./onnx/shufflenet_v2_x1_0.onnx ./plan/shufflenet_v2_x1_0.plan
./bin/trt_onnx_parser ./onnx/squeezenet1_0.onnx ./plan/squeezenet1_0.plan
./bin/trt_onnx_parser ./onnx/squeezenet1_1.onnx ./plan/squeezenet1_1.plan
./bin/trt_onnx_parser ./onnx/vgg11.onnx ./plan/vgg11.plan
./bin/trt_onnx_parser ./onnx/vgg11_bn.onnx ./plan/vgg11_bn.plan
./bin/trt_onnx_parser ./onnx/vgg13.onnx ./plan/vgg13.plan
./bin/trt_onnx_parser ./onnx/vgg13_bn.onnx ./plan/vgg13_bn.plan
./bin/trt_onnx_parser ./onnx/vgg16.onnx ./plan/vgg16.plan
./bin/trt_onnx_parser ./onnx/vgg16_bn.onnx ./plan/vgg16_bn.plan
./bin/trt_onnx_parser ./onnx/vgg19.onnx ./plan/vgg19.plan
./bin/trt_onnx_parser ./onnx/vgg19_bn.onnx ./plan/vgg19_bn.plan
./bin/trt_onnx_parser ./onnx/wide_resnet101_2.onnx ./plan/wide_resnet101_2.plan
./bin/trt_onnx_parser ./onnx/wide_resnet50_2.onnx ./plan/wide_resnet50_2.plan
```

Running this script is straightforward:

```
mkdir -p ./plan
trt_onnx_parser_all.sh
```


## Step 6. Run TensorRT inference using Python

The inference programs in Python and C++ described in the rest of this
article reuse several files introduced in Articles 1 and 2.
These include:

* `imagenet_classes.txt` - class descriptions for ImageNet labels (Article 1)
* `./data/husky01.dat` - pre-processed input tensor for the husky image (Article 2)

See the respective articles for details on obtaining these files.

The Python program `trt_infer_plan.py` implements TensorRT inference using
the previously generated TensorRT plan and a pre-processed input image.

```
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
        sys.exit("Usage: python3 trt_infer_plan.py <plan_path> <input_path>")

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
```

This program uses the following TensorRT API object classes:

* `Logger` - logger used by several other object classes
* `Runtime` - used to deserialize TensorRT plans to TensorRT CUDA engines
* `ICudaEngine` - engine for executing inference on built networks
* `IExecutionContext` - context for executing inference using CUDA engine

The program performs the following steps:

* reads the plan
* reads the pre-processed image
* reads the ImageNet categories
* creates `logger: Logger` representing a logger instance
* creates `runtime: Runtime` representing a runtime instance
* uses `runtime` to deserialize the plan into `engine: ICudaEngine`
* creates `context: IExecutionContext` for the `engine`
* obtains a CUDA stream reference
* allocates a Numpy array to hold the output data on the host
* allocates device memory buffers for the input and output tensors
* specifies input/output bindings as a list holding addresses
of all input and output buffers
* copies the input tensor from host to device
* runs inference for the `context` with the specified bindings and CUDA stream handle
* copies the output tensor from device to host
* applies the softmax transformation to the output
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities in a human-readable form

The program has two command line arguments: a path to the TensorRT plan file and
a path to the file containing the pre-processed input image.

To run this program for the previously created ResNet50 plan and husky image, 
use the command:

```
python3 trt_infer_plan.py ./plan/resnet50.plan ./data/husky01.dat
```

The program output will look like:

```
Siberian husky 49.52%
Eskimo dog 42.90%
malamute 5.87%
dogsled 1.22%
Saint Bernard 0.32%
```

## Step 7. Run TensorRT inference using C++

The inference with TensorRT models can be also implemented using the TensorRT C++ API.

The C++ program `trt_infer_plan.cpp` serves this purpose.

```
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <NvInfer.h>

#include "common.h"

// wrapper class for inference engine

class Engine {
public:
    Engine();
    ~Engine();
public:
    void Init(const std::vector<char> &plan);
    void Infer(const std::vector<float> &input, std::vector<float> &output);
    void DiagBindings();
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IRuntime> m_runtime;
    UniquePtr<nvinfer1::ICudaEngine> m_engine;
};

Engine::Engine(): m_active(false) { }

Engine::~Engine() { }

void Engine::Init(const std::vector<char> &plan) {
    assert(!m_active);
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (m_runtime == nullptr) {
        Error("Error creating infer runtime");
    }
    m_engine.reset(m_runtime->deserializeCudaEngine(plan.data(), plan.size(), nullptr));
    if (m_engine == nullptr) {
        Error("Error deserializing CUDA engine");
    }
    m_active = true;
}

void Engine::Infer(const std::vector<float> &input, std::vector<float> &output) {
    assert(m_active);
    UniquePtr<nvinfer1::IExecutionContext> context;
    context.reset(m_engine->createExecutionContext());
    if (context == nullptr) {
        Error("Error creating execution context");
    }
    CudaBuffer<float> inputBuffer;
    inputBuffer.Init(3 * 224 * 224);
    assert(inputBuffer.Size() == input.size());
    inputBuffer.Put(input.data());
    CudaBuffer<float> outputBuffer;
    outputBuffer.Init(1000);
    void *bindings[2];
    bindings[0] = inputBuffer.Data();
    bindings[1] = outputBuffer.Data();
    bool ok = context->executeV2(bindings);
    if (!ok) {
        Error("Error executing inference");
    }
    output.resize(outputBuffer.Size());
    outputBuffer.Get(output.data());
}

void Engine::DiagBindings() {
    int nbBindings = static_cast<int>(m_engine->getNbBindings());
    printf("Bindings: %d\n", nbBindings);
    for (int i = 0; i < nbBindings; i++) {
        const char *name = m_engine->getBindingName(i);
        bool isInput = m_engine->bindingIsInput(i);
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        std::string fmtDims = FormatDims(dims);
        printf("  [%d] \"%s\" %s [%s]\n", i, name, isInput ? "input" : "output", fmtDims.c_str());
    }
}

// I/O utilities

void ReadClasses(const char *path, std::vector<std::string> &classes) {
    std::string line;
    std::ifstream ifs(path, std::ios::in);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
    ifs.close();
}

void ReadPlan(const char *path, std::vector<char> &plan) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg();
    plan.resize(size);
    ifs.seekg(0, ifs.beg);
    ifs.read(plan.data(), size);
    ifs.close();
}

void ReadInput(const char *path, std::vector<float> &input) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    size_t size = 3 * 224 * 224;
    input.resize(size);
    ifs.read(reinterpret_cast<char *>(input.data()), size * sizeof(float));
    ifs.close();
}

void PrintOutput(const std::vector<float> &output, const std::vector<std::string> &classes) {
    int top5p[5];
    float top5v[5];
    TopK(static_cast<int>(output.size()), output.data(), 5, top5p, top5v);
    printf("Top-5 results\n");
    for (int i = 0; i < 5; i++) {
        std::string label = classes[top5p[i]];
        float prob = 100.0f * top5v[i];
        printf("  [%d] %s %.2f%%\n", i, label.c_str(), prob);
    }
}

// main program

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: trt_infer_plan <plan_path> <input_path>\n");
        return 1;
    }
    const char *planPath = argv[1];
    const char *inputPath = argv[2];

    printf("Start %s\n", planPath);

    std::vector<std::string> classes;
    ReadClasses("imagenet_classes.txt", classes);

    std::vector<char> plan;
    ReadPlan(planPath, plan);

    std::vector<float> input;
    ReadInput(inputPath, input);

    std::vector<float> output;

    Engine engine;
    engine.Init(plan);
    engine.DiagBindings();
    engine.Infer(input, output);

    Softmax(static_cast<int>(output.size()), output.data());
    PrintOutput(output, classes);

    return 0;
}
```

This program uses the following TensorRT API object classes:

* `nvinfer1::ILogger` - logger used by several other object classes
* `nvinfer1::IRuntime` - used to deserialize TensorRT plans to TensorRT CUDA engines
* `nvinfer1::ICudaEngine` - engine for executing inference on built networks
* `nvinfer1::IExecutionContext` - context for executing inference using CUDA engine

Class `Engine` holds smart pointers to instances of these objects.
It exposes two principal public methods: `Init` and `Infer`.

The `Init` method performs the following steps:

* creates `m_runtime` representing a runtime instance
* uses `m_runtime` to deserialize the plan into `m_engine`

The `Infer` method performs the following steps:

* creates `context` for the `m_engine`
* allocates CUDA memory buffer for the input tensor
* copies the input tensor from host to device
* allocates CUDA memory buffer for the output tensor
* specifies input/output bindings as an array holding addresses
of all input and output buffers
* runs inference for the `context` with the specified bindings and CUDA stream handle
* copies the output tensor from device to host

The program performs the following steps:

* reads the plan
* reads the pre-processed image
* reads the ImageNet categories
* creates the `engine` and initializes it using the `Init` method
* runs inference with the `engine` using the `Infer` method
* applies the softmax transformation to the output
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities in a human-readable form

NOTE: In this program we intentionally use the deprecated version of
`IRuntime::deserializeCudaEngine` method requiring the last `nullptr` argument
because, at the times of writing, using the new version without this
argument sometimes caused unexpected program behavior on the considered
GPU devices. The root cause of this problem is not yet clarified;
there might be an undocumented bug in TensorRT inference library.

The shell script `build_trt_infer_plan.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/trt_infer_plan \
    -I /usr/local/cuda/include \
    trt_infer_plan.cpp common.cpp \
    -L /usr/local/cuda/lib64 -lnvinfer -lcudart
```

Running this script is straightforward:

```
./build_trt_infer_plan.sh
```

The program has two command line arguments: a path to the TensorRT plan file and
a path to the file containing the pre-processed input image.

To run this program for the previously created ResNet50 plan and husky image, 
use the command:

```
./bin/trt_infer_plan ./plan/resnet50.plan ./data/husky01.dat
```

The program output will look like:

```
Bindings: 2
  [0] "input.1" input [1 3 224 224]
  [1] "495" output [1 1000]
Top-5 results
  [0] Siberian husky 49.53%
  [1] Eskimo dog 42.90%
  [2] malamute 5.87%
  [3] dogsled 1.22%
  [4] Saint Bernard 0.32%
```


## Step 8. Run TensorRT benchmarking using Python

The Python program `trt_bench_plan.py` implements inference benchmarking using
the previously generated TensorRT plan and a pre-processed input image.

```
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
        sys.exit("Usage: python3 trt_bench_plan.py <plan_path>")

    plan_path = sys.argv[1]

    print("Start " + plan_path)

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
    output = np.zeros((1000), dtype=np.float32)
    d_input = cuda.mem_alloc(input.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # copy input to device, run inference
    cuda.memcpy_htod(d_input, input)

    #  warm up
    for i in range(1, 10):
        context.execute_v2(bindings=bindings)

    # benchmark
    start = perf_counter()
    for i in range(1, 100):
        context.execute_v2(bindings=bindings)
    end = perf_counter()
    elapsed = ((end - start) / 100) * 1000
    print('Model {0}: elapsed time {1:.2f} ms'.format(plan_path, elapsed))
    # record for automated extraction
    print('#{0};{1:f}'.format(plan_path, elapsed))

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
```

This program uses the following TensorRT API object classes:

* `Logger` - logger used by several other object classes
* `Runtime` - used to deserialize TensorRT plans to TensorRT CUDA engines
* `ICudaEngine` - engine for executing inference on built networks
* `IExecutionContext` - context for executing inference using CUDA engine

The program performs the following steps:

* reads the plan
* generates random input
* creates `logger: Logger` representing a logger instance
* creates `runtime: Runtime` representing a runtime instance
* uses `runtime` to deserialize the plan into `engine: ICudaEngine`
* creates `context: IExecutionContext` for the `engine`
* obtains a CUDA stream reference
* allocates a Numpy array to hold the output data on the host
* allocates device memory buffers for the input and output tensors
* specifies input/output bindings as a list holding addresses
of all input and output buffers
* copies the input tensor from host to device
* measures performance by repeated execution of inference for the `context` with
the specified bindings and CUDA stream handle
* copies the output tensor from device to host
* applies the softmax transformation to the output
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities in a human-readable form

The program prints a special formatted line starting with `"#"` that
will be later used for automated extraction of performance metrics.

The program uses a path to the TensorRT plan file as its single command line argument.

To run this program for the previously created ResNet50 plan, use the command:

```
python3 trt_bench_plan.py ./plan/resnet50.plan
```

The program output will look like:

```
Model resnet50_py.plan: elapsed time 1.59 ms
Top-5 results
  610 6.29%
  549 5.21%
  446 5.00%
  783 3.20%
  892 2.93%
```

The shell script `bench_plan_all_py.sh` performs benchmarking of all supported torchvision
models:

```
#!/bin/bash

echo "#head;TensorRT (Python)"

python3 trt_bench_plan.py ./plan/alexnet.plan
python3 trt_bench_plan.py ./plan/densenet121.plan
python3 trt_bench_plan.py ./plan/densenet161.plan
python3 trt_bench_plan.py ./plan/densenet169.plan
python3 trt_bench_plan.py ./plan/densenet201.plan
python3 trt_bench_plan.py ./plan/mnasnet0_5.plan
python3 trt_bench_plan.py ./plan/mnasnet1_0.plan
python3 trt_bench_plan.py ./plan/mobilenet_v2.plan
python3 trt_bench_plan.py ./plan/mobilenet_v3_large.plan
python3 trt_bench_plan.py ./plan/mobilenet_v3_small.plan
python3 trt_bench_plan.py ./plan/resnet101.plan
python3 trt_bench_plan.py ./plan/resnet152.plan
python3 trt_bench_plan.py ./plan/resnet18.plan
python3 trt_bench_plan.py ./plan/resnet34.plan
python3 trt_bench_plan.py ./plan/resnet50.plan
python3 trt_bench_plan.py ./plan/resnext101_32x8d.plan
python3 trt_bench_plan.py ./plan/resnext50_32x4d.plan
python3 trt_bench_plan.py ./plan/shufflenet_v2_x0_5.plan
python3 trt_bench_plan.py ./plan/shufflenet_v2_x1_0.plan
python3 trt_bench_plan.py ./plan/squeezenet1_0.plan
python3 trt_bench_plan.py ./plan/squeezenet1_1.plan
python3 trt_bench_plan.py ./plan/vgg11.plan
python3 trt_bench_plan.py ./plan/vgg11_bn.plan
python3 trt_bench_plan.py ./plan/vgg13.plan
python3 trt_bench_plan.py ./plan/vgg13_bn.plan
python3 trt_bench_plan.py ./plan/vgg16.plan
python3 trt_bench_plan.py ./plan/vgg16_bn.plan
python3 trt_bench_plan.py ./plan/vgg19.plan
python3 trt_bench_plan.py ./plan/vgg19_bn.plan
python3 trt_bench_plan.py ./plan/wide_resnet101_2.plan
python3 trt_bench_plan.py ./plan/wide_resnet50_2.plan 
```

Running this script is straightforward:

```
./bench_plan_all_py.sh >bench_trt_py.log
```

The benchmarking log will be saved in `bench_trt_py.log` that later will be
used for performance comparison of various deployment methods.


## Step 9. Run TensorRT benchmarking using C++

The benchmarking of TensorRT models can be also implemented using the TensorRT C++ API.

The C++ program `trt_bench_plan.cpp` serves this purpose.

```
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>

#include <NvInfer.h>

#include "common.h"

// wrapper class for inference engine

class Engine {
public:
    Engine();
    ~Engine();
public:
    void Init(const std::vector<char> &plan);
    void StartInfer(const std::vector<float> &input);
    void RunInfer();
    void EndInfer(std::vector<float> &output);
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IRuntime> m_runtime;
    UniquePtr<nvinfer1::ICudaEngine> m_engine;
    UniquePtr<nvinfer1::IExecutionContext> m_context;
    CudaBuffer<float> m_inputBuffer;
    CudaBuffer<float> m_outputBuffer;
};

Engine::Engine(): m_active(false) { }

Engine::~Engine() { }

void Engine::Init(const std::vector<char> &plan) {
    assert(!m_active);
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (m_runtime == nullptr) {
        Error("Error creating infer runtime");
    }
    m_engine.reset(m_runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (m_engine == nullptr) {
        Error("Error deserializing CUDA engine");
    }
    m_active = true;
}

void Engine::StartInfer(const std::vector<float> &input) {
    assert(m_active);
    m_context.reset(m_engine->createExecutionContext());
    if (m_context == nullptr) {
        Error("Error creating execution context");
    }
    m_inputBuffer.Init(3 * 224 * 224);
    assert(m_inputBuffer.Size() == input.size());
    m_inputBuffer.Put(input.data());
    m_outputBuffer.Init(1000);
}

void Engine::RunInfer() {
    void *bindings[2];
    bindings[0] = m_inputBuffer.Data();
    bindings[1] = m_outputBuffer.Data();
    bool ok = m_context->executeV2(bindings);
    if (!ok) {
        Error("Error executing inference");
    }
}

void Engine::EndInfer(std::vector<float> &output) {
    output.resize(m_outputBuffer.Size());
    m_outputBuffer.Get(output.data());
}

// I/O utilities

void ReadPlan(const char *path, std::vector<char> &plan) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg();
    plan.resize(size);
    ifs.seekg(0, ifs.beg);
    ifs.read(plan.data(), size);
    ifs.close();
}

void GenerateInput(std::vector<float> &input) {
    int size = 3 * 224 * 224;
    input.resize(size);
    float *p = input.data();
    std::srand(1234);
    for (int i = 0; i < size; i++) {
        p[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

void PrintOutput(const std::vector<float> &output) {
    int top5p[5];
    float top5v[5];
    TopK(static_cast<int>(output.size()), output.data(), 5, top5p, top5v);
    printf("Top-5 results\n");
    for (int i = 0; i < 5; i++) {
        int label = top5p[i];
        float prob = 100.0f * top5v[i];
        printf("  [%d] %d %.2f%%\n", i, label, prob);
    }
}

// main program

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: trt_bench_plan <plan_path>\n");
        return 1;
    }
    const char *planPath = argv[1];

    printf("Start %s\n", planPath);

    int repeat = 100;

    std::vector<char> plan;
    ReadPlan(planPath, plan);

    std::vector<float> input;
    GenerateInput(input);

    std::vector<float> output;

    Engine engine;
    engine.Init(plan);
    engine.StartInfer(input);

    for (int i = 0; i < 10; i++) {
        engine.RunInfer();
    }

    WallClock clock;
    clock.Start();
    for (int i = 0; i < repeat; i++) {
        engine.RunInfer();
    }
    clock.Stop();
    float t = clock.Elapsed();
    printf("Model %s: elapsed time %f ms / %d = %f\n", planPath, t, repeat, t / float(repeat));
    // record for automated extraction
    printf("#%s;%f\n", planPath, t / float(repeat)); 

    engine.EndInfer(output);

    Softmax(static_cast<int>(output.size()), output.data());
    PrintOutput(output);

    return 0;
}
```

This program uses the following TensorRT API object classes:

* `nvinfer1::ILogger` - logger used by several other object classes
* `nvinfer1::IRuntime` - used to deserialize TensorRT plans to TensorRT CUDA engines
* `nvinfer1::ICudaEngine` - engine for executing inference on built networks
* `nvinfer1::IExecutionContext` - context for executing inference using CUDA engine

Class `Engine` holds smart pointers to instances of these objects.
It exposes four principal public methods: `Init`, `StartInfer`,
`RunInfer`, and `RunInfer`.

The `Init` method performs the following steps:

* creates `m_runtime` representing a runtime instance
* uses `m_runtime` to deserialize the plan into `m_engine`

The `StartInfer` method performs the following steps:

* creates `m_context` for the `m_engine`
* allocates CUDA memory buffer for the input tensor
* copies the input tensor from host to device
* allocates CUDA memory buffer for the output tensor

The `RunInfer` method performs the following steps:

* specifies input/output bindings as an array holding addresses
of all input and output buffers
* runs inference for the `m_context` with the specified bindings

The `EndInfer` method performs the following step:

* copies the output tensor from device to host

The program performs the following steps:

* reads the plan
* generates random input
* creates the `engine` 
* initializes inference on the `engine` using the `InitInfer` method
* measures performance by repeated execution of inference with the `engine`
using the `RunInfer` method
* completes inference on the `engine` using the `EndInfer` method
* applies the softmax transformation to the output
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities in a human-readable form

The program prints a special formatted line starting with `"#"` that
will be later used for automated extraction of performance metrics.

NOTE: In this program we intentionally use the deprecated version of
`IRuntime::deserializeCudaEngine` method requiring the last `nullptr` argument
because, at the times of writing, using the new version without this
argument sometimes caused unexpected program behavior on the considered
GPU devices. The root cause of this problem is not yet clarified;
there might be an undocumented bug in TensorRT inference library.

The shell script `build_trt_bench_plan.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/trt_bench_plan \
    -I /usr/local/cuda/include \
    trt_bench_plan.cpp common.cpp \
    -L /usr/local/cuda/lib64 -lnvinfer -lcudart
```

Running this script is straightforward:

```
./build_trt_bench_plan.sh
```

The program has two command line arguments: a path to the TensorRT plan file and
a path to the file containing the pre-processed input image.

To run this program for the previously created ResNet50 plan, use the command:

```
./bin/trt_bench_plan ./plan/resnet50.plan
```

The program output will look like:

```
Model resnet50.plan: elapsed time 179.491653 ms / 100 = 1.794917
Top-5 results
  [0] 610 4.25%
  [1] 549 3.90%
  [2] 783 3.64%
  [3] 892 3.51%
  [4] 446 3.18%
```

The shell script `bench_plan_all.sh` performs benchmarking of all supported torchvision
models:

```
#!/bin/bash

echo "#head;TensorRT (C++)"

./bin/trt_bench_plan ./plan/alexnet.plan
./bin/trt_bench_plan ./plan/densenet121.plan
./bin/trt_bench_plan ./plan/densenet161.plan
./bin/trt_bench_plan ./plan/densenet169.plan
./bin/trt_bench_plan ./plan/densenet201.plan
./bin/trt_bench_plan ./plan/mnasnet0_5.plan
./bin/trt_bench_plan ./plan/mnasnet1_0.plan
./bin/trt_bench_plan ./plan/mobilenet_v2.plan
./bin/trt_bench_plan ./plan/mobilenet_v3_large.plan
./bin/trt_bench_plan ./plan/mobilenet_v3_small.plan
./bin/trt_bench_plan ./plan/resnet101.plan
./bin/trt_bench_plan ./plan/resnet152.plan
./bin/trt_bench_plan ./plan/resnet18.plan
./bin/trt_bench_plan ./plan/resnet34.plan
./bin/trt_bench_plan ./plan/resnet50.plan
./bin/trt_bench_plan ./plan/resnext101_32x8d.plan
./bin/trt_bench_plan ./plan/resnext50_32x4d.plan
./bin/trt_bench_plan ./plan/shufflenet_v2_x0_5.plan
./bin/trt_bench_plan ./plan/shufflenet_v2_x1_0.plan
./bin/trt_bench_plan ./plan/squeezenet1_0.plan
./bin/trt_bench_plan ./plan/squeezenet1_1.plan
./bin/trt_bench_plan ./plan/vgg11.plan
./bin/trt_bench_plan ./plan/vgg11_bn.plan
./bin/trt_bench_plan ./plan/vgg13.plan
./bin/trt_bench_plan ./plan/vgg13_bn.plan
./bin/trt_bench_plan ./plan/vgg16.plan
./bin/trt_bench_plan ./plan/vgg16_bn.plan
./bin/trt_bench_plan ./plan/vgg19.plan
./bin/trt_bench_plan ./plan/vgg19_bn.plan
./bin/trt_bench_plan ./plan/wide_resnet101_2.plan
./bin/trt_bench_plan ./plan/wide_resnet50_2.plan
```

Running this script is straightforward:

```
./bench_plan_all.sh >bench_trt.log
```

The benchmarking log will be saved in `bench_trt.log` that later will be
used for performance comparison of various deployment methods.


## Step 10. Extract performance metrics from benchmarking logs

The Python program `merge_perf.py` introduced in Article 2 extracts 
performance metrics from multiple benchmarking log files and merges them 
in a single CSV file in a form suitable for further analysis.

The program has two or more command line arguments, each argument specifying a path
to the log file.

The program extracts special records starting with `"#"` from all input files, 
merges the extracted information, and saves it as a single CSV file. 
Each line of the output file  corresponds to one model and each column corresponds to 
one deployment method.

Assuming that benchmarking described in the Articles 1, 2, and 3 has been
performed in the sibling directories `art01`, `art02`, and `art03` respectively 
and the current directory is `art03`, the following command can be used to merge the three log
files considered so far:

```
python3 merge_perf.py ../art01/bench_torch.log ../art02/bench_ts_py.log ../art02/bench_ts.log bench_trt_py.log bench_trt.log >perf03.csv
```

The output file `perf03.csv` will look like:

```
Model;PyTorch;TorchScript (Python);TorchScript (C++);TensorRT (Python);TensorRT (C++)
alexnet;1.23;1.19;1.04;0.58;0.60
densenet121;19.79;19.45;13.34;3.73;3.67
densenet161;29.43;30.32;20.70;7.99;7.40
densenet169;28.47;30.06;20.11;8.17;7.32
densenet201;33.48;36.21;22.70;12.24;10.96
mnasnet0_5;5.45;4.65;3.67;0.64;0.61
mnasnet1_0;5.66;4.62;3.95;0.80;0.80
mobilenet_v2;6.19;5.63;4.02;0.77;0.76
mobilenet_v3_large;8.07;6.71;5.18;0.98;0.91
mobilenet_v3_small;6.37;5.66;4.19;0.74;0.67
resnet101;15.80;12.76;10.81;3.12;3.18
resnet152;23.66;19.08;16.37;4.57;4.57
resnet18;3.39;2.69;2.30;1.08;1.04
resnet34;6.11;4.83;4.11;1.84;1.79
resnet50;7.99;6.42;5.47;1.75;1.75
resnext101_32x8d;21.69;18.82;16.66;8.06;8.11
resnext50_32x4d;6.45;5.10;4.41;2.13;2.08
shufflenet_v2_x0_5;6.33;5.12;4.01;0.47;0.49
shufflenet_v2_x1_0;6.84;5.59;4.44;0.88;0.86
squeezenet1_0;3.05;2.95;2.33;0.41;0.42
squeezenet1_1;3.03;2.79;2.31;0.31;0.31
vgg11;1.91;1.81;1.84;1.74;1.75
vgg11_bn;2.37;2.09;1.96;1.75;1.75
vgg13;2.26;2.21;2.27;2.16;2.15
vgg13_bn;2.62;2.42;2.43;2.14;2.17
vgg16;2.82;2.79;2.88;2.64;2.61
vgg16_bn;3.23;3.03;3.06;2.61;2.65
vgg19;3.29;3.36;3.40;3.17;3.14
vgg19_bn;3.72;3.60;3.64;3.07;3.13
wide_resnet101_2;15.50;12.32;10.55;5.58;5.45
wide_resnet50_2;7.88;6.66;5.35;2.83;2.95 
```

## Conclusion

Analysis of these performance data reveals that using TensorRT
provides substantial performance increase compared to all previously considered
deployment methods.

Differences between TensorRT performance data for Python and C++ are within the experimental error. 
Python and C++ can be considered equally good for running TensorRT.


## Further reading

This documentation on NVIDIA TensorRT 8.0.3 can be used for further references:

* [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/developer-guide/index.html)
* [NVIDIA TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/sample-support-guide/index.html)

At the time of writing, the detailed API information was available only for version 8.0.1:

* [NVIDIA TensorRT C++ API](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-801/api/c_api/index.html)
* [NVIDIA TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-801/api/python_api/index.html)

The index of documents covering all TensorRT versions is available
[here](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html).


