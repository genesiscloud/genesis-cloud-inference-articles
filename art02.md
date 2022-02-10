
# Article 2. Deployment techniques for PyTorch models using TorchScript

This article covers using TorchScript for deployment of PyTorch models.

TorchScript represents a way to create serializable and optimizable models from PyTorch code.
Technically, TorchScript is a statically typed subset of Python. TorchScript code
is executed using a special interpreter. Static typing allows for more performance efficient 
execution of TorchScript models compared to their original PyTorch versions.
In a typical scenario, models are trained in PyTorch using conventional tools in Python and 
then exported via TorchScript for deployment to a production environment. TorchScript models
can be executed by Python or C++ programs not requiring the presence of the PyTorch environment.

Pytorch provides two methods for generating TorchScript from the model code known as
**tracing** and **scripting**. When tracing is used, the model is provided with the sample
input, the regular inference is performed, and all the operations executed are
traced and recorded as TorchScript. In case of scripting, the TorchScript code is
generated from the static inspection of the model.

The [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
tutorial can be referenced for more detailed discussion of the respective techniques.

We will use the scripting method in the examples of this article.

We assume that you will continue using the Genesis Cloud GPU-enabled instance that
you created and configured while studying the Article 1.

Various assets (source code, shell scripts, and data files) used in this article
can be found in the supporting
[GitHub repository](https://github.com/lxgo/genesis-kbase/tree/main/art02).

To run examples described in this article we recommend cloning the entire 
[repository](https://github.com/lxgo/genesis-kbase) on your Genesis Cloud instance.
The subdirectory `art02` must be made your current directory.


## Step 1. Generation of TorchScript code for classification models

We will continue using the torchvision image classification models for our examples.
As the first step, we will demonstrate generation of TorchScript code for
the already familiar ResNet50 model.

The Python program `generate_ts_resnet50.py` serves this purpose.

```
import torch
import torchvision.models as models

def generate_model(name, model):
    print('Generate', name)
    m = model(pretrained=True).cuda()
    m_scripted = torch.jit.script(m)
    m_scripted.save('./ts/' + name + '.ts')

generate_model('resnet50', models.resnet50)
```

This program:

* creates a pretrained ResNet50 model
* places the model on CUDA device using the `cuda` method
* generates TorchScript code using scripting via `torch.jit.script` function
* saves the generated code in a file using the `save` method

We want to execute the generated TorchScript code on a GPU, therefore
the model must be placed on a CUDA device before scripting.

For convenience, we will place all TorchScript files in a separate subdirectory `ts`.

To run this program, enter the following commands:

```
mkdir -p ts
python3 generate_ts_resnet50.py
```

The Python program `generate_ts_all.py` can be used to produce TorchScript code
for most of the image classification models available in torchvision.

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

def generate_model(name, model):
    print('Generate', name)
    m = model(pretrained=True).cuda()
    m_scripted = torch.jit.script(m)
    m_scripted.save('./ts/' + name + '.ts')

for name, model in MODELS:
    generate_model(name, model)
```

To run this program, enter the following commands:

```
mkdir -p ts
python3 generate_ts_all.py
```

We will use the generated TorchScript models for benchmarking.
Code generation for `googlenet` is currently disabled because this
model uses calling conventions different from the other torchvision models.


## Step 2. Running TorchScript code from a Python program

The Python program `infer_resnet50_ts.py` can be used to run inference
for the ResNet50 TorchScript code with the single image as input.

```
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMG_PATH = "./data/husky01.jpg"

# load the TorchScript model
resnet50 = torch.jit.load("./ts/resnet50.ts")
resnet50.eval()

# specify image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# import and transform image
img = Image.open(IMG_PATH)
img = transform(img)

# create a batch, run inference
input = torch.unsqueeze(img, 0)

# move the input to GPU
assert torch.cuda.is_available()
input = input.to("cuda")

with torch.no_grad():
    output = resnet50(input)

# apply softmax and get Top-5 results
output = F.softmax(output, dim=1)
top5 = torch.topk(output[0], 5)

# read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# print results
for ind, val in zip(top5.indices, top5.values):
    print("{0} {1:.2f}%".format(categories[ind], val * 100)) 
```

Functionality is similar to the program `infer_resnet50.py`, however
here the TorchScript code is executed instead of the original PyTorch
model.

The program performs these main actions:

* loads the TorchScript code for the ResNet50 model
* sets the model in evaluation (inference) mode
* specifies transformations for image pre-processing
* reads the image file using PIL package
* applies transformations to the image
* creates an input batch tensor containing one transformed image
* presuming that CUDA is available, moves the input batch to CUDA device
* disables gradient computations in PyTorch
* runs inference for the model producing the output tensor
* applies the softmax transformation to the output
* gets labels and probabilities for top 5 results
* reads ImageNet class descriptions
* prints top 5 classes and probabilities in human-readable form

To run this program, use the command:

```
python3 infer_resnet50_ts.py
```

The program output will look like:

```
Siberian husky 49.52%
Eskimo dog 42.90%
malamute 5.87%
dogsled 1.22%
Saint Bernard 0.32%
```

You can also experiment with the other classification models from the torchvision
library and other input images.


## Step 3. Benchmarking the TorchScript model

To compare performance of TorchScript code to the original PyTorch model,
we use the Python program `perf_resnet50_ts.py`:

```
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

# create models

resnet50 = models.resnet50(pretrained=True).cuda()
resnet50_ts = torch.jit.script(resnet50)
input = torch.rand(1, 3, 224, 224).cuda()

resnet50.eval()
resnet50_ts.eval()

# benchmark original model

with torch.no_grad():
    for i in range(1, 10):
        resnet50(input)
    start = perf_counter()
    for i in range(1, 100):
        resnet50(input)
    end = perf_counter()

print('Perf original model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# benchmark TorchScript model

with torch.no_grad():
    for i in range(1, 10):
        resnet50_ts(input)
    start = perf_counter()
    for i in range(1, 100):
        resnet50_ts(input)
    end = perf_counter()

print('Perf TorchScript model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# compare Top-5 results

output = resnet50(input)
output_ts = resnet50_ts(input)

top5 = F.softmax(output, dim=1).topk(5).indices
top5_ts = F.softmax(output_ts, dim=1).topk(5).indices

print('Original model top 5 results:\n {}'.format(top5))
print('TorchScript model top 5 results:\n {}'.format(top5_ts))
```

This program:

* creates a pre-trained ResNet50 model and places it on CUDA device
* uses scripting to produce the TorchScript code for this model
* generates a dummy input tensor with required shape and random contents
* sets both models in evaluation (inference) mode
* benchmarks the original model
* benchmarks the TorchScript model
* applies the softmax transformation to the outputs
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities

The benchmarking of each model includes 10 "warmup" inference runs
followed by 100 runs for which the total wall clock time is measured.
The measured time is divided by the number of runs and the average
time for one run in milliseconds is displayed.

To run this program, use the command:

```
python3 perf_resnet50_ts.py
```

The program output will look like:

```
Perf original model 9.11 ms
Perf TorchScript model 6.24 ms
Original model top 5 results:
 tensor([[549, 783, 446, 490, 610]], device='cuda:0')
TorchScript model top 5 results:
 tensor([[549, 783, 446, 490, 610]], device='cuda:0')
```

The Python program `bench_model_ts.py` is more general; it implements benchmarking
of any supported torchvision image classification model:

```
import sys
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 bench_model_ts.py <model_name>")

    name = sys.argv[1]
    print('Start ' + name)

    # create model

    builder = getattr(models, name)
    model_orig = builder(pretrained=True).cuda()
    model = torch.jit.script(model_orig)
    model.eval()

    input = torch.rand(1, 3, 224, 224).cuda()

    # benchmark TorchScript model

    with torch.no_grad():
        for i in range(1, 10):
            model(input)
        start = perf_counter()
        for i in range(1, 100):
            model(input)
        end = perf_counter()

    elapsed = ((end - start) / 100) * 1000
    print('Model {0}: elapsed time {1:.2f} ms'.format(name, elapsed))
    # record for automated extraction
    print('#{0};{1:f}'.format(name, elapsed))

    # print Top-5 results

    output = model(input)
    top5 = F.softmax(output, dim=1).topk(5)
    top5p = top5.indices.detach().cpu().numpy()
    top5v = top5.values.detach().cpu().numpy()

    print("Top-5 results")
    for ind, val in zip(top5p[0], top5v[0]):
        print("  {0} {1:.2f}%".format(ind, val * 100))

main()
```

The program uses a model name as its single command line argument.

The program performs the following steps:

* creates a model builder for the specified model name
* uses this builder to create a model; places the model on CUDA device
* uses scripting to produce the TorchScript version for this model
* sets the model in evaluation (inference) mode
* creates an input tensor with random dummy contents; places it on CUDA device
* benchmarks the model
* prints benchmarking results
* applies the softmax transformation to the outputs
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities

The program prints a special formatted line starting with `"#"` that
will be later used for automated extraction of performance metrics.

To run this program for ResNet50, use the command:

```
python3 bench_model_ts.py resnet50
```

The program output will look like:

```
Start resnet50
Model resnet50: elapsed time 6.48 ms
#resnet50;6.484319
Top-5 results
  549 4.64%
  892 3.64%
  783 3.17%
  610 3.15%
  446 2.88%
```

The shell script `bench_ts_all_py.sh` performs benchmarking of all supported torchvision
models:

```
#!/bin/bash

echo "#head;TorchScript (Python)"

python3 bench_model_ts.py alexnet
python3 bench_model_ts.py densenet121
python3 bench_model_ts.py densenet161
python3 bench_model_ts.py densenet169
python3 bench_model_ts.py densenet201
python3 bench_model_ts.py mnasnet0_5
python3 bench_model_ts.py mnasnet1_0
python3 bench_model_ts.py mobilenet_v2
python3 bench_model_ts.py mobilenet_v3_large
python3 bench_model_ts.py mobilenet_v3_small
python3 bench_model_ts.py resnet101
python3 bench_model_ts.py resnet152
python3 bench_model_ts.py resnet18
python3 bench_model_ts.py resnet34
python3 bench_model_ts.py resnet50
python3 bench_model_ts.py resnext101_32x8d
python3 bench_model_ts.py resnext50_32x4d
python3 bench_model_ts.py shufflenet_v2_x0_5
python3 bench_model_ts.py shufflenet_v2_x1_0
python3 bench_model_ts.py squeezenet1_0
python3 bench_model_ts.py squeezenet1_1
python3 bench_model_ts.py vgg11
python3 bench_model_ts.py vgg11_bn
python3 bench_model_ts.py vgg13
python3 bench_model_ts.py vgg13_bn
python3 bench_model_ts.py vgg16
python3 bench_model_ts.py vgg16_bn
python3 bench_model_ts.py vgg19
python3 bench_model_ts.py vgg19_bn
python3 bench_model_ts.py wide_resnet101_2
python3 bench_model_ts.py wide_resnet50_2
```

Running this script is straightforward:

```
./bench_ts_all_py.sh >bench_ts_py.log
```

The benchmarking log will be saved in `bench_ts_py.log` that later will be
used for performance comparison of various deployment methods.


## Step 4. Install LibTorch

LibTorch is a runtime library designed for execution of TorchScript code
without using Python. This library is required for running the TorchScript
interpreter from the C++ programs.

LibTorch requires separate installation. For this purpose,
visit the PyTorch [product site](https://pytorch.org/) and select the desired 
configuration as follows:

* PyTorch Build: Stable (1.10.1)
* Your OS: Linux
* Package: LibTorch
* Language: C++ / Java
* Compute Platform: CUDA 11.3

The URL references to the distribution files for the selected configuration will be presented.
Select the reference labeled "Download here (cxx11 ABI)"; at the time of writing of this
article it was:

```
https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.1%2Bcu113.zip
```

Download this distribution and unpack its content into a separate directory (we will use `~/vendor`
in this article):

```
cd ~/transit
wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.1%2Bcu113.zip
mkdir -p ~/vendor
unzip libtorch-cxx11-abi-shared-with-deps-1.10.1+cu113.zip -d ~/vendor
```

As before, we recommend using a scratch directory '~/transit' as your current directory during the installation.
The package contents will be placed in `~/vendor/libtorch`.


## Step 5. Preparing the pre-processed input for C++ inference program

To simplify our C++ inference examples and ensure comparable results,
we will pre-process the input image using a stand-alone program that 
reads the image file, performs all required transformations and saves 
the result tensor in a plain binary file. The C++ programs will read 
this file and pass its contents directly to the inference engine.

The simple Python program `read_image.py` implements this
stand-alone pre-processing:

```
from torchvision import models, transforms
from PIL import Image

IMG_PATH = "./data/husky01.jpg"
DATA_PATH = "./data/husky01.dat"

# specify image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# import and transform image
img = Image.open(IMG_PATH)
img = transform(img)

# convert to numpy array and write to file
data = img.numpy()
data.tofile(DATA_PATH) 
```

To run it, enter the command:

```
python3 read_image.py
```


## Step 6. Running TorchScript inference using C++

The C++ program `infer_model_ts.cpp` runs inference using a TorchScript model
and pre-processed input image.

```
#include <cassert>
#include <iostream>
#include <fstream>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: infer_model_ts <torchscript-model-path> <input-data-path>" << std::endl;
        return -1;
    }

    // make sure CUDA is available; get CUDA device
    bool haveCuda = torch::cuda::is_available();
    assert(haveCuda);
    torch::Device device = torch::kCUDA;

    std::cout << "Loading model..." << std::endl;

    // load model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1], device);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading model" << std::endl;
        std::cerr << e.what_without_backtrace() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully" << std::endl;
    std::cout << std::endl;

    // switch off autigrad, set evalation mode
    torch::NoGradGuard noGrad; 
    module.eval(); 

    // read classes
    std::string line;
    std::ifstream ifsClasses("imagenet_classes.txt", std::ios::in);
    if (!ifsClasses.is_open()) {
        std::cerr << "Cannot open imagenet_classes.txt" << std::endl;
        return -1;
    }
    std::vector<std::string> classes;
    while (std::getline(ifsClasses, line)) {
        classes.push_back(line);
    }
    ifsClasses.close();

    // read input
    std::ifstream ifsData(argv[2], std::ios::in | std::ios::binary);
    if (!ifsData.is_open()) {
        std::cerr << "Cannot open " << argv[2] << std::endl;
        return -1;
    }
    size_t size = 3 * 224 * 224 * sizeof(float);
    std::vector<char> data(size);
    ifsData.read(data.data(), data.size());
    ifsData.close();

    // create input tensor on CUDA device 
    at::Tensor input = torch::from_blob(data.data(), {1, 3, 224, 224}, torch::kFloat);
    input = input.to(device);

    // create inputs
    std::vector<torch::jit::IValue> inputs{input};

    // execute model
    at::Tensor output = module.forward(inputs).toTensor();

    // apply softmax and get Top-5 results
    namespace F = torch::nn::functional;
    at::Tensor softmax = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5 = softmax.topk(5);
    
    // get probabilities and labels
    at::Tensor probs = std::get<0>(top5);
    at::Tensor labels = std::get<1>(top5);

    // print probabilities and labels
    for (int i = 0; i < 5; i++) {
        float prob = 100.0f * probs[0][i].item<float>();
        long label = labels[0][i].item<long>();
        std::cout << std::fixed << std::setprecision(2) << prob << "% " << classes[label] << std::endl; 
    }
    std::cout << std::endl;

    std::cout << "DONE" << std::endl;
    return 0;
}
```

The program is functionally similar to previously described Python program `infer_resnet50_ts.py`.

The shell script `build_infer_model_ts.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/infer_model_ts \
    -I ~/vendor/libtorch/include \
    -I ~/vendor/libtorch/include/torch/csrc/api/include \
    infer_model_ts.cpp \
    -L ~/vendor/libtorch/lib \
    -lc10_cuda -lc10 \
    -Wl,--no-as-needed -ltorch_cuda -Wl,--as-needed \
    -ltorch_cpu -ltorch
```

Running this script is straightforward:

```
./build_infer_model_ts.sh
```

The command line invoking the `g++` compiler refers to several LibTorch shared libraries
located in `~/vendor/libtorch/lib`. To make these libraries accessible,
the environment variable `LD_LIBRARY_PATH` must be augmented before running
the program as follows:

```
export LD_LIBRARY_PATH=~/vendor/libtorch/lib:$LD_LIBRARY_PATH
```

(NOTE: This setting will most likely prevent further normal functioning of Python programs
using regular PyTorch because of the conflict of LibTorch libraries with their
equivalents from the regular PyTorch installation. Roll back the above change
of `LD_LIBRARY_PATH` if you want to run PyTorch applications implemented in Python 
during the same session on the same instance.)

The program has two command line arguments: a path to the TorchScript file and a path to
the pre-processed input binary file. For example, to run inference with the ResNet50
TorchScript code and the data file created at the previous step, use the following
command:

```
./bin/infer_model_ts ./ts/resnet50.ts ./data/husky01.dat
```

The program output will look like:

```
Loading model...
Model loaded successfully

49.52% Siberian husky
42.90% Eskimo dog
5.87% malamute
1.22% dogsled
0.32% Saint Bernard

DONE
```

## Step 7. Benchmarking TorchScript inference in C++

The C++ program `bench_ts.cpp` performs inference benchmarking for a TorchScript model:

```
#include <cstdio>
#include <cassert>
#include <string>
#include <iostream>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

//
//    WallClock
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
//    Main program
//

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: bench_ts <torchscript-model-path>" << std::endl;
        return -1;
    }

    std::string name(argv[1]);

    std::cout << "Start model " << name << std::endl;

    int repeat = 100; 

    bool haveCuda = torch::cuda::is_available();
    assert(haveCuda);

    torch::Device device = torch::kCUDA;

    std::cout << "Loading model..." << std::endl;

    // load model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1], device);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading model" << std::endl;
        std::cerr << e.what_without_backtrace() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully" << std::endl;

    // switch off autograd, set evluation mode
    torch::NoGradGuard noGrad; 
    module.eval(); 

    // create input
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}, device));

    // warm up
    for (int i = 0; i < 10; i++) {
        module.forward(inputs);
    }

    // benchmark
    Timer timer;
    timer.Start();
    for (int i = 0; i < repeat; i++) {
        module.forward(inputs);
    }
    timer.Stop();
    float t = timer.Elapsed();
    std::cout << "Model " << name << ": elapsed time " << 
        t << " ms / " << repeat << " iterations = " << t / float(repeat) << std::endl; 
    // record for automated extraction
    std::cout << "#" << name << ";" << t / float(repeat) << std::endl;

    // execute model
    at::Tensor output = module.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    at::Tensor softmax = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5 = softmax.topk(5);
    at::Tensor labels = std::get<1>(top5);

    std::cout << labels[0] << std::endl;

    std::cout << "DONE" << std::endl << std::endl;
    return 0;
}
```

The program is functionally similar to previously described Python program `perf_resnet50_ts.py`.

The shell script `build_bench_ts.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/bench_ts \
    -I ~/vendor/libtorch/include \
    -I ~/vendor/libtorch/include/torch/csrc/api/include \
    bench_ts.cpp \
    -L ~/vendor/libtorch/lib \
    -lc10_cuda -lc10 \
    -Wl,--no-as-needed -ltorch_cuda -Wl,--as-needed \
    -ltorch_cpu -ltorch
```

Running this script is straightforward:

```
./build_bench_ts.sh
```

The program has one command line argument representing a path to the TorchScript file. 
For example, to run inference with the ResNet50 TorchScript code, use the following
command:

```
./bin/bench_ts ./ts/resnet50.ts
```

The program output will look like:

```
Start model ./ts/resnet50.ts
Loading model...
Model loaded successfully
Model ./ts/resnet50.ts: elapsed time 547.467 ms / 100 iterations = 5.47467
 490
 549
 446
 610
 556
[ CUDALongType{5} ]
DONE
```

The shell script `bench_ts_all.sh` can be used to benchmark the entire collection
of image classification TorchScript models.

Running this script is straightforward:

```
./bench_ts_all.sh >bench_ts.log
```

The benchmarking log will be saved in `bench_ts.log` that later will be
used for performance comparison of various deployment methods.


## Step 8. Extract performance metrics from benchmarking logs

The Python program `merge_perf.py` extracts performance metrics from multiple
benchmarking log files and merges them in a single CSV file in a form
suitable for further analysis:

```
import sys

def get_model_name(s):
    pos = s.rfind("/")
    if pos >= 0:
        s = s[pos+1:]
    pos = s.find(".")
    if pos >= 0:
        s = s[:pos]
    return s

def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python3 merge_perf.py <path1> <path2> ...") 

    heads = []
    model_set = set()
    perf_all = {}
    for path in sys.argv[1:]:
        with open(path, "r") as fp:
            head = None
            perf = {}
            lines = fp.readlines()
            for line in lines:
                if not line.startswith("#"):
                    continue
                line = line[1:].strip()
                fields = line.split(";")
                if fields[0] == "head":
                    head = fields[1]
                else:
                    model = get_model_name(fields[0])
                    model_set.add(model)
                    perf[model] = float(fields[1])
            if head is None:
                raise ValueError("Missing head tag in " + path)
            heads.append(head)
            for key, value in perf.items():
                perf_all[head + "#" + key] = value

    line = "Model"
    for head in heads:
        line += ";" + head
    print(line)

    models = sorted(list(model_set))
    for model in models:
        line = model
        for head in heads:
            key = head + "#" + model
            value = "-"
            if key in perf_all:
                value = "{0:.2f}".format(perf_all[key])
            line += ";" + value
        print(line)

main()
```

The program has two or more command line arguments, each argument specifying a path
to the log file.

The program extracts special records starting with `"#"` from all input files, 
merges the extracted information, and saves it as a single CSV file. 
Each line of the output file  corresponds to one model and each column corresponds to 
one deployment method.

For example, assuming that benchmarking described in the Articles 1 and 2 has been
performed in the sibling directories `art01` and `art02` respectively and the current
directory is `art02`, the following command can be used to merge the three log
files considered so far:

```
python3 merge_perf.py ../art01/bench_torch.log bench_ts_py.log bench_ts.log >perf02.csv
```

The output file `perf02.csv` will look like:

```
Model;PyTorch;TorchScript (Python);TorchScript (C++)
alexnet;1.23;1.19;1.04
densenet121;19.79;19.45;13.34
densenet161;29.43;30.32;20.70
densenet169;28.47;30.06;20.11
densenet201;33.48;36.21;22.70
mnasnet0_5;5.45;4.65;3.67
mnasnet1_0;5.66;4.62;3.95
mobilenet_v2;6.19;5.63;4.02
mobilenet_v3_large;8.07;6.71;5.18
mobilenet_v3_small;6.37;5.66;4.19
resnet101;15.80;12.76;10.81
resnet152;23.66;19.08;16.37
resnet18;3.39;2.69;2.30
resnet34;6.11;4.83;4.11
resnet50;7.99;6.42;5.47
resnext101_32x8d;21.69;18.82;16.66
resnext50_32x4d;6.45;5.10;4.41
shufflenet_v2_x0_5;6.33;5.12;4.01
shufflenet_v2_x1_0;6.84;5.59;4.44
squeezenet1_0;3.05;2.95;2.33
squeezenet1_1;3.03;2.79;2.31
vgg11;1.91;1.81;1.84
vgg11_bn;2.37;2.09;1.96
vgg13;2.26;2.21;2.27
vgg13_bn;2.62;2.42;2.43
vgg16;2.82;2.79;2.88
vgg16_bn;3.23;3.03;3.06
vgg19;3.29;3.36;3.40
vgg19_bn;3.72;3.60;3.64
wide_resnet101_2;15.50;12.32;10.55
wide_resnet50_2;7.88;6.66;5.35 
```

The Python program `tab_perf.py` can be used to display the CSV data in the tabular format:

```
import sys

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 tab_perf.py <input_csv_path>")  

    input_path = sys.argv[1]

    min_col_width = 12
    margin = 4

    lines = []
    with open(input_path, "r") as fp:
        for line in fp:
            line = line.strip()
            lines.append(line)

    num_cols = len(lines[0].split(";"))
    col_widths = [min_col_width] * num_cols
    val_widths = [0] * num_cols

    for lno, line in enumerate(lines):
        fields = line.split(";")
        assert len(fields) == num_cols
        for col in range(num_cols):
            width = len(fields[col])
            if width > col_widths[col]:
                col_widths[col] = width
            if lno != 0 and width > val_widths[col]:
                val_widths[col] = width

    for lno, line in enumerate(lines):
        output = ""
        fields = line.split(";")
        for col in range(num_cols):
            field = fields[col]
            space = col_widths[col] - len(field)
            if col == 0:
                output += field          
                output += " " * space
            else:
                if lno == 0:
                    rpad = space // 2
                else:
                    rpad = (col_widths[col] - val_widths[col]) // 2
                lpad = space - rpad
                output += " " * (margin + lpad)
                output += field          
                output += " " * rpad
        print(output)
        if lno == 0:
            tab_width = sum(col_widths) + margin * (num_cols - 1)
            output = "-" * tab_width
            print(output)

main()
```

To run this program, use the following command line:

```
python3 tab_perf.py perf02.csv >perf02.txt
```

The output file `perf02.txt` will look like:

```
Model                    PyTorch      TorchScript (Python)    TorchScript (C++)
-------------------------------------------------------------------------------
alexnet                    1.23                1.19                  1.04      
densenet121               19.79               19.45                 13.34      
densenet161               29.43               30.32                 20.70      
densenet169               28.47               30.06                 20.11      
densenet201               33.48               36.21                 22.70      
mnasnet0_5                 5.45                4.65                  3.67      
mnasnet1_0                 5.66                4.62                  3.95      
mobilenet_v2               6.19                5.63                  4.02      
mobilenet_v3_large         8.07                6.71                  5.18      
mobilenet_v3_small         6.37                5.66                  4.19      
resnet101                 15.80               12.76                 10.81      
resnet152                 23.66               19.08                 16.37      
resnet18                   3.39                2.69                  2.30      
resnet34                   6.11                4.83                  4.11      
resnet50                   7.99                6.42                  5.47      
resnext101_32x8d          21.69               18.82                 16.66      
resnext50_32x4d            6.45                5.10                  4.41      
shufflenet_v2_x0_5         6.33                5.12                  4.01      
shufflenet_v2_x1_0         6.84                5.59                  4.44      
squeezenet1_0              3.05                2.95                  2.33      
squeezenet1_1              3.03                2.79                  2.31      
vgg11                      1.91                1.81                  1.84      
vgg11_bn                   2.37                2.09                  1.96      
vgg13                      2.26                2.21                  2.27      
vgg13_bn                   2.62                2.42                  2.43      
vgg16                      2.82                2.79                  2.88      
vgg16_bn                   3.23                3.03                  3.06      
vgg19                      3.29                3.36                  3.40      
vgg19_bn                   3.72                3.60                  3.64      
wide_resnet101_2          15.50               12.32                 10.55      
wide_resnet50_2            7.88                6.66                  5.35       
```


## Conclusion

Analysis of these performance data reveals that running TorchScript code with C++
and TorchLib provides substantial performance increase (typically about by the factor of 1.5)
compared to running the original PyTorch model with Python. 

Running TorchScript code with Python is slower than running it with C++
but faster than running the original PyTorch model.

