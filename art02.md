
# Article 2. Deployment techniques for PyTorch models using TorchScript

This article covers using TorchScript for deployment of PyTorch models.

TorchScript represents a way to create serializable and optimizable models from PyTorch code.
Technically, TorchScript is a statically typed subset of Python. TorchScript code
is executed using a special interpreter. Static typing allows for more performance efficient 
execution of TorchScript models compared to their original PyTorch versions.
In a typical scenario, models are trained in PyTorch using conventional tools in Python and 
then exported via TorchScript for deployment to a production environment. TorchScript models
can be executed by Python or C++ programs not requiring presence of the PyTorch environment.

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

To run this script, enter the following commands:

```
mkdir -p ts
python3 generate_ts_resnet50.py
```

The Python program `generate_ts_all.py` can be used to produce TorchScript code
for most of the image classification models available in torchscript.

```
import torch
import torchvision.models as models

MODELS = [
    ('alexnet', models.alexnet),

    ('densenet121', models.densenet121),
    ('densenet161', models.densenet161),
    ('densenet169', models.densenet169),
    ('densenet201', models.densenet201),

#    ('googlenet', models.googlenet),

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

We will later use the generated TorchScript models for benchmarking.
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

# Read the categories
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
resnet50_scripted = torch.jit.script(resnet50)
dummy_input = torch.rand(1, 3, 224, 224).cuda()

resnet50.eval()
resnet50_scripted.eval()

# benchmark original model

for i in range(1, 10):
    resnet50(dummy_input)
start = perf_counter()
for i in range(1, 100):
    resnet50(dummy_input)
end = perf_counter()
print('Perf original model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# benchmark TorchScript model

for i in range(1, 10):
    resnet50_scripted(dummy_input)
start = perf_counter()
for i in range(1, 100):
    resnet50_scripted(dummy_input)
end = perf_counter()
print('Perf TorchScript model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# compare Top-5 results

unscripted_output = resnet50(dummy_input)
scripted_output = resnet50_scripted(dummy_input)

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

print('Original model top 5 results:\n {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n {}'.format(scripted_top5))
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

Typically, on NVIDIA RTX 3080 the TorchScript code for ResNet50 runs inference
about 1.5 times faster compared to the original PyTorch model.


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
        std::cerr << "Usage: infer_model_ts <path-to-exported-model> <path-to-input-data>" << std::endl;
        return -1;
    }

    // make sure CUDA us available; get CUDA device
    bool have_cuda = torch::cuda::is_available();
    assert(have_cuda);
    torch::Device device = torch::kCUDA;

    std::cout << "Loading model..." << std::endl;

    // deserialize ScriptModule
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

    // ensure that autograd is off
    torch::NoGradGuard noGrad; 
    // turn off dropout and other training-time layers/functions
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

    // execute model and package output as tensor
    at::Tensor output = module.forward(inputs).toTensor();

    // apply softmax and get Top-5 results
    namespace F = torch::nn::functional;
    at::Tensor softmax = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5 = softmax.topk(5);
    
    // get probabilities ans labels
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

class WallClock {
public:
    WallClock();
    ~WallClock();
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

WallClock::WallClock(): elapsed(0.0f) { }

WallClock::~WallClock() { }

// interface

void WallClock::Reset() {
    elapsed = 0.0f;
}

void WallClock::Start() {
    start = std::chrono::steady_clock::now();
}

void WallClock::Stop() {
    end = std::chrono::steady_clock::now();
    elapsed +=
        std::chrono::duration_cast<
            std::chrono::duration<float, std::milli>>(end - start).count();
}

float WallClock::Elapsed() {
    return elapsed;
}

//
//    Main program
//

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: bench_ts <path-to-exported-model>" << std::endl;
        return -1;
    }

    std::string name(argv[1]);

    if (name.find("googlenet") != std::string::npos) {
        std::cout << "Skip inference: " << name << std::endl;
        std::cout << "DONE" << std::endl << std::endl;
        return 0;
    }

    // execute model and package output as tensor
    std::cout << "Start model " << name << std::endl;

    int repeat = 100; // make it configuravle?

    bool have_cuda = torch::cuda::is_available();
    assert(have_cuda);

    torch::Device device = torch::kCUDA;

    std::cout << "Loading model..." << std::endl;

    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1], device);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading model" << std::endl;
        std::cerr << e.what_without_backtrace() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully" << std::endl;

    // ensures that autograd is off
    torch::NoGradGuard noGrad; 
    // turn off dropout and other training-time layers/functions
    module.eval(); 

    // create input
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}, device));

    // warm up
    for (int i = 0; i < 10; i++) {
        module.forward(inputs);
    }

    // benchmark
    WallClock clock;
    clock.Start();
    for (int i = 0; i < repeat; i++) {
        module.forward(inputs);
    }
    clock.Stop();
    float t = clock.Elapsed();
    std::cout << "Model " << name << ": elapsed time " << 
        t << " ms / " << repeat << " iterations = " << t / float(repeat) << std::endl; 

    // execute model and package output as tensor
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

Typically, inference with ResNet50 TorchScript code using the C++ program runs faster
compared to the equivalent Python program.

The shell script `bench_ts_all.sh` can be used to benchmark the entire collection
of image classification TorchScript models.

