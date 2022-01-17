
# Article 1. Installation and basic use of CUDA, PyTorch, and torchvision

This article will guide you through the basic steps required for installation
and basic use of PyTorch and related software components on a Genesis Cloud
GPU instance. The following topics are covered:

* creation of a Genesis Cloud GPU instance
* installation of CUDA and cuDNN
* installation of PyTorch and torchvision
* inference using torchvision image classification models with Python

We will use a Genesis Cloud instance equipped with NVIDIA RTX 3080 GPU
and the following software versions:

* OS: Ubuntu 20.04
* CUDA 11.3.1
* cuDNN 8.2.1
* PyTorch 1.10.1
* torchvision 0.11.2

Various assets (source code, shell scripts, and data files) used in this article
can be found in the supporting
[GitHub repository](https://github.com/lxgo/genesis-kbase/tree/main/art01).


## Step 1. Creating a GPU instance on Genesis Cloud

We assume that you have an account at Genesis Cloud. We start with creation of
a new GPU instance that will be used for running examples described in this
and several following articles.

To create a new instance, visit a [page](https://compute.genesiscloud.com/dashboard/instances/create) 
titled "Create New Instance". On this page:

* choose a meaningful Hostname and, optionally, a Nickname
* in Select Instance Type: choose GPU NVIDIA / GeForce RTX 3080
* in Configuration: keep default values
* **do not** select "Install NVIDIA GPU driver 460.80" (or any other driver version, if mentioned)
* in Select Image: choose Ubuntu 20.04
* Authentication: select method (SSH Key is recommended)
 
Once ready, click the "Create Instance" button.

Once your instance is ready, login and proceed with the following steps.

## Step 2. Install CUDA

As the next step, we will install CUDA.

To install the desired version of CUDA, visit the 
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) page.
Select the line for CUDA Toolkit 11.3.1. You will be redirected to the
[corresponding page](https://developer.nvidia.com/cuda-11-3-1-download-archive).
On this page, make the following selections:

* Operating System: Linux
* Architecture: x86_64
* Distribution: Ubuntu
* Version: 20.04
* Installer Type: deb (local)

The sequence of commands for installation of the selected version will be presented.
At the time of writing of this article, these commands were:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

As these commands might change in the future, we recommend using the commands that are actually presented
on this page. 

Enter these commands one by one (or build and execute the respective shell script).
The last command will launch the CUDA installation process that might take a while.

For this and similar installation steps, we advice to create a scratch directory
(for example, `~/transit`) and set it as current directory during the installation:

```
mkdir -p ~/transit
cd ~/transit
```

Upon the successful installation, we recommend rebooting your instance by stopping and starting it
from the Genesis Cloud Web console.

We strongly advise you to take time and study [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
available by the reference on this page. 

To validate CUDA installation, type the command:

```
nvidia-smi
```

You should get the output looking like:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:00:05.0 Off |                  N/A |
|  0%   21C    P8     6W / 320W |      5MiB / 10018MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       874      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

To use the NVIDIA CUDA compiler driver `nvcc` (which will be needed for more advanced examples),
update the `PATH` environment variable:

```
export PATH=/usr/local/cuda/bin:$PATH
```

Then, to check the `nvcc` availability, type:

```
nvcc --version
```

## Step 3. Install cuDNN

The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library 
of primitives for deep neural networks. To install cuDNN, visit the
[distribution page](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/).
Select packages corresponding to the desired combination of CUDA and cuDNN versions.
For each such combination there are two packages of interest representing the runtime and developer libraries.
At the time of writing of this article, for CUDA 11.3 and cuDNN 8.2.1, these packages were:

```
libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
```

Download these files by entering the respective `wget` commands, for example:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
```

As before, we recommend to perform installation from a separate scratch directory, e.g., `~/transit`.

Then install the packages using the commands:

```
sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
```


## Step 4. Simple CUDA / cuDNN program example

The C++ program `cudnn_softmax` implements a simple cuDNN example
that initializes a CUDA tensor with random values, applies
the cuDNN Softmax operation to it, and prints top 5 values of
the result:

```
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cassert>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

// error handling

void Error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

// CUDA helpers

void CallCuda(cudaError_t stat) {
    if (stat != cudaSuccess) {
        Error("%s", cudaGetErrorString(stat));
    }
}

void *Malloc(int size) {
    void *ptr = nullptr;
    CallCuda(cudaMalloc(&ptr, size));
    return ptr;
}

void Free(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void Memget(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void Memput(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

template<typename T>
class CudaBuffer {
public:
    CudaBuffer(): 
        m_size(0), m_data(nullptr) { }
    ~CudaBuffer() {
        Done();
    }
public:
    void Init(int size) {
        assert(m_data == nullptr);
        m_size = size;
        m_data = static_cast<T *>(Malloc(size * sizeof(T)));
    }
    void Done() {
        if (m_data != nullptr) {
            Free(m_data);
            m_size = 0;
            m_data = nullptr;
        }
    }
    int Size() const {
        return m_size;
    }
    const T *Data() const {
        return m_data;
    }
    T *Data() {
        return m_data;
    }
    void Get(float *host) const {
        Memget(host, m_data, m_size * sizeof(T));
    }
    void Put(const float *host) {
        Memput(m_data, host, m_size * sizeof(T));
    }
private:
    int m_size;
    T *m_data;
};

// cuDNN helpers

void CallCudnn(cudnnStatus_t stat) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        Error("%s", cudnnGetErrorString(stat));
    }
}

cudnnHandle_t g_cudnnHandle;

void CudnnInit() {
    CallCudnn(cudnnCreate(&g_cudnnHandle));
}

void CudnnDone() {
    cudnnDestroy(g_cudnnHandle);
}

cudnnHandle_t CudnnHandle() {
    return g_cudnnHandle;
}

// wrapper class for softmax primitive

class Softmax {
public:
    Softmax();
    ~Softmax();
public:
    void Init(int n, int c, int h, int w);
    void Done();
    void Forward(const float *x, float *y);
private:
    bool m_active;
    cudnnHandle_t m_handle;
    cudnnTensorDescriptor_t m_desc;
};

Softmax::Softmax(): 
        m_active(false),
        m_handle(nullptr),
        m_desc(nullptr) { }

Softmax::~Softmax() {
    Done();
}

void Softmax::Init(int n, int c, int h, int w) {
    assert(!m_active);
    m_handle = CudnnHandle();
    CallCudnn(cudnnCreateTensorDescriptor(&m_desc));
    CallCudnn(cudnnSetTensor4dDescriptor(
        m_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n,
        c,
        h,
        w));
    m_active = true;
}

void Softmax::Done() {
    if (!m_active) {
        return;
    }
    cudnnDestroyTensorDescriptor(m_desc);
    m_active = false;
}

void Softmax::Forward(const float *x, float *y) {
    assert(m_active);
    static float one = 1.0f;
    static float zero = 0.0f;
    CallCudnn(cudnnSoftmaxForward(
        m_handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &one,
        m_desc,
        x,
        &zero,
        m_desc,
        y));
}

// Top-K helper

void TopK(int count, const float *data, int k, int *pos, float *val) {
    for (int i = 0; i < k; i++) {
        pos[i] = -1;
        val[i] = 0.0f;
    }
    for (int p = 0; p < count; p++) {
        float v = data[p];
        int j = -1;
        for (int i = 0; i < k; i++) {
            if (pos[i] < 0 || val[i] < v) {
                j = i;
                break;
            }
        }
        if (j >= 0) {
            for (int i = k - 1; i > j; i--) {
                pos[i] = pos[i-1];
                val[i] = val[i-1];
            }
            pos[j] = p;
            val[j] = v;
        }
    }
}

// handling input and output

void SetInput(CudaBuffer<float> &b) {
    int size = b.Size();
    std::vector<float> h(size);
    float *p = h.data();
    std::srand(1234);
    for (int i = 0; i < size; i++) {
        p[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    b.Put(p);
}

void GetOutput(const CudaBuffer<float> &b) {
    int size = b.Size();
    std::vector<float> h(size);
    float *p = h.data();
    b.Get(p);
    int top5p[5];
    float top5v[5];
    TopK(size, p, 5, top5p, top5v);
    for (int i = 0; i < 5; i++) {
        printf("[%d] pos %d val %g\n", i, top5p[i], top5v[i]);
    }
}

// main program

int main() {
    CudnnInit();
    Softmax softmax;
    int n = 1;
    int c = 1000;
    int h = 1;
    int w = 1;
    softmax.Init(n, c, h, w);
    int size = n * c * h * w;
    CudaBuffer<float> x;
    CudaBuffer<float> y;
    x.Init(size);
    y.Init(size);
    SetInput(x);
    softmax.Forward(x.Data(), y.Data());
    GetOutput(y);
    y.Done();
    x.Done();
    softmax.Done();
    CudnnDone();
} 
```

The shell script `build_cudnn_softmax.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/cudnn_softmax -I /usr/local/cuda/include \
    cudnn_softmax.cpp \
    -L /usr/local/cuda/lib64 -lcudnn -lcudart 
```

Running this script is straightforward:

```
./build_cudnn_softmax.sh
```

To run the program, use the command line:

```
./bin/cudnn_softmax
```

The program output will look like:

```
[0] pos 754 val 0.00159118
[1] pos 814 val 0.00159045
[2] pos 961 val 0.00159016
[3] pos 938 val 0.00158901
[4] pos 717 val 0.00158761
```


## Step 5. Install PyTorch

To install and use PyTorch, Python interpreter and package installer `pip` are required.
When a new instance is created on Genesis Cloud, Python 3 is automatically preinstalled; 
however, `pip` must be installed explicitly. This can be done using the commands:

```
sudo apt install python3-pip
```

To install PyTorch, visit the [product site](https://pytorch.org/) and select the desired 
configuration as follows:

* PyTorch Build: Stable (1.10.1)
* Your OS: Linux
* Package: Pip
* Language: Python
* Compute Platform: CUDA 11.3

The command for installation of the selected configuration will be presented.
At the time of writing of this article, this commands was:

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Replace `pip3` with `python3 -m pip` and execute the resulting command:

```
python3 -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

On completion of the installation, start Python in the interactive mode and enter a few commands
to validate availability of `torch` package, CUDA device, and cuDNN:

```
python3
>>> import torch
>>> torch.__version__
'1.10.1+cu113'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
'NVIDIA GeForce RTX 3080'
>>> torch.backends.cudnn.version()
8200
```

## Step 6. Inference using torchvision

The [torchvision](https://pytorch.org/vision/stable/index.html) package is part of the PyTorch
project. It includes various computational assets (model architectures, image transformations,
and datasets) facilitating using PyTorch for computer vision. 

The torchvision package is installed automatically together with PyTorch. In this and following
articles we will use torchvision models to demonstrate various aspects of deep learning implementation
on Genesis Cloud infrastructure.

We will start with an example of direct use of torchvision models for image classification.

As input, we will need an arbitrary image containing a single object to be classified.
We will use this [husky image](https://commons.wikimedia.org/wiki/Category:Siberian_Husky#/media/File:Siberian-husky-1291343_1920.jpg)
in our experiments (it is in public domain). Use these commands to create a subdirectory
`data` for holding input files and to downloads the image:

```
mkdir data
wget https://upload.wikimedia.org/wikipedia/commons/4/4b/Siberian-husky-1291343_1920.jpg -O data/husky01.jpg
```

Torchvision image classification models have been trained on the ImageNet dataset and use
1000 image classes labeled by consecutive integer numbers. The text file `imagenet_classes.txt` 
containing class descriptions for all labels can be obtained using this command:

```
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

Torchvision provides an extensive set of 
[image classification models](https://pytorch.org/vision/stable/models.html#classification).
We will use ResNet50 in the following example.

The following Python program `infer_resnet50.py` inputs the image, performs classification
and outputs top 5 results with the respective probabilities.

```
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMG_PATH = "./data/husky01.jpg"

# load the pre-trained model
resnet50 = models.resnet50(pretrained=True)
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

# move the input and model to GPU
if torch.cuda.is_available():
    input = input.to("cuda")
    resnet50.to("cuda")

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

The program performs these main actions:

* loads the pre-trained ResNet50 model
* sets the model in evaluation (inference) mode
* specifies transformations for image pre-processing
* reads the image file using PIL package
* applies transformations to the image
* creates an input batch tensor containing one transformed image
* if CUDA is available, moves both the input batch and the model to CUDA device
* disables gradient computations in PyTorch
* runs inference for the model producing the output tensor
* applies the softmax transformation to the output
* gets labels and probabilities for top 5 results
* reads ImageNet class descriptions
* prints top 5 classes and probabilities in human-readable form

The program uses the same sequence of image transformations as commonly applied
during the ImageNet dataset during the model training.

Combination of `model.eval()` and `torch.no_grad()` calls is commonly used
when running inference with PyTorch models.

To run this program, use the command:

```
python3 infer_resnet50.py
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
library and other images.


## Step 7. Benchmarking the torchvision models

We will perform simple inference benchmarking of torchscript models by running
inference multiple times and measuring the average wall clock time required
for one run.

The Python program `bench_resnet50.py` implements benchmarking of ResNet50:

```
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

name = 'resnet50'
print('Start ' + name)

# create model

name = 'resnet50'
resnet50 = models.resnet50(pretrained=True).cuda()
resnet50.eval()

# create dummy input

input = torch.rand(1, 3, 224, 224).cuda()

# benchmark model

with torch.no_grad():
    for i in range(1, 10):
        resnet50(input)

start = perf_counter()
with torch.no_grad():
    for i in range(1, 100):
        resnet50(input)
end = perf_counter()

elapsed = ((end - start) / 100) * 1000
print('Model {0}: elapsed time {1:.2f} ms'.format(name, elapsed))

# print Top-5 results

output = resnet50(input)
top5 = F.softmax(output, dim=1).topk(5).indices
print('Top 5 results:\n {}'.format(top5))  
```

The program performs the following steps:

* creates ResNet50 model; places the model on CUDA device
* sets the model in evaluation (inference) mode
* creates an input tensor with random dummy contents; places it on CUDA device
* disables gradient computations in PyTorch and benchmarks the model
* prints benchmarking results
* applies the softmax transformation to the outputs
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities

The benchmarking includes 10 "warmup" inference runs
followed by 100 runs for which the total wall clock time is measured.
The measured time is divided by the number of runs and the average
time for one run in milliseconds is displayed.

To run this program, use the command:

```
python3 bench_resnet50.py
```

The program output will look like:

```
Start resnet50
Model resnet50: elapsed time 8.01 ms
Top 5 results:
 tensor([[783, 610, 549, 892, 600]], device='cuda:0')
```

The Python program `bench_model.py` is more general; it implements benchmarking
of almost any given torchvision image classification model:

```
import sys
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 bench_model <model_name>")
 
    name = sys.argv[1]
    print('Start ' + name)

    # create model

    builder = getattr(models, name)
    model = builder(pretrained=True).cuda()
    model.eval()

    # create dummy input

    input = torch.rand(1, 3, 224, 224).cuda()

    # benchmark model

    with torch.no_grad():
        for i in range(1, 10):
            model(input)

    start = perf_counter()
    with torch.no_grad():
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

NOTE: The model `googlenet` is not supported because of the specific format
of input tensors it uses.

The program performs the following steps:

* creates a model builder for the specified model name
* uses this builder to create a model; places the model on CUDA device
* sets the model in evaluation (inference) mode
* creates an input tensor with random dummy contents; places it on CUDA device
* disables gradient computations in PyTorch and benchmarks the model
* prints benchmarking results
* applies the softmax transformation to the outputs
* gets labels and probabilities for top 5 results
* prints top 5 classes and probabilities

The program prints a special formatted line starting with `"#"` that
will be later used for automated extraction of performance metrics.

To run this program for ResNet50, use the command:

```
python3 bench_model.py resnet50
```

The program output will look like:

```
Start resnet50
Model resnet50: elapsed time 7.94 ms
#resnet50;7.940051
Top-5 results
  610 5.20%
  549 4.50%
  783 3.84%
  892 3.42%
  446 3.23%
```

The shell script `bench_all.sh` performs benchmarking of all supported torchvision
models:

```
#!/bin/bash

echo "#head;PyTorch"

python3 bench_model.py alexnet
python3 bench_model.py densenet121
python3 bench_model.py densenet161
python3 bench_model.py densenet169
python3 bench_model.py densenet201
python3 bench_model.py mnasnet0_5
python3 bench_model.py mnasnet1_0
python3 bench_model.py mobilenet_v2
python3 bench_model.py mobilenet_v3_large
python3 bench_model.py mobilenet_v3_small
python3 bench_model.py resnet101
python3 bench_model.py resnet152
python3 bench_model.py resnet18
python3 bench_model.py resnet34
python3 bench_model.py resnet50
python3 bench_model.py resnext101_32x8d
python3 bench_model.py resnext50_32x4d
python3 bench_model.py shufflenet_v2_x0_5
python3 bench_model.py shufflenet_v2_x1_0
python3 bench_model.py squeezenet1_0
python3 bench_model.py squeezenet1_1
python3 bench_model.py vgg11
python3 bench_model.py vgg11_bn
python3 bench_model.py vgg13
python3 bench_model.py vgg13_bn
python3 bench_model.py vgg16
python3 bench_model.py vgg16_bn
python3 bench_model.py vgg19
python3 bench_model.py vgg19_bn
python3 bench_model.py wide_resnet101_2
python3 bench_model.py wide_resnet50_2 
```

Running this script is straightforward:

```
./bench_all.sh >bench_torch.log
```

The benchmarking log will be saved in `bench_torch.log` that later will be
used for performance comparison of various deployment methods.


## Conclusion

Using PyTorch and Python directly is perhaps the most simple and straightforward way
for running inference; however there exist much more performance efficient methods for 
model deployment on the GPU-enabled infrastructure. We will discuss these methods
in the subsequent articles.

