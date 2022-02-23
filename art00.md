# Deployment of Deep Learning models on Genesis Cloud - Tutorials & Benchmarks
## Introduction
We are proud to introduce our new article series that will guide you on how to run state of the art deep learning models on Genesis Cloud infrastructure. These articles will be initially published as blog posts and will be added to our knowledge base ADD LINK after their release. Please note: The order of the articles is important as articles are written as a series and information contained in the initial articles might be required for understanding the subsequent articles.

In this series of articles we will use 1x RTX 3080 instance type on Genesis Cloud (our recommended GPU for inference use) and showcase four (4) different deployment strategies for deep learning inference using (a) PyTorch (TorchScript), (b) TensorRT, and (c) Triton. 

For the models, we will focus on computer vision applications using the torchvision model collection. This collection will serve as an example and includes various pretrained versions of classic deep learning algorithms such as alexnet, densenet, mobilenet, resnet, shufflenet, and squeezenet.

## Articles
* Article 1: PyTorch, torchvision, and simple inference examples - available now ADD LINK
* Article 2: Deployment techniques for PyTorch models using TorchScript - upcoming (early March 2022)
* Article 3: Deployment techniques for PyTorch models using TensorRT - upcoming (March 2022)
* Article 4: Using Triton for production deployment of TensorRT models - upcoming (April 2022)

## Why run deep learning inference on a GPU?
In the early days of machine learning GPUs were mainly used for training deep learning models while inference could still be done on a CPU.
While the field of machine learning progressed immensely in the past 10 years, the models have grown in both size and complexity, meaning that today the standard infrastructure setup for latency-sensitive deep learning applications are based on GPU cloud instances instead of CPU-only instances.

Rationale for using a GPU is not just performance but also cost. Compared to CPUs, GPUs are often two orders of a magnitude more efficient in processing deep neural networks. This means, that cost-savings can be achieved by switching to a GPU instance especially when operating with high throughput applications.

## How to run deep learning inference on a Genesis Cloud GPU instance?
All you need are a Genesis Cloud GPU instance, a trained deep learning model, data to be processed, and the supporting software. We will show you how to master it all.

Each article will contain:
* Installation and/or building instructions for various components
* Necessary background information
* Sample code for validation of installation and further experiments
* Annotations and explanations to help you understand the sample code
* Prebuilt models ready for deployment (when applicable)
* Benchmarking scripts and results (when applicable)

In case you aren't using Genesis Cloud yet, get started here. ADD LINK

**Now start accelerating on machine learning with Genesis Cloud **🚀


## Appendix
### Software
* PyTorch (TorchScript)
* TensorRT
* NVIDIA Triton


### Models
* alexnet
* densenet121
* densenet161
* densenet169
* densenet201
* mnasnet0_5
* mnasnet1_0
* mobilenet_v2
* mobilenet_v3_large
* mobilenet_v3_small
* resnet101
* resnet152
* resnet18
* resnet34
* resnet50
* resnext101_32x8d
* resnext50_32x4d
* shufflenet_v2_x0_5
* shufflenet_v2_x1_0
* squeezenet1_0
* squeezenet1_1
* vgg11
* vgg11_bn
* vgg13
* vgg13_bn
* vgg16
* vgg16_bn
* vgg19
* vgg19_bn
* wide_resnet101_2
* wide_resnet50_2


### Datasets
ImageNet dataset (pictures of dogs labelled by breed)
