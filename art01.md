
# Article 1: Installation and basic use of CUDA, PyTorch, and torchvision

This article will guide you through the basic steps required for installation
and basic use of PyTorch and related softwae components on a Genesis Cloud
GPU instance. The following topics are covered:

* creation of a Genesis Cloud GPU instance
* installation of CUDA and cuDNN
* installation of PyTorch and torchvision
* inference using torchvision image classification models with Python

We will use a Genesis Cloud instance equipped with NVIDIA RTX 3080 GPU
and the following software versions:

* OS: Ubuntu 20.04
* CUDA 11.3
* cuDNN 8.2.0
* PyTorch 1.10.1
* torchvision 0.11.2

## Step 1: Creating a GPU instance on Genesis Cloud

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

## Step 2: Install CUDA

As the next step, we will install CUDA. Although will not use CUDA directly for our first
examples, it makes sense to perform the installation now.

To install the desired version of CUDA, visit the 
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) page.
Select the line for CUDA Toolkit 11.3.0. You will be redirected to the
[corresponding page](https://developer.nvidia.com/cuda-11.3.0-download-archive).
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
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
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

Upon the successful installation, we recommend to reboot your instance by stoppong and starting it
from Genesis Cloud Web console.

We strongly advice to take time and study [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
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

## Step 3: Install cuDNN

The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library 
of primitives for deep neural networks. To install cuDNN, visit the
[distribution page](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/).
Select packages corresponding to the desired combination of CUDA and cuDNN versions.
For each such combination there are two pakages of interest representing the runtime and developer libraries.
At the time of writing of this article, for CUDA 11.3 and cuDNN 8.2.0, these packages were:

```
libcudnn8_8.2.0.53-1+cuda11.3_amd64.deb
libcudnn8-dev_8.2.0.53-1+cuda11.3_amd64.deb
```

Download these files by entering the respective `wget` commands, for example:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.2.0.53-1+cuda11.3_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_8.2.0.53-1+cuda11.3_amd64.deb
```

As before, we recommend to perform installation from a separate scratch dirctory, e.g., `~/transit`.

Then install the packages using the commands:

```
sudo dpkg -i libcudnn8_8.2.0.53-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.0.53-1+cuda11.3_amd64.deb
```

### Step 4: Install PyTorch

To install and use PyTorch, Python interpreter and package installer `pip` are required.
When a new instance is created on Genesis Cloud, Python 3 is authomatically preinstalled; 
however, `pip` must be installed explcitly. This can be done using the commands:

```
sudo apt install python3-pip
```

