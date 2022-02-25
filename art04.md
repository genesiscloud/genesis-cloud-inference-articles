
# Article 4. Using Triton for production deployment of TensorRT models

This article covers using Triton for production deployment of TensorRT models.

NVIDIA Triton Inference Server is an open source solution created for
fast and scalable deployment of deep learning inference in production. 

Detailed Triton information is available on
the [official product page](https://developer.nvidia.com/nvidia-triton-inference-server).

Various assets (source code, shell scripts, and data files) used in this article
can be found in the supporting
[GitHub repository](https://github.com/lxgo/genesis-kbase/tree/main/art04).

To run examples described in this article we recommend cloning the entire 
[repository](https://github.com/lxgo/genesis-kbase) on your Genesis Cloud instance.
The subdirectory `art04` must be made your current directory.


## Prerequisites

For experimenting with Triton, you can reuse the Genesis Cloud instance
that you configured and used for running examples from the Articles 1, 2, and 3.
In particular, the following software must be installed and configured:

* CUDA 11.3.1
* cuDNN 8.2.1
* Python 3.x interpreter and package installer `pip`
* PyTorch 1.10.1 with torchvision 0.11.2
* TensorRT 8.0.3

Alternatively, you can perform the following steps to create a new instance and 
install the required software as described in the previous articles.

* Create a GPU instance on Genesis Cloud (Article 1, Step 1)
* Install CUDA (Article 1, Step 2)
* Install cuDNN (Article 1, Step 3)
* Install PyTorch (Article 1, Step 5)
* Install TensorRT (Article 3, Step 1)

Furthermore running examples described in this article requires files
containng husky images in raw and preprocessed formats

For your convenience, these files are included in assets available in 
the supporting repository and located in the subdirectory `art04`:

```
input/husky01.jpg       # original husky image
input/husky01.dat       # preprocessed husky image
```

Instructions for obtaining these files are provided in the previous articles:

* Original husky image: Article 1, Step 6
* Preprocessed husky image: Article 2, Step 5

These instructions can be also used for experimenting with different input images.

For Tritorn deployment, we will transform ONNX models to TensorRT plans.
Since TensorRT plans are device-specific, we have not included them
in the repository assets and we will perform respective transformation
using methods similar to those described in Article 3.


## Directory layout

We will use the following directory layout on our instance:

```
factory          # root directory for building Triton components
kbase            # root directory for assets for all articles
    ...          # (optional) working directories for the previous articles
    art04        # working directory with assets for this article
models           # configurations for installed Triton server models
vendor           # root directory for the third party software
transit          # directory for installation of binary components
triton           # Triton installation root directory
```


## Step 1. Install CMake

We will build all the required Triton components from the source code.
For this purpose we will use CMake, an extensible, open-source system 
that manages the build process in an operating system and in a compiler-independent manner.
At the time of writing, TensorRT required minimum CMake version 3.18 and
only version 3.16 was available for the automated installation using the `apt`
tool. We will therefore install the recent version CMake from the product
[download page](https://cmake.org/download/). At the time of writing
this was version 3.22.2.

We will use the directory `~/vendor` for installation of the third party tool.
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

Now you can remove the installation script `cmake-3.22.2-linux-x86_64.sh`.


## Step 2. Install Triton server

We will use Triton server 2.18.0, which was the most recent stable release at the time of writing.

NOTE: NVIDIA uses dual versioning nomenclature for Triton components. A Triton component
can be labeled both via the regular release identifier (like the above 2.18.0) or via 
the version of corresponding NVIDIA GPU-optimized container (NGC). For example, regular server
release 2.18.0 corresponds to NGC 22.01. Although we will not use NGC in this article,
the NGC nomenclature must be taken in account while installing some Triton components.

We will download and build Triton components in the directory `~/factory` and install
built packages in `~/triton`. Directory `~/transit` will be used for temporary storing
the downloaded software components. Create these directories if they don't exist yet:

```
mkdir -p ~/transit
mkdir -p ~/factory
mkdir -p ~/triton
```

Make `~/transit` your current directory:

```
cd ~/transit
```

Install the dependencies required for builting Triton server as follows:

```
sudo apt-get update
sudo apt-get install rapidjson-dev
sudo apt-get install libboost-dev
sudo apt-get install libre2-dev
sudo apt-get install libb64-dev
sudo apt-get install libnuma-dev
```

The source code for Triton server is available in this
[GitHib repository](https://github.com/triton-inference-server/server).
To get release 2.18.0, keep `~/transit` your current directory and
get the archived source code as specified on the 
[release page](https://github.com/triton-inference-server/server/releases/tag/v2.18.0):

```
wget https://github.com/triton-inference-server/server/archive/refs/tags/v2.18.0.tar.gz
```

Then make `~/factory` your current directory and unpack the downloaded Triton archive there:

```
cd ~/factory
tar xvfz ~/transit/v2.18.0.tar.gz
```

The archive contents will be unpacked to the subdirectory `server-2.18.0`.
Make it your current directory:

```
cd server-2.18.0
```

The shell script for building the Triton server is provided in the asset directory for this article;
copy it to the current directory and make sure it has execution permissions:

```
cp ~/kbase/art04/build_server.sh .
chmod +x build_server.sh
```

This script has the following contents:

```
#!/bin/bash

python3 ./build.py \
    --no-container-build \
    --cmake-dir=/home/ubuntu/factory/server-2.18.0/build \
    --build-dir=/home/ubuntu/factory/server-2.18.0/scratch \
    --install-dir=/home/ubuntu/triton/server \
    --enable-logging \
    --enable-stats \
    --enable-tracing \
    --enable-metrics \
    --enable-gpu \
    --endpoint=http \
    --endpoint=grpc \
    --repo-tag=common:r22.01 \
    --repo-tag=core:r22.01 \
    --repo-tag=backend:r22.01 \
    --repo-tag=thirdparty:r22.01 \
    --backend=ensemble \
    --backend=tensorrt:r22.01 \
    --repoagent=checksum
```

This script specifies a lightweight Triton configuration supporting only TensorRT
and ensemble backends. Since TensorRT represents by far the most efficient method
of deployment of deep learning models on the NVIDIA GPU platforms we will not
need, for the purpose of oyr study, any other backends supported by Triton.

Note that the release identification `r22.01` corresponds to the NGC nomenclature.

To build Triton server, start this script:

```
./build_server.sh
```

It will take some time to complete. Upon completion, the Triton server
will be installed in directory `~/triton/server`.

To validate the build results, use the command:

```
~/triton/server/bin/tritonserver --help
```

It will print all command line options for the Triton server.


## Step 3. Install Triton client libraries

Triton clients send inference requests to the Triton server and receive
inference results. Triton supports HTTP and gRPC protocols. In this article
we will consider only HTTP. The application programming interfaces (API)
for Triton clients are available in Python and C++.

We  will build the Triton client libraries from the source code
which is available in this
[GitHib repository](https://github.com/triton-inference-server/client).

For building the libraries with the gRPC protocol support, 
install the dependency packages as follows:

```
sudo apt update
sudo apt install libopencv-dev python3-opencv
sudo apt install python3-grpc-tools
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install uuid-dev
```

These packages are not required for building the libraries supporting
the HTTP protocol only.

Then make `~/factory` your current directory and clone the repository 
branch corresponding to the desired release of Triton software:

```
cd ~/factory
git clone -b r22.01 https://github.com/triton-inference-server/client
```

Note that the release identification `r22.01` corresponds to the NGC nomenclature.

The archive contents will be unpacked to the subdirectory `client`.
Create subdirectory `client/build` and make it your current directory:

```
mkdir -p client/build
cd client/build
```

The shell script for creating CMake build configuration for the Triton client libraryis provided in the asset directory for this article; copy it to the current directory and make sure it has execution permissions:

```
cp ~/kbase/art04/build_client.sh .
chmod +x build_client.sh
```

This script has the following contents:

```
#!/bin/bash

cmake \
    -DCMAKE_INSTALL_PREFIX=/home/ubuntu/triton/client \
    -DTRITON_ENABLE_CC_HTTP=ON \
    -DTRITON_ENABLE_CC_GRPC=ON \
    -DTRITON_ENABLE_PERF_ANALYZER=ON \
    -DTRITON_ENABLE_PYTHON_HTTP=ON \
    -DTRITON_ENABLE_PYTHON_GRPC=ON \
    -DTRITON_ENABLE_JAVA_HTTP=OFF \
    -DTRITON_ENABLE_GPU=ON \
    -DTRITON_ENABLE_EXAMPLES=ON \
    -DTRITON_ENABLE_TESTS=ON \
    -DTRITON_COMMON_REPO_TAG=r22.01 \
    -DTRITON_THIRD_PARTY_REPO_TAG=r22.01 \
    -DTRITON_CORE_REPO_TAG=r22.01 \
    -DTRITON_BACKEND_REPO_TAG=r22.01 \
    ..
```

NOTE: If you do not intend to use the gRPC protocol, set values of
`TRITON_ENABLE_CC_GRPC` and `TRITON_ENABLE_PYTHON_GRPC` in this
script to `OFF`. This will significantly reduce the build time.
Examples in this article do not use the gRPC protocol.
The respective lines in the script will look like:

```
    -DTRITON_ENABLE_CC_GRPC=OFF \
    ...
    -DTRITON_ENABLE_PYTHON_GRPC=OFF \
```

Verify that CMake is accessible:

```
cmake --version
```

If necessary, add reference to CMake binary executables to `PATH` 
environment variable as described above.

To create CMake build configuration, start this script:

```
./build_client.sh
```

Once the build configuration is ready, use this command
to build and install the Triton client libraries:

```
make
```

This will take some time to complete. Upon completion, the Triton client libaries
will be installed in directory `~/triton/client`.

The Python API must be installed using the generated wheel file
`~/triton/client/python/tritonclient-0.0.0-py3-none-manylinux1_x86_64.whl`:

```
python3 -m pip install ~/triton/client/python/tritonclient-0.0.0-py3-none-manylinux1_x86_64.whl[all]
```

(Here the suffix `[all]` specifies support of both HTTP and gRPC protocols; 
if you need only the HTTP protocol, use the suffix `[http]` instead.)

To validate the installation, start the Python interpreter and import
the `tritonclient.http` package:

```
python3
>>> import tritonclient.http
```


## Step 4. Configure and start Triton server

We will start with creation of an image classification model in
TensorRT plan format and installing it in the server model directory.

Set the asset directory for this article your current directory:

```
cd ~/kbase/art04
```

Make sure that all shell scripts in this directory have execution permissions:

```
chmod +x *.sh
```

Then produce the ONNX representation for the torchvision ResNet model
using the Python program `generate_onnx_resnet50.py`:

```
import torch
import torchvision.models as models

input = torch.rand(1, 3, 224, 224)

model = models.resnet50(pretrained=True)
model.eval()
output = model(input)
torch.onnx.export(
    model,
    input,
    "./onnx/resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True)
```

This program is similar to one introduced in the Article 3 (Step 3).
The only difference is that here we specify fixed names
`"input"` and `"output"` for the model input and output tensors.
These names are required for the model configuration and
fixing them will simplify the process.

We store generated ONNX files in the subdirectory `onnx` which
must be created before running the program:

```
mkdir -p onnx
```

To run the program, use the command:

```
python3 generate_onnx_resnet50.py
```

The program will produce a file `resnet50.onnx` containing the ONNX model representation.

Then we convert the ONNX representation to the TensorRT plan.
For this purpose we use the Python program `trt_onnx_parser.py' identical
to one introduced in the Article 3 (Step 4).
We store generated plan files in the subdirectory `plan` which
must be created before running the program:

```
mkdir -p plan
```

To run this program for conversion of ResNet50 ONNX representation, use the command:

```
python3 trt_onnx_parser.py ./onnx/resnet50.onnx ./plan/resnet50.plan
```

We will store all the installed Triton models in the directory `~/models`.
Create this directory if it does not yet exist:

```
cd -p ~/models
```

Each installed model must be stored in a separate subdirectory.
This subdirectory has the following layout:

```
models                      # root directory for installed models
    resnet50                # subdirectory for ResNet50 model
        config.pbtxt        # model configuration in protobuf text format
        labels.txt          # ImageNet labels in text format
        1                   # Subdirectory for version 1 of the model
            model.plan      # TensorRT plan
```

The model subdirectory name (`resnet50` in this example) is used to 
identify the model in requests submitted to the server.
The server will use `label.txt` to convert the numeric output tensor
in a human-readable text response. Each model version must be installed
in a separate subdirectory (in this example there is only one version
stored in subdirectory `1`). The TensorRT plan file must have the fixed name
`model.plan`.

The configuration file `config.pbtxt` contains model configuration:

```
platform: "tensorrt_plan"
max_batch_size: 0
input [
  {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    reshape { shape: [ 1, 1000 ] }
    label_filename: "labels.txt"
  }
]
```

The fields in this file have the following meaning:

* `platform`: backend used for handling this model
* `max_batch_size`: maximum batch size
* `input`: list of specifications of input tensors
* `output`: list of specifications of output tensors

The field `max_batch_size` must have non-zero value of the model
supports variable batch size specified by the client request.
For the models with fixed batch size (as in this example)
this field must be set to zero.

The files `config.pbtxt` and `labels.txt` are provided in
the asset directory for this article. To install all the required
model configuration files, copy them to the server model directory
as follows:

```
mkdir -p ~/models/resnet50
mkdir -p ~/models/resnet50/1
cp config.pbtxt ~/models/resnet50
cp labels.txt ~/models/resnet50
cp plan/resnet50.plan ~/models/resnet50/1/model.plan
```

Now the server can be started. Open a new terminal session for
this Genesis Cloud instance and make the asset directory for this article
your current directory.

Use the shell script `start_server.sh` for starting the server: 

```
#!/bin/bash

~/triton/server/bin/tritonserver \
    --backend-directory=/home/ubuntu/triton/server/backends \
    --model-repository ~/models \
    --allow-http 1 \
    --http-port 8000
```

This script specifies several configurable server options.
The full list of supported options and their default values can be
obtained by starting the server with the single `--help option`:

```
~/triton/server/bin/tritonserver --help
```

Leave the server running and switch back to the terminal session
that you previously used for building the Triton software.
The server must be up and running during experiments with
the client software.

NOTE: For these experiments, we run the Triton server from the command
line in a separate terminal session. For the production use, the server
can be installed as a service running in the background.


## Step 5. Build Python client for image classification

The Python program `image_client.py` implements a simple Triron client
for image classification requests. It represents a scaled down version
of the code available in
[Triton Client Libraries and Examples](https://github.com/triton-inference-server/client)
repository on GitHub.
For simplicity, we removed support for gRPC and asynchronous communication.

```
import argparse
import os
import sys

from PIL import Image
import numpy as np
from attrdict import AttrDict

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

FLAGS = None

def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the requirements 
    for an image classification network (as expected by this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '" + model_metadata.name + 
            "' output type is " + output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of dimensions
    # as long as all but 1 is size 1 (e.g. {10}, {1, 10}, {10, 1, 1} are all ok).
    # Ignore the batch dimension if there is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting the batch dimension)
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name, len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if (input_config.format != mc.ModelInput.FORMAT_NCHW and
            input_config.format != mc.ModelInput.FORMAT_NHWC):
        raise Exception(
            "unexpected input format " +
            mc.ModelInput.Format.Name(input_config.format) +
            ", expecting " +
            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
            " or " +
            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size, 
        input_metadata.name,
        output_metadata.name, 
        c, 
        h, 
        w, 
        input_config.format,
        input_metadata.datatype)

def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered

def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """
    output_array = results.as_numpy(output_name)
    output_array_type = output_array.dtype.type

    # Include special handling for non-batching models
    if not batching:
        output_array = [output_array]

    if len(output_array) != batch_size:
        raise Exception("expected {} results, got {}".format(
            batch_size, len(output_array)))

    for results in output_array:
        for result in results:
            if output_array_type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))

def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    inputs = [httpclient.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)
    outputs = [httpclient.InferRequestedOutput(output_name, class_count=FLAGS.classes)]
    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version

def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)
    return _model_metadata, _model_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action="store_true",
        required=False,
        default=False,
        help='Enable verbose output')
    parser.add_argument(
        '-m',
        '--model-name',
        type=str,
        required=True,
        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        required=False,
        default=1,
        help='Batch size. Default is 1.')
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        required=False,
        default=1,
        help='Number of class results to report. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8000',
        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument(
        'image_filename',
        type=str,
        nargs='?',
        default=None,
        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    model_metadata, model_config = convert_http_metadata_config(model_metadata, model_config)

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config)

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    # Preprocess the images into input data according to model requirements
    image_data = []
    for filename in filenames:
        img = Image.open(filename)
        image_data.append(preprocess(img, format, dtype, c, h, w, FLAGS.scaling))

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False

    sent_count = 0

    while not last_request:
        input_filenames = []
        repeated_image_data = []

        for idx in range(FLAGS.batch_size):
            input_filenames.append(filenames[image_idx])
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]

        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    batched_image_data, input_name, output_name, dtype, FLAGS):
                sent_count += 1
                responses.append(
                    triton_client.infer(
                        FLAGS.model_name,
                        inputs,
                        request_id=str(sent_count),
                        model_version=FLAGS.model_version,
                        outputs=outputs))

        except InferenceServerException as e:
            print("inference failed: " + str(e))
            sys.exit(1)

    for response in responses:
        this_id = response.get_response()["id"]
        print("Request {}, batch size {}".format(this_id, FLAGS.batch_size))
        postprocess(response, output_name, FLAGS.batch_size, (max_batch_size > 0))

    print("DONE") 
```

Before running this program, install the required Python dependency as follows:

```
python3 -m pip install attrdict
```

To run the client to send the image classification request for
the input image `input/husky01.jpg` use the command:

```
python3 image_client.py -m resnet50 -c 5 -s INCEPTION input/husky01.jpg
```

This command uses the following client options:

* `-m` model name
* `-c` number of top output labels
* `-s` image preprocessing (scaling) algorithm

The output will like like:

```
Request 1, batch size 1
    19.308859 (250) = SIBERIAN HUSKY
    18.734100 (248) = ESKIMO DOG
    16.524956 (249) = MALAMUTE
    12.580538 (269) = TIMBER WOLF
    12.521436 (273) = DINGO
DONE
```

The list of all supported client options can be obtained using
the command:

```
python3 image_client.py -h
```

NOTE: Scaling (preprocessing) algorithms implemented by `image_client.py`
have been inherited from the original NVIDIA example code and differ
from the preprocessing algorithm that has been used for training of
torchvision models and appplied in the previous articles.
We are using the `INCEPTION` algorithm in this example as it most closely
matches the correct algorithm and yields reasonable classification results.
The interested reader can modify the code `image_client.py` to implement
the original preprocessing algorithm.


## Step 6. Build C++ client for image classification

The C++ program `image_client.cpp` implements a simple Triron client
for image classification requests. It represents a scaled down version
of the code available in
[Triton Client Libraries and Examples](https://github.com/triton-inference-server/client)
repository on GitHub.
To keep the code concise, we implemented various simplifications which include:

* removed support for gRPC and asynchronous communication
* hardcoded settings for the less important options
* removed most of the model validation code
* removed image preprocessing: an already preprocessed image is expected as input

```
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include "http_client.h"

namespace tc = triton::client;

namespace {

void CheckError(const tc::Error &err, const char *msg) {
    if (!err.IsOk()) {
        std::cerr << "Error: " << msg << ": " << err << std::endl;
        exit(1);
    }
}

struct ModelInfo {
    std::string output_name;
    std::string input_name;
    std::string input_datatype;
};

void ParseModel(const rapidjson::Document &model_metadata, ModelInfo &model_info) {
    const auto &input_itr = model_metadata.FindMember("inputs");
    const auto &output_itr = model_metadata.FindMember("outputs");

    const auto &input_metadata = *input_itr->value.Begin();
    const auto &output_metadata = *output_itr->value.Begin();

    model_info.output_name = std::string(
        output_metadata["name"].GetString(),
        output_metadata["name"].GetStringLength());
    model_info.input_name = std::string(
        input_metadata["name"].GetString(),
        input_metadata["name"].GetStringLength());
    model_info.input_datatype = std::string(
        input_metadata["datatype"].GetString(),
        input_metadata["datatype"].GetStringLength());
}

void FileToInputData(const std::string &filename, std::vector<uint8_t> &input_data) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open '" << filename << "'" << std::endl;
        exit(1);
    }
    size_t size = 3 * 224 * 224 * sizeof(float);
    input_data.resize(size);
    ifs.read(reinterpret_cast<char *>(input_data.data()), size);
    ifs.close();
}

void Postprocess(
        tc::InferResult *result,
        const std::string &filename,
        const std::string &output_name,
        size_t topk) {
    if (!result->RequestStatus().IsOk()) {
        std::cerr << "inference failed with error: "
            << result->RequestStatus() << std::endl;
        exit(1);
    }

    std::vector<std::string> result_data;
    tc::Error err = result->StringData(output_name, &result_data);
    CheckError(err, "unable to get output data");

    size_t result_size = result_data.size();
    if (result_size != topk) {
        std::cerr << "unexpected number of strings in the result"
            << ", expected " << topk << ", got " << result_size << std::endl;
        exit(1);
    }

    std::cout << "Image '" << filename << "':" << std::endl;
    for (size_t c = 0; c < topk; c++) {
        std::istringstream is(result_data[c]);
        int count = 0;
        std::string token;
        while (getline(is, token, ':')) {
            if (count == 0) {
                std::cout << "    " << token;
            } else if (count == 1) {
                std::cout << " (" << token << ")";
            } else if (count == 2) {
                std::cout << " = " << token;
            }
            count++;
        }
        std::cout << std::endl;
    }
}

tc::Error ParseJson(rapidjson::Document *document, const std::string &json_str) {
    document->Parse(json_str.c_str(), json_str.size());
    if (document->HasParseError()) {
        return tc::Error(
            "failed to parse JSON at" + std::to_string(document->GetErrorOffset()) +
            ": " + std::string(GetParseError_En(document->GetParseError())));
    }
    return tc::Error::Success;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: image_client <model_name> <input_filename>" << std::endl;
        exit(1);
    }

    std::string model_name{argv[1]};
    std::string input_filename{argv[2]};

    bool verbose = false;
    int topk = 5;
    std::string model_version = "";
    std::string url = "localhost:8000";
    tc::Headers http_headers; // empty

    std::vector<int64_t> shape{3, 224, 224};

    tc::Error err;

    // Create the inference client for the server.
    std::unique_ptr<tc::InferenceServerHttpClient> http_client;
    err = tc::InferenceServerHttpClient::Create(&http_client, url, verbose);
    CheckError(err, "unable to create client for inference");

    std::string model_metadata;
    err = http_client->ModelMetadata(&model_metadata, model_name, model_version, http_headers);
    CheckError(err, "failed to get model metadata");
    rapidjson::Document model_metadata_json;
    err = ParseJson(&model_metadata_json, model_metadata);
    CheckError(err, "failed to parse model metadata");

    ModelInfo model_info;
    ParseModel(model_metadata_json, model_info);

    // Read input data
    std::vector<uint8_t> input_data;
    FileToInputData(input_filename, input_data);

    // Initialize the inputs with the data.
    tc::InferInput *input;
    err = tc::InferInput::Create(&input, model_info.input_name, shape, model_info.input_datatype);
    CheckError(err, "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr(input);

    tc::InferRequestedOutput *output;
    // Set the number of classification expected
    err = tc::InferRequestedOutput::Create(&output, model_info.output_name, topk);
    CheckError(err, "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

    std::vector<tc::InferInput *> inputs{input_ptr.get()};
    std::vector<const tc::InferRequestedOutput *> outputs{output_ptr.get()};

    // Configure context
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;

    // Prepare request
    err = input_ptr->Reset();
    CheckError(err, "failed resetting input");
    err = input_ptr->AppendRaw(input_data);
    CheckError(err, "failed setting input");
    options.request_id_ = "0";

    // Send request
    tc::InferResult *result;
    err = http_client->Infer(&result, options, inputs, outputs, http_headers);
    CheckError(err, "failed sending synchronous infer request");
    std::unique_ptr<tc::InferResult> result_ptr(result);

    // Post-process result to make prediction
    Postprocess(result_ptr.get(), input_filename, model_info.output_name, topk);

    return 0;
}
```

The shell script `build_image_client.sh` must be used to compile and link this program:

```
#!/bin/bash

mkdir -p ./bin

CLI_INC=~/triton/client/include
CLI_LIB=~/triton/client/lib

g++ -o ./bin/image_client --std=c++11 \
    -I $CLI_INC \
    image_client.cpp \
    -L $CLI_LIB -lhttpclient
```

Running this script is straightforward:

```
./build_image_client.sh
```

The program has two command line arguments: a name of the installed model and
a path to the file containing the pre-processed input image.

Before running it, update the `PATH` environment variable as follows:

```
export LD_LIBRARY_PATH=/home/ubuntu/triton/client/lib:$LD_LIBRARY_PATH
```

To run this program for the previously installed `resnet50` model and husky image, 
use the command:

```
./bin/image_client resnet50 input/husky01.dat
```

The program output will look like:

```
Image 'input/husky01.dat':
    15.620337 (250) = SIBERIAN HUSKY
    15.476687 (248) = ESKIMO DOG
    13.487585 (249) = MALAMUTE
    11.918570 (537) = DOGSLED
    10.572811 (247) = SAINT BERNARD
```


## Acknowledgements

Examples `image_code.py` and `image_code.cpp` are derived from the respective
sample code published in
[Triton Client Libraries and Examples](https://github.com/triton-inference-server/client)
repository on GitHub. The original code is distributed under the following license
terms:

```
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

