
# Article 5. Introduction to transformer models and Hugging Face library

This article opens a new series of tutorials covering various aspects
of deployment of deep learning solutions on Genesis Cloud infrastructure.
This new series will focus on using deep learning transformer
models from the Hugging Face library.

This article contains an introduction to transformer models.
It will guide you through the basic steps required to install 
the library and associated software components on a Genesis Cloud
GPU instance. It then presents a representative set
of natural language processing (NLP) tasks that can be solved
using the transformer models.

The following topics are covered:

* Creation of a Genesis Cloud GPU instance
* Installation of Hugging Face Transformers and related software
* Using transformers for solving various NLP problems
* Inference using transformer models models with Python

We will use a Genesis Cloud instance equipped with NVIDIA® GeForce™ RTX 3080 GPU and the following software versions:

* OS: Ubuntu 20.04
* CUDA 11.6.1
* cuDNN 8.4.1
* PyTorch 1.12.1
* Hugging Face Transformers v4.21.3

Various assets (source code, shell scripts, and data files) used in this article
can be found in the supporting
[GitHub repository](https://github.com/lxgo/genesis-kbase/tree/main/art05).

To run examples described in this article we recommend cloning the entire 
[repository](https://github.com/lxgo/genesis-kbase) on your Genesis Cloud instance.
The subdirectory `art05` must be made your current directory.

The texts used in the example programs in this article are borrowed from Wikipedia.


## Introduction

A _transformer_ is a deep learning model that adopts the mechanism of self-attention, by
differentially weighting the significance of each part of the input data.
The fundamental principles of transformer models are described in this
[paper](https://arxiv.org/pdf/1706.03762.pdf).

Hugging Face Transformers library provides APIs and tools to easily download and train 
state-of-the-art pretrained models. These models support common tasks in several
problem domains. Furthermore, Transformers support framework interoperability 
between PyTorch, TensorFlow, and JAX. 

In this series of articles, we will focus on the Natural Language Processing
(NLP) tasks using the PyTorch framework.

The detailed library documentation can be found at the
[Hugging Face Transformers site](https://huggingface.co/docs/transformers/v4.22.1/en/index).

Steps 1 to 4 explain how to create a GPU instance on Genesis Cloud
and how to install the required basic software. They repeat the instructions originally
published in the first article of this series and are placed here for the reader's
convenience. Starting from Step 5, the new material focusing on the transformer
models will be introduced.


## Step 1. Creating a GPU instance on Genesis Cloud

We assume that you have an account on Genesis Cloud. We start by creating
a new GPU instance that will be used to run examples described in this
and several following articles.

To create a new instance, visit a [page](https://compute.genesiscloud.com/dashboard/instances/create) 
titled "Create New Instance". On this page:

* Choose a meaningful Hostname and, optionally, a Nickname
* In Select Location: keep default setting
* In Select Instance Type: choose GPU NVIDIA GeForce RTX 3080
* In Configuration: keep default values
* **Do not** select "Install NVIDIA GPU driver 470" (or any other driver version, if mentioned)
* In Select Image: choose Ubuntu 20.04
* Authentication: select method (SSH Key is recommended)
 
Once ready, click the "Create Instance" button.

Once your instance is ready, log in and proceed with the following steps.


## Step 2. Install CUDA

As the next step, we will install CUDA.

To install the desired version of CUDA, visit the 
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) page.
Select the line for CUDA Toolkit 11.6.1. You will be redirected to the
[corresponding page](https://developer.nvidia.com/cuda-11-6-1-download-archive).
On this page, make the following selections:

* Operating System: Linux
* Architecture: x86_64
* Distribution: Ubuntu
* Version: 20.04
* Installer Type: deb (local)

The sequence of commands for installation of the selected version will be presented.
At the time of writing this article, these commands were:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

As these commands might change in the future, we recommend using the commands that are actually presented
on this page. 

Enter these commands one by one (or build and execute the respective shell script).
The last command will start the CUDA installation process which may take some time.

For this and similar installation steps, we recommend to create a scratch directory
(for example, `~/transit`) and set it as current directory during the installation:

```
mkdir -p ~/transit
cd ~/transit
```

After successful installation, reboot your instance by stopping and starting it
from the Genesis Cloud Web Console.

We strongly advise you to take time and study [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
available by the reference on this page. 

To validate CUDA installation, type the command:

```
nvidia-smi
```

You should get the output looking like:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:00:05.0 Off |                  N/A |
|  0%   28C    P8     7W / 320W |      5MiB / 10240MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       902      G   /usr/lib/xorg/Xorg                  4MiB |
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
For each combination there are two packages of interest representing the runtime and developer libraries.
At the time of writing this article, for CUDA 11.3 and cuDNN 8.2.1, these packages were:

```
libcudnn8_8.4.1.50-1+cuda11.6_amd64.deb
libcudnn8-dev_8.4.1.50-1+cuda11.6_amd64.deb
```

Download these files by entering the respective `wget` commands, for example:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.4.1.50-1+cuda11.6_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_8.4.1.50-1+cuda11.6_amd64.deb
```

We recommend to install from a separate scratch directory, e.g., `~/transit`.

Then install the packages using the commands:

```
sudo dpkg -i libcudnn8_8.4.1.50-1+cuda11.6_amd64.deb
sudo dpkg -i libcudnn8-dev_8.4.1.50-1+cuda11.6_amd64.deb
```


## Step 4. Install PyTorch

To install and use PyTorch, Python interpreter and package installer `pip` are required.
When a new instance is created on Genesis Cloud, Python 3 is automatically preinstalled; 
however, `pip` must be installed explicitly. This can be done using the commands:

```
sudo apt install python3-pip
```

To install PyTorch, use this command:

```
python3 -m pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

Once the installation is finished, launch Python in interactive mode and enter a few commands
to validate the availability of `torch` package, CUDA device, and cuDNN:

```
python3
>>> import torch
>>> torch.__version__
'1.12.1+cu116'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
'NVIDIA GeForce RTX 3080'
>>> torch.backends.cudnn.version()
8302
```


## Step 5. Install Transformers

To install the Transformers library v4.21.3, use this command:

```
python3 -m pip install transformers==4.21.3
```

Pretrained Hugging Face models are downloaded and cached on the local drive.
Downloading takes place when the model is referenced the first time in
the user program.
The default cache location is `~/.cache/huggingface/transformers/`.
Optionally, you can change the cache location by setting the `TRANSFORMERS_CACHE`
environment variable. For example, to set the cache location to
`~/transformers/cache', use these commands:

```
mkdir -p ~/transformers/cache
export TRANSFORMERS_CACHE=~/transformers/cache
```

To verify the installation, start the Python interpreter and enter these commands:

```
python3
>>> from transformers import pipeline
>>> print(pipeline('sentiment-analysis')('Genesis Cloud: Unbeatable GPU power!'))
...
[{'label': 'POSITIVE', 'score': 0.965853214263916}]
```

These commands will start the sentiment analysis pipeline using the sentence
"Genesis Cloud: Unbeatable GPU power!" as input. Sentiment analysis represents
a text classification task that involves labeling the input text as negative or positive. 
The above command classifies the input text as positive with
a confidence score of about 96.5%.

Note that, for simplicity, this command does not specify the exact transformer model,
therefore the library will choose the reasonable default and issue the respective
notice. Also, when the model is used the first time, it will be downloaded from
the online repository known as ModelHub and the downloading process will be logged on the screen.
(The respective messages are skipped in the above code snapshot.)


## Step 6. Basic concepts

Transformer models can be used to solve a variety of NLP tasks.
In this article, we will explore examples of Python code demonstrating use of
pre-trained models from the Transformers library for solving these tasks:

* Sequence classification
* Extractive question answering
* Masked language modeling
* Summarization

Transformer models targeting the NLP tasks are supposed to process
human-readable text. However, models implemented using
PyTorch or other similar frameworks require tensors of numerical values
as input. Therefore, input texts must be preprocessed into
a format acceptable for the models. The texts must be converted into
sequences of numbers and packed as tensors.

The conversion of the textual input data is performed by a separate component
known as _tokenizer_. The tokenizer splits the input text into tokens
according to a set of rules. Depending on the type of the tokenizer, the tokens
can represent words or subwords. Each token is assigned a numerical value
known as _token ID_; this assignment is usually implemented through a lookup table.
Different tokenizers may use different sets of rules.
Each pre-trained model has an associated tokenizer; the type of this tokenizer
depends on the architecture of the model.

The Transformers library implements Application Programming Interfaces (API)
that differ by the level of abstraction and degree of flexibility and power.
In this article, we will present Python code examples using both _pipeline_ and _Auto Class_ APIs.

Pipelines represent the easy-to-use abstractions that require a relatively little
effort for coding. Pipelines are good for quick experiments but not recommended
for production use.

The pipeline API provides a function `pipeline` that constructs an object implementing
inference for the specified task. This object can be called to perform inference
for the given input. The input can be passed as a simple text. The output
is represented via Python dictionaries with items containing the task-specific
components of the inference results.

By default, the library automatically selects a model for the specified task.
Optionally, a specific model from the model hub can be requested.

Tasks are specified using the pre-defined text strings.
For example, the Python programs in this article use these strings:

* `"sentiment-analysis"` for sequence classification
* `"question-answering"` for extractive question answering
* `"fill-mask"` for masked language modeling
* `"summarization"` for summarization

The pipeline API is described on this 
[documentation page](https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/pipelines)

The Auto Class API provides a way for direct model use. With this approach, 
the instances of tokenizers and models are constructed and configured explicitly.
The user gets more flexibility and power through direct access to these
objects and the structure of their input and output data.

The Transformers library provides a collection of generic classes like `AutoTokenizer` and `AutoModel`.
These classes implement the method `from_pretrained` that gets the pathname of the pretrained
model as an input and creates the respective tokenizer or model instance. This method will automatically
guess the model architecture and retrieve the pre-trained model weights, configuration, and vocabulary.

The library also provides the task-specific Auto Classes for models. For example, the Python
programs in this article use these task-specific classes:

* `AutoModelForSequenceClassification` for sequence classification
* `AutoModelForQuestionAnswering` for extractive question answering
* `AutoModelForMaskedLM` for masked language modeling
* `AutoModelForSeq2SeqLM` for summarization

The Auto Class API is described on this 
[documentation page](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/auto)

Programmers can also directly use classes specific for the given model architecture
and the task. For example, for the BERT model and the extractive question answering task,
the classes `BertTokenizer` and `BertForQuestionAnswering` can be used.
We will not discuss the use of model-specific classes in this introductory article.


## Step 7. Tokenization and model input

At this step, we will explore the work of a tokenizer associated with
the pre-trained `bert-base-uncased` model.

The Python program `tokenize_bert.py` used the `AutoTokenizer` class
for this purpose:

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text = "Husky is a general term for a dog used in the polar regions."

input = tokenizer(text)
print(input)

text2 = tokenizer.decode(input["input_ids"])
print(text2)
```

The program performs these steps:

* Using the `AutoTokenizer` class to construct a tokenizer instance 
from a pre-trained model specification.
* Specifying the input text.
* Using the tokenizer to convert the input text into a dictionary containing
all required model inputs.
* Printing the conversion results.
* Using the tokenizer to convert (decode) the generated token ID sequence back to the textual representation.
* Printing the decoded text.

Use this command to run the program:

```
python3 tokenize_bert.py
```

The program will display this output:

```
{'input_ids': [101, 20164, 4969, 1110, 170, 1704, 1858, 1111, 170, 3676, 1215, 1107, 1103, 15281, 4001, 119, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
[CLS] Husky is a general term for a dog used in the polar regions. [SEP]
```

(To improve readability, here we have skipped the model downloading report and changed the output layout.)

The tokenizer converted the input text into a tensor of integer token IDs and assigned it to
the item `input_ids`. Furthermore, it also created additional input tensors required by the model.
All transformers require the uniform set of input tensors that include:

* `input_ids` - a sequence of token indices providing numerical representations.
of tokens in the input text
* `token_type_ids` - optional, used when two different sequences are to be joined.
in a single `input_ids` entry, for example, for tasks like classification on pairs of 
sentences or question answering.
* `attention_masks` - a binary tensor indicating to the model which tokens should be 
attended to, and which should not; typically used for padding a variable length input
sequence to represent them as tensors with fixed dimensions.

Note that the tokenizer added two numeric values `101` and `102` respectively at 
the beginning and the end of the token ID sequence. These values correspond
to the special tokens, `CLS` and `SEP` (classifier and separator) visible in the decoded text. 
Usage of special tokens as well as theis numeric values vary for different tokenizer types and
depend on the associated transformer model. Tokenizers generate the required
special tokens automatically.

You can further explore the work of tokenizers for various NLP tasks
described in the following step by adding the respective `print` commands
to the code of example Python programs.


## Step 8. Examples of NLP tasks

In this step, we will explore examples of Python code demonstrating the use of pre-trained models from the Transformers library to solve a representative set of tasks.

For each task, we will demonstrate two approaches based on using the pipeline
and Auto Class APIs respectively.

For simplicity, the example programs in this step do not use GPUs and do not attempt to optimize inference. The respective techniques will be introduced in the next
step dedicated to the benchmarking.


### Task 1. Sequence classification

Sequence classification is the task of classifying sequences according to a given number of classes or labels.
The sentiment analysis represents a subfield of sequence classification, where the goal is specifically to classify the sentiment of a given text sequence as `NEGATIVE` or `POSITIVE`.

The Python program `pipeline_sentiment_analysis.py` performs sentiment
analysis of the input texts using the pipeline API:

```
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

input1 = "Border Collies are extremely energetic, acrobatic, and athletic."
result = classifier(input1)[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

input2 = "Border Collies are infamous for chewing holes in walls and furniture, and for destructive scraping and hole digging, due to boredom."
result = classifier(input2)[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

output = classifier([input1, input2])
for i, result in enumerate(output):
    print(f"[{i}] label: {result['label']}, with score: {round(result['score'], 4)}")
```

This program implements sentiment analysis of two input texts.
Two versions of inference are demonstrated: sequential processing of the inputs one by one
and processing of all inputs grouped in a single batch.
The library automatically selects the model architecture used for the inference.

The program performs these steps:

* Creation of a pipeline for the `sentiment-analysis` task.
* Applying the pipeline separately to each of two inputs and printing its results.
* Applying the pipeline to the list of two inputs and printing its results.

The pipeline returns a dictionary for the single input or a list of dictionaries
for the batch of inputs. Each dictionary represents a result for
one input sequence and has these two items:

* `label` - label (`NEGATIVE` or `POSITIVE`) of a class with the best score.
* `score` - score value assigned to this class.

Use this command to run the program:

```
python3 pipeline_sentiment_analysis.py
```

The program will display this output:

```
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.9995
[0] label: POSITIVE, with score: 0.9998
[1] label: NEGATIVE, with score: 0.9995
```

The Python program `auto_sentiment_analysis.py` performs sentiment
analysis of the input texts using the Auto Class API:

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def infer(model, input):
    with torch.no_grad():
        output = model(**input)
    return torch.softmax(output.logits, dim=1).numpy()

def get_score(result):
    return [round(v * 100, 2) for v in result]

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input1 = "Border Collies are extremely energetic, acrobatic, and athletic."
input2 = "Border Collies are infamous for chewing holes in walls and furniture, and for destructive scraping and hole digging, due to boredom."

input = tokenizer(input1, return_tensors="pt")
result = infer(model, input)

score = get_score(result[0])
print(f"NEGATIVE: {score[0]}% POSITIVE: {score[1]}%")

input = tokenizer(input2, return_tensors="pt")
result = infer(model, input)

score = get_score(result[0])
print(f"NEGATIVE: {score[0]}% POSITIVE: {score[1]}%")

input = tokenizer([input1, input2], padding="max_length", max_length=32, return_tensors="pt")
result = infer(model, input)

for i, r in enumerate(result):
    score = get_score(r)
    print(f"[{i}] NEGATIVE: {score[0]}% POSITIVE: {score[1]}%")
```

This program implements sentiment analysis of two input texts.
Two versions of inference are demonstrated: sequential processing of the inputs one by one
and processing of all inputs grouped in a single batch.
The program explicitly specifies the same model architecture as one that was automatically selected
in the previous pipeline example.

The program performs these steps:

* Setting the model name to `distilbert-base-uncased-finetuned-sst-2-english`
* Using the `AutoTokenizer` class to construct a tokenizer instance 
from a pre-trained model specification
* Using the `AutoModelForSequenceClassification` class to construct a model instance 
from a pre-trained model specification
* Specifying two input texts

For classifying each input text separately:

* Use tokenizer to convert the text into a sequence of tokens.
* Run inference for this sequence.
* From the inference output, get scores for the `NEGATIVE` and `POSITIVE` classes.
* Print the results.

In order to repeat classification for the input texts grouped in a single batch:

* Use the tokenizer to convert the list of input texts into a list of token sequences.
* Run inference for this list of sequences.
* From the inference output, get scores for the `NEGATIVE` and `POSITIVE` classes
for each input sequence.
* Print the results for each input sequence.

The tokenizer returns a dictionary containing items corresponding to the required
model inputs. The key of each item represents the input name (`input_ids` or `attention_mask`)
and the value of each item is the respective PyTorch tensor object.

All sequences in a batch must have the same length. Therefore, the tokenizer 
for the batch input is called with the additional arguments specifying 
the maximum sequence length of 32 and padding of all input sequences
to this length.

The model returns a `SequenceClassifierOutput` object with the field `logits` containing
a tensor of raw logit values for all classes of all batch elements. The score values are computed 
by applying the softmax function to the logit values of each batch element.

Use this command to run the program:

```
python3 auto_sentiment_analysis.py
```

The program will display this output:

```
NEGATIVE: 0.02% POSITIVE: 99.98%
NEGATIVE: 99.95% POSITIVE: 0.05%
[0] NEGATIVE: 0.02% POSITIVE: 99.98%
[1] NEGATIVE: 99.95% POSITIVE: 0.05%
```


### Task 2. Extractive question answering

Extractive question answering is a type of question answering task where the goal is to identify a specific passage of text from a given document that provides the answer to a specific question.

The Python program `pipeline_question_answering.py` performs question
answering using the pipeline API:

```
from transformers import pipeline

def print_result(result):
    answer = result["answer"]
    score = round(result["score"], 4)
    start = result["start"]
    end = result["end"]
    print(f"Answer: {answer}, score: {score}, start: {start}, end: {end}")

qa = pipeline("question-answering")

context = r"""
The Alaskan Malamute is a large breed of dog that was originally bred for
its strength and endurance to haul heavy freight as a sled dog and hound.
The usual colors are various shades of gray and white, sable and white,
black and white, seal and white, red and white, or solid white.
The physical build of the Malamute is compact and strong with substance,
bone and snowshoe feet. Alaskan Malamutes are still in use as sled dogs
for personal travel, hauling freight, or helping move light objects.
However, most Malamutes today are kept as family pets or as show or performance dogs.
Malamutes are very fond of people, a trait that makes them particularly sought-after
family dogs, but unreliable watchdogs as they do not tend to bark.
"""

question1 = "How are Alaskan Malamutes used?"
result = qa(question=question1, context=context)
print_result(result)

question2 = "What are the usual colors of Alaskan Malamutes?"
result = qa(question=question2, context=context)
print_result(result)

question3 = "Are Alaskan Malamutes reliable watchdogs?"
result = qa(question=question3, context=context)
print_result(result)

result = qa(question=[question1, question2, question3], context=context)
for i, r in enumerate(result):
    print(f"[Result {i}]")
    print_result(r)
```

This program specifies the input context and implements extractive question answering for three
input questions.
Two versions of inference are demonstrated: sequential processing of the input questions
one by one and processing of all input questions grouped in a single batch.
The library automatically selects the model architecture used for the inference.

The program performs these three steps:

* Creation of a pipeline for the `question-answering` task.
* Applying the pipeline separately to each of two inputs and printing results.
* Applying the pipeline to the list of two inputs and printing results.

The pipeline returns a dictionary for the single input or a list of dictionaries
for the batch of inputs. Each dictionary represents a result for
one input question and contains these items:

* `answer` - fragment from the input context containing the answer
* `start` - start position of the answer in the input context
* `end` - end position of the answer in the input context
* `score` - score value assigned to the answer

Use this command to run the program:

```
python3 pipeline_question_answering.py
```

The program will display this output:

```
No model was supplied, defaulted to distilbert-base-cased-distilled-squad and revision 626af31 (https://huggingface.co/distilbert-base-cased-distilled-squad).
Using a pipeline without specifying a model name and revision in production is not recommended.
Answer: personal travel, hauling freight, or helping move light objects, score: 0.1061, start: 438, end: 501
Answer: gray and white, sable and white,
black and white, score: 0.1228, start: 189, end: 238
Answer: unreliable watchdogs as they do not tend to bark., score: 0.4684, start: 691, end: 740
[Result 0]
Answer: personal travel, hauling freight, or helping move light objects, score: 0.1061, start: 438, end: 501
[Result 1]
Answer: gray and white, sable and white,
black and white, score: 0.1228, start: 189, end: 238
[Result 2]
Answer: unreliable watchdogs as they do not tend to bark., score: 0.4684, start: 691, end: 740
```

The Python program `auto_question_answering.py` performs question
answering using the Auto Class API:

```
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def print_result(tokenizer, questions, input_ids, result):
    start_scores = result.start_logits
    end_scores = result.end_logits

    for question, token_ids, start_score, end_score in zip(questions, input_ids, start_scores, end_scores):
        # Get the most likely location of answer
        start = torch.argmax(start_score)
        end = torch.argmax(end_score) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(token_ids[start:end])
        )

        print(f"Question: {question}")
        print(f"Answer: {answer}")

model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = r"""
The Alaskan Malamute is a large breed of dog that was originally bred for
its strength and endurance to haul heavy freight as a sled dog and hound.
The usual colors are various shades of gray and white, sable and white,
black and white, seal and white, red and white, or solid white.
The physical build of the Malamute is compact and strong with substance,
bone and snowshoe feet. Alaskan Malamutes are still in use as sled dogs
for personal travel, hauling freight, or helping move light objects.
However, most Malamutes today are kept as family pets or as show or performance dogs.
Malamutes are very fond of people, a trait that makes them particularly sought-after
family dogs, but unreliable watchdogs as they do not tend to bark.
"""

questions = [
    "How are Alaskan Malamutes used?",
    "What are the usual colors of Alaskan Malamutes?",
    "Are Alaskan Malamutes reliable watchdogs?"
]

for question in questions:
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    print_result(tokenizer, [question], input_ids, outputs)

batch_questions = [(question, context) for question in questions]
inputs = tokenizer(
    batch_questions,
    add_special_tokens=True,
    padding="max_length",
    max_length=256,
    return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs)

print_result(tokenizer, questions, input_ids, outputs)
```

This program specifies the input context and implements question answering for three
input questions.
Two versions of inference are demonstrated: sequential processing of the input questions
one by one and processing of all input questions grouped in a single batch.
The program explicitly specifies the same model architecture as one that was automatically selected
in the previous pipeline example.

The program performs these steps:

* Setting the model name to `distilbert-base-cased-distilled-squad`.
* Using the `AutoTokenizer` class to construct a tokenizer instance 
from a pre-trained model specification.
* Using the `AutoModelForQuestionAnswering` class to construct a model instance 
from a pre-trained model specification.
* Specifying an input context and three input questions.

To perform inference for each input question:

* Use the tokenizer to convert a combination of the question and the context into a sequence of tokens.
* Run inference for this sequence.
* From the inference output, get positions of a fragment containing the answer in the input context.
* Print the results.

In order to repeat inference for the questions grouped in a single batch:

* Use tokenizer to convert a list of combinations of each question and the context 
into a sequence of token sequences.
* Run inference for this list of sequences
* From the inference output, get positions of fragments containing the answer in the input context.
* Print the results for each input question.

The tokenizer returns a dictionary containing items corresponding to the required
model inputs. The key of each item represents the input name (`input_ids` or `attention_mask`)
and the value of each item is the respective PyTorch tensor object.

The tokenizer adds special tokens to separate the question and the context in each input sequence.

All sequences in a batch must have the same length. Therefore, the tokenizer 
for the batch input is called with the additional arguments specifying 
the maximum sequence length of 256 and padding of all input sequences
to this length.

The model returns a `QuestionAnsweringModelOutput` object with the fields `start_logits`
and `end_logits` containing tensors of start and end logit values for all tokens of all input sequences. 
For each token, the logit values specify the confidence of this token being the start or the end
of the answer. The positions of the answer in the context are computed by applying the `argmax`
function to the logit tensors. The respective subsequence is extracted from the tokenized context.
The tokenizer is used to convert this subsequence into the human-readable answer text.

Use this command to run the program:

```
python3 auto_question_answering.py
```

The program will display this output:

```
Question: How are Alaskan Malamutes used?
Answer: as sled dogs for personal travel, hauling freight, or helping move light objects
Question: What are the usual colors of Alaskan Malamutes?
Answer: gray and white, sable and white, black and white, seal and white, red and white, or solid white
Question: Are Alaskan Malamutes reliable watchdogs?
Answer: unreliable watchdogs as they don't tend to bark.
Question: How are Alaskan Malamutes used?
Answer: as sled dogs for personal travel, hauling freight, or helping move light objects
Question: What are the usual colors of Alaskan Malamutes?
Answer: gray and white, sable and white, black and white, seal and white, red and white, or solid white
Question: Are Alaskan Malamutes reliable watchdogs?
Answer: unreliable watchdogs as they don't tend to bark.
```


### Task 3. Masked language modeling

Masked language modeling is the task of masking tokens in a sequence with a masking token, 
and prompting the model to predict an appropriate token to fill that mask. 

The Python program `pipeline_fill_mask.py` performs masked language
modeling using the pipeline API:

```
from transformers import pipeline

unmasker = pipeline("fill-mask")

text = f"The German Shepherd is a breed of working {unmasker.tokenizer.mask_token} of medium to large size."

result = unmasker(text)

for i, r in enumerate(result):
    score = round(r["score"], 4)
    token = r["token_str"]
    sequence = r["sequence"]
    print(f"[{i}] Score: {score} Token: [{token}] Sequence: {sequence}")
```

This program implements masked language modeling on the input text.
The library automatically selects the model architecture used for the inference.

The program performs these steps:

* Creation of a pipeline for the `fill-mask` task.
* Specifying the input text with the word "dog" replaced with the special masking token string.
* Applying the pipeline to this input.
* Printing results.

The text string corresponding to the masking token is specific for the used tokenizer
and can be accessed as `unmasker.tokenizer.mask_token`.

The pipeline returns a list of dictionaries containing the top 5 of most likely results. 
Each dictionary in the list represents has these items:

* `score` - score value assigned to this result
* `token` - integer ID for the result token
* `token_str` - text representation of the result token
* `sequence` - text sequence with the masking token replaced with the result token

Use this command to run the program:

```
python3 pipeline_fill_mask.py
```

The program will display this output:

```
No model was supplied, defaulted to distilroberta-base and revision ec58a5b (https://huggingface.co/distilroberta-base).
Using a pipeline without specifying a model name and revision in production is not recommended.
[0] Score: 0.8203 Token: [ dog] Sequence: The German Shepherd is a breed of working dog of medium to large size.
[1] Score: 0.0422 Token: [ dogs] Sequence: The German Shepherd is a breed of working dogs of medium to large size.
[2] Score: 0.0231 Token: [ Shepherd] Sequence: The German Shepherd is a breed of working Shepherd of medium to large size.
[3] Score: 0.0163 Token: [ shepherd] Sequence: The German Shepherd is a breed of working shepherd of medium to large size.
[4] Score: 0.0157 Token: [ animal] Sequence: The German Shepherd is a breed of working animal of medium to large size.
```

The Python program `auto_fill_mask.py` performs masked language
modeling using the Auto Class API:

```
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model_name = "distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = f"The German Shepherd is a breed of working {tokenizer.mask_token} of medium to large size."

inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

with torch.no_grad():
    result = model(**inputs)

token_logits = result.logits
mask_token_logits = token_logits[0, mask_token_index, :]
mask_token_logits = torch.softmax(mask_token_logits, dim=1)

top5 = torch.topk(mask_token_logits, 5, dim=1)
top5_scores = top5.values[0].tolist()
top5_tokens = top5.indices[0].tolist()

for score, token in zip(top5_scores, top5_tokens):
    token_str = tokenizer.decode([token])
    sequence = text.replace(tokenizer.mask_token, token_str)
    print(f"Score: {round(score, 4)} Token: [{token_str}] Sequence: {sequence}")
```

This program implements masked language modeling on the input text.
The program explicitly specifies the same model architecture as one that was automatically selected
in the previous pipeline example.

The program performs these steps:

* Setting the model name to `distilroberta-base`.
* Using the `AutoTokenizer` class to construct a tokenizer instance 
from a pre-trained model specification.
* Using the `AutoModelForMaskedLM` class to construct a model instance 
from a pre-trained model specification.
* Specifying the input text with the word "dog" replaced with the special masking token string.
* Using tokenizer to convert the input text into a sequence of tokens.
* Getting position of the masking token in this sequence.
* Running inference on this sequence.
* Extracting the logits from the inference results.
* Applying the `softmax` operation to the logits and get top 5 results with the highest values.

For each of the top 5 results:

* Use tokenizer to decode the corresponding token ID into a text string
* Construct the answer text by replacing the masking token with this string
* Print the score, the token text, and the answer

The text string corresponding to the masking token and corresponding numeric token ID are 
specific for the used tokenizen and can be accessed as `tokenizer.mask_token` and
`tokenizer.mask_token_id` respectively.

The tokenizer returns a dictionary containing items corresponding to the required
model inputs. The key of each item represents the input name (`input_ids` or `attention_mask`)
and the value of each item is the respective PyTorch tensor object.

The model returns a `MaskedLMOutput` object with the field `logits`
containing a tensor filled with logit values for all tokens in the vocabulary
of the specified tokenizer. The higher logit values signify the higher probability
of the corresponding token being the suitable candidate for the masked word.

Use this command to run the program:

```
python3 auto_fill_mask.py
```

The program will display this output:

```
Score: 0.8203 Token: [ dog] Sequence: The German Shepherd is a breed of working  dog of medium to large size.
Score: 0.0422 Token: [ dogs] Sequence: The German Shepherd is a breed of working  dogs of medium to large size.
Score: 0.0231 Token: [ Shepherd] Sequence: The German Shepherd is a breed of working  Shepherd of medium to large size.
Score: 0.0163 Token: [ shepherd] Sequence: The German Shepherd is a breed of working  shepherd of medium to large size.
Score: 0.0157 Token: [ animal] Sequence: The German Shepherd is a breed of working  animal of medium to large size.
```


### Task 4. Summarization

Summarization is the task of summarizing a document or an article into a shorter text.

The Python program `pipeline_summarization.py` performs summarization
of the input texts using the pipeline API:

```from transformers import pipeline

summarizer = pipeline("summarization")

article = r"""
The Alaskan Malamute is a large breed of dog that was originally bred for
its strength and endurance to haul heavy freight as a sled dog and hound.
The usual colors are various shades of gray and white, sable and white,
black and white, seal and white, red and white, or solid white.
The physical build of the Malamute is compact and strong with substance,
bone and snowshoe feet. Alaskan Malamutes are still in use as sled dogs
for personal travel, hauling freight, or helping move light objects.
However, most Malamutes today are kept as family pets or as show or performance dogs.
Malamutes are very fond of people, a trait that makes them particularly sought-after
family dogs, but unreliable watchdogs as they do not tend to bark.
"""

result = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(result[0]["summary_text"])
```

This program implements summarization on the input text.
The library automatically selects the model architecture used for the inference.

The program performs these steps:

* Creation of a pipeline for the `summarization` task.
* Specifying the input text.
* Applying the pipeline to this input.
* Printing results.

The pipeline returns a list of dictionaries containing the results. 
Each dictionary in the list has the item `summary_text` containing
the summary.

Use this command to run the program:

```
python3 pipeline_summarization.py
```

The program will display this output:

```
No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 and revision a4f8f3e (https://huggingface.co/sshleifer/distilbart-cnn-12-6).
Using a pipeline without specifying a model name and revision in production is not recommended.
 The Alaskan Malamute is a large breed of dog that was originally bred for its strength and endurance to haul heavy freight as a sled dog and hound . Most Malamutes today are kept as family pets or as show or performance dogs .
```

The Python program `auto_summarization.py` performs summarization
of the input texts using the Auto Class API:

```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

article = r"""
The Alaskan Malamute is a large breed of dog that was originally bred for
its strength and endurance to haul heavy freight as a sled dog and hound.
The usual colors are various shades of gray and white, sable and white,
black and white, seal and white, red and white, or solid white.
The physical build of the Malamute is compact and strong with substance,
bone and snowshoe feet. Alaskan Malamutes are still in use as sled dogs
for personal travel, hauling freight, or helping move light objects.
However, most Malamutes today are kept as family pets or as show or performance dogs.
Malamutes are very fond of people, a trait that makes them particularly sought-after
family dogs, but unreliable watchdogs as they do not tend to bark.
"""

inputs = tokenizer(article, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=130, min_length=30)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This program implements summarization on the input text.
The program explicitly specifies the same model architecture as one that was automatically selected
in the previous pipeline example.

The program performs these steps:

* Setting the model name to `sshleifer/distilbart-cnn-12-6`.
* Using the `AutoTokenizer` class to construct a tokenizer instance 
from a pre-trained model specification.
* Using the `AutoModelForSeq2SeqLM` class to construct a model instance 
from a pre-trained model specification.
* Specifying the input text.
* Using tokenizer to convert the input text into a sequence of tokens.
* Using the model `generate` method to generate the sequence of tokens representing the summary.
* Using the tokenizer to convert the output sequence of tokens to the human-readable text.

The tokenizer returns a dictionary containing items corresponding to the required
model inputs. The key of each item represents the input name (`input_ids` or `attention_mask`)
and the value of each item is the respective PyTorch tensor object.

The model `generate` method returns a tensor containing a sequence of tokens
corresponding to the generated summary.

Use this command to run the program:

```
python3 auto_summarization.py
```

The program will display this output:

```
 The Alaskan Malamute is a large breed of dog that was originally bred to haul heavy freight as a sled dog and hound. Most Malamutes today are kept as family pets or as show or performance dogs.
```


## Step 9. Simple transformer inference benchmarking using PyTorch

The Python program `torch_bench.py` can be used to run inference for 
a pre-trained transformer model using PyTorch framework and 
output the performance metrics.

```
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
```

Some of the program command line options are:

* `-m` path to model or URL to Hugging Face hub
* `-b` batch size
* `-s` sequence length

The program loads the specified model and repeatedly performs inference
with the given batch size and sequence length.

The program starts with a number of "warmup" runs specified via the `--warmup` option
followed by a number of measure runs specified via the `--nb-measures` option.
The wall clock time required for completion of each measure run is evaluated
and the respected statistics are displayed upon the completion of all runs.
The randomly generated input is used and the option `--seed` can be specified to initialize
the random number generator.

The program performs these steps:

* Parsing the command line options.
* Verifying that the PyTorch version supports CUDA.
* Setting PyTorch logging level to "error" for suppressing insignificant warnings.
* Initializion of the random number generator.
* Creating a tokenizer object for the specified model name.
* Fetching the names of model inputs from the tokenizer object.
* Creating a model object for the specified model name.
* Using the model `eval` method to configure certain layer types for inference.
* Using the model `cuda` method to specify model execution on the current CUDA device.
* Generating the random content for all model input tensors.
* Launch inference for the given model, input data, and number of runs.
* Printing the timing statistics.

The function `launch_inference` implements a sequence of ONNX inference runs.
It performs these steps:

* Sequentially invoking the inference function for each warm up input, collect the outputs.
* Creating a time buffer for storing the timing statistics.
* Sequentially invoke the inference function for the specified number of measurement runs, and
tracking inference time for each run

The function `infer` implements one inference run.
It uses `torch.no_grad` context manager function to disable gradient calculations.

The function `print_timings` computes and prints these timing statistics for the measurement runs:

* `mean` - mean
* `sd` - standard deviation
* `min` - minimum
* `max` - maximum
* `median` - median
* `95p` - 95% percentile
* `99p` - 99% percentile

To run benchmarking for `bert-base-uncased` model with the batch size of 1 and sequence length of 16, 
use this command:

```
python3 torch_bench.py -m bert-base-uncased -b 1 -s 16
```

For the further benchmarking we will use all combinations the batch size values of 1, 8, 64 and
the sequence length values of 16, 64, and 512. The shell script `torch_bench_all.sh` performs 
benchmarking for `bert-base-uncased` model with these parameter values:

```
#!/bin/bash

python3 torch_bench.py -m bert-base-uncased -b 1 -s 16
python3 torch_bench.py -m bert-base-uncased -b 1 -s 64
python3 torch_bench.py -m bert-base-uncased -b 1 -s 512

python3 torch_bench.py -m bert-base-uncased -b 8 -s 16
python3 torch_bench.py -m bert-base-uncased -b 8 -s 64
python3 torch_bench.py -m bert-base-uncased -b 8 -s 512

python3 torch_bench.py -m bert-base-uncased -b 64 -s 16
python3 torch_bench.py -m bert-base-uncased -b 64 -s 64
python3 torch_bench.py -m bert-base-uncased -b 64 -s 512
```

Running this script is straightforward:

```
./torch_bench_all.sh
```

The script will display results similar to these:

```
[bert-base-uncased] [b=1 s=16] mean=12.77ms, sd=0.51ms, min=12.33ms, max=17.81ms, median=12.50ms, 95p=13.26ms, 99p=15.34ms
[bert-base-uncased] [b=1 s=64] mean=12.81ms, sd=0.56ms, min=12.04ms, max=20.64ms, median=12.76ms, 95p=13.19ms, 99p=15.64ms
[bert-base-uncased] [b=1 s=512] mean=12.77ms, sd=0.54ms, min=11.74ms, max=18.64ms, median=12.76ms, 95p=13.28ms, 99p=15.17ms
[bert-base-uncased] [b=8 s=16] mean=13.00ms, sd=0.44ms, min=12.72ms, max=21.27ms, median=12.92ms, 95p=13.40ms, 99p=15.16ms
[bert-base-uncased] [b=8 s=64] mean=13.41ms, sd=0.31ms, min=12.57ms, max=16.40ms, median=13.41ms, 95p=13.66ms, 99p=14.81ms
[bert-base-uncased] [b=8 s=512] mean=66.96ms, sd=0.26ms, min=66.41ms, max=67.87ms, median=66.93ms, 95p=67.36ms, 99p=67.48ms
[bert-base-uncased] [b=64 s=16] mean=14.61ms, sd=0.38ms, min=12.77ms, max=18.14ms, median=14.67ms, 95p=14.75ms, 99p=15.39ms
[bert-base-uncased] [b=64 s=64] mean=53.68ms, sd=0.25ms, min=52.32ms, max=54.80ms, median=53.74ms, 95p=53.95ms, 99p=54.02ms
[bert-base-uncased] [b=64 s=512] mean=519.02ms, sd=1.52ms, min=513.91ms, max=521.10ms, median=519.62ms, 95p=520.47ms, 99p=520.89ms
```

This timing statistics looks somewhat counter-intuitive. One would expect higher run times for larger batch size or
sequence length values but that is not always a case here. Apparently, PyTorch introduces a hidden overhead
that affects the statistics. Direct use of Python framework is not the most efficient way
of running the deep learning inference. In the following articles of these series, we will explore
more efficient approaches.

