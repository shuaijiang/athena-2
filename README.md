
# Athena

*Athena* is an open-source implementation of end-to-end speech processing engine. Our vision is to empower both industrial application and academic research on end-to-end models for speech processing. To make speech processing available to everyone, we're also releasing example implementation and recipe on some opensource dataset for various tasks (ASR, TTS, Voice Conversion, Speaker Recognition, etc).

All of our models are implemented in Tensorflow>=2.0.0. For ease of use, we provide Kaldi-free pythonic feature extractor with [Athena_transform](https://github.com/athena-team/athena-transform).

## 1) Table of Contents

- [Athena](#athena)
  - [1) Table of Contents](#1-table-of-contents)
  - [2) Key Features](#2-key-features)
  - [3) Installation](#3-installation)
    - [3.1) Creating a virtual environment [Optional]](#31-creating-a-virtual-environment-optional)
    - [3.2) Install *tensorflow* backend](#32-install-tensorflow-backend)
    - [3.3) Install *horovod* for multiple-device training [Optional]](#33-install-horovod-for-multiple-device-training-optional)
    - [3.4) Install *pydecoder* for WFST decoding [Optional]](#34-install-pydecoder-for-WFST-decoding-optional)
    - [3.5) Install *athena* package](#35-install-athena-package)
    - [3.6) Test your installation](#36-test-your-installation)
    - [Notes](#notes)
  - [4) Data Preparation](#4-data-preparation)
    - [4.1) Create Manifest](#41-create-manifest)
  - [5) Training](#5-training)
    - [5.1) Setting the Configuration File](#51-setting-the-configuration-file)
    - [5.2) Train a Model](#52-train-a-model)
  - [6) Deployment](#6-deployment)
  - [7) Results](#7-results)
    - [7.1) ASR](#71-asr)
  - [8) Directory Structure](#8-directory-structure)

## 2) Key Features

- Hybrid CTC/Transformer based end-to-end ASR
- Speech-Transformer
- Unsupervised pre-training
- Multi-GPU training on one machine or across multiple machines with Horovod
- End-to-end Tacotron2 based TTS with support for multi-speaker and GST
- WFST-based decoding
- Deployment with Tensorflow C++

## 3) Installation

### 3.1) Creating a virtual environment [Optional]

This project has only been tested on Python 3. We highly recommend creating a virtual environment and installing the python requirements there.

```bash
# Setting up virtual environment
python -m venv venv_athena
source venv_athena/bin/activate
```

### 3.2) Install *tensorflow* backend

For more information, you can checkout the [tensorflow website](https://github.com/tensorflow/tensorflow).

```bash
# we highly recommend firstly update pip
pip install --upgrade pip
pip install tensorflow==2.0.0
```

### 3.3) Install *horovod* for multiple-device training [Optional]

For multiple GPU/CPU training
You have to install the *horovod*, you can find out more information from the [horovod website](https://github.com/horovod/horovod#install).

### 3.4) Install *pydecoder* for WFST decoding [Optional]

For WFST decoding
You have to install *pydecoder*, installation guide for *pydecoder* can be found [athena-decoder website](https://github.com/athena-team/athena-decoder#installation)

### 3.5) Install *athena* package

```bash
git clone https://github.com/athena-team/athena.git
cd athena
pip install -r requirements.txt
python setup.py bdist_wheel sdist
python -m pip install --ignore-installed dist/athena-0.1.0*.whl
```

- Once athena is successfully installed , you should do `source tools/env.sh` firstly before doing other things.
- For installing some other supporting tools, you can check the `tools/install*.sh` to install kenlm, sph2pipe, spm and ... [Optional]

### 3.6) Test your installation

- On a single cpu/gpu

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
python athena/main.py examples/translate/spa-eng-example/transformer.json
```

- On multiple cpu/gpu in one machine (you should make sure your hovorod is successfully installed)

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
horovodrun -np 4 -H localhost:4 athena/horovod_main.py examples/translate/spa-eng-example/transformer.json
```

### Notes

- If you see errors such as `ERROR: Cannot uninstall 'wrapt'` while installing TensorFlow, try updating it using command `conda update wrapt`. Same for similar dependencies such as `entrypoints`, `llvmlite` and so on.
- You may want to make sure you have `g++` version 7 or above to make sure you can successfully install TensorFlow.

## 4) Data Preparation

### 4.1) Create Manifest

Athena accepts a textual manifest file as data set interface, which describes speech data set in csv format. In such file, each line contains necessary meta data (e.g. key, audio path, transcription) of a speech audio. For custom data, such manifest file needs to be prepared first. An example is shown as follows:

```csv
wav_filename	wav_length_ms	transcript
/dataset/train-clean-100-wav/374-180298-0000.wav	465004	chapter sixteen i might have told you of the beginning of this liaison in a few lines but i wanted you to see every step by which we came i to agree to whatever marguerite wished
/dataset/train-clean-100-wav/374-180298-0001.wav	514764	marguerite to be unable to live apart from me it was the day after the evening when she came to see me that i sent her manon lescaut from that time seeing that i could not change my mistress's life i changed my own
/dataset/train-clean-100-wav/374-180298-0002.wav	425484	i wished above all not to leave myself time to think over the position i had accepted for in spite of myself it was a great distress to me thus my life generally so calm
/dataset/train-clean-100-wav/374-180298-0003.wav	356044	assumed all at once an appearance of noise and disorder never believe however disinterested the love of a kept woman may be that it will cost one nothing
```

## 5) Training

### 5.1) Setting the Configuration File

All of our training/ inference configurations are written in config.json. Below is an example configuration file with comments to help you understand.

<details><summary>expand json</summary><div>

```json
{
  "batch_size":32,
  "num_epochs":20,
  "sorta_epoch":1,
  "ckpt":"examples/asr/hkust/ckpts/transformer",

  "solver_gpu":[0],
  "solver_config":{
    "clip_norm":100,
    "log_interval":10,
    "enable_tf_function":true
  },


  "model":"speech_transformer",
  "num_classes": null,
  "pretrained_model": null,
  "model_config":{
    "return_encoder_output":false,
    "num_filters":512,
    "d_model":512,
    "num_heads":8,
    "num_encoder_layers":12,
    "num_decoder_layers":6,
    "dff":1280,
    "rate":0.1,
    "label_smoothing_rate":0.0
  },

  "optimizer":"warmup_adam",
  "optimizer_config":{
    "d_model":512,
    "warmup_steps":8000,
    "k":0.5
  },

  "dataset_builder": "speech_recognition_dataset",
  "num_data_threads": 1,
  "trainset_config":{
    "data_csv": "examples/asr/hkust/data/train.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"},
    "input_length_range":[10, 8000]
  },
  "devset_config":{
    "data_csv": "examples/asr/hkust/data/dev.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"},
    "input_length_range":[10, 8000]
  },
  "testset_config":{
    "data_csv": "examples/asr/hkust/data/dev.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"}
  }
}
```

</div></details>

### 5.2) Train a Model

With all the above preparation done, training becomes straight-forward. `athena/main.py` is the entry point of the training module. Just run:
```
$ python athena/main.py <your_config_in_json_file>
````

Please install Horovod and MPI at first, if you want to train model using multi-gpu. See the [Horovod page](https://github.com/horovod/horovod) for more instructions.

To run on a machine with 4 GPUs with Athena:
```
$ horovodrun -np 4 -H localhost:4 python athena/horovod_main.py <your_config_in_json_file>
```

To run on 4 machines with 4 GPUs each with Athena:
```
$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python athena/horovod_main.py <your_config_in_json_file>
```

## 6) Deployment
After training, you can deploy the model on servers using the TensorFlow C++ API. Below are some steps to achieve this functionality with an ASR model. 

1. Install all dependencies, including TensorFlow, Protobuf, absl, Eigen3 and kenlm (optional).
2. Freeze the model to pb format with `athena/deploy_main.py`.
3. Compile the C++ codes.
4. Load the model and do argmax decoding in C++ codes, see `deploy/src/argmax.cpp` for the entry point.

After compiling, an executable file will be generated and you can run the executable file:
```
$ ./argmax
```

Detailed implementation is described [here](https://github.com/athena-team/athena/blob/master/deploy/README.md).

## 7) Results

### 7.1) ASR

Language  | Model Name | Training Data | Hours of Speech | Error Rate
:-----------: | :------------: | :----------: |  -------: | -------:
English  | Transformer | [LibriSpeech Dataset](http://www.openslr.org/12/) | 960 h |
English  | Transformer | [Switchboard Dataset](https://catalog.ldc.upenn.edu/LDC97S62) | 260h | 8.6% (WER) |
English  | Transformer | [TIMIT Dataset](https://catalog.ldc.upenn.edu/LDC93S1) | 3 h | 16.8% (PER) |
Mandarin | Transformer | HKUST Dataset | 151 h | 22.75% (CER)
Mandarin | Transformer | [AISHELL Dataset](http://www.openslr.org/33/) | 178 h | 6.6% (CER)

To compare with other published results, see [wer_are_we.md](https://github.com/athena-team/athena/blob/master/docs/wer_are_we.md).

## 8) Directory Structure

Below is the basic directory structure for Athena

```bash
|-- Athena
|   |-- data  # - root directory for input-related operations
|   |   |-- datasets  # custom datasets for ASR, TTS and pre-training
|   |-- layers  # some layers
|   |-- models  # some models
|   |-- tools # contains various tools, e.g. decoding tools
|   |-- transform # custom featureizer based on C++
|   |   |-- feats
|   |   |   |-- ops # c++ code on tensorflow ops
|   |-- utils # utils, e.g. checkpoit, learning_rate, metric, etc
|-- deploy  # deployment with Tensorflow C++
|   |-- include
|   |-- src
|-- docker
|-- docs  # docs
|-- examples  # example scripts for ASR, TTS, etc
|   |-- asr  # each subdirectory contains a data preparation scripts and a run script for the task
|   |   |-- aishell
|   |   |-- hkust
|   |   |-- librispeech
|   |   |-- switchboard
|   |   |-- timit
|   |-- translate # examples for translate
|   |   |-- spa-eng-example
|   |-- tts # examples for tts
|   |   |-- data_baker
|   |   |-- libritts
|   |   |-- ljspeech
|-- tools  # need to source env.sh before training
```
