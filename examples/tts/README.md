# Examples for TTS Tasks

Currently supported TTS tasks are LJSpeech and Chinese Standard Mandarin Speech Copus(data baker). An example script for Libritts will be released shortly. 

Currently Tacotron2 is supported for training and synthesis. Transformer and Fastspeech based model will be released shortly.

To perform the full procedure of TTS experiments, simply run:
```bash
source tools/env.sh
bash examples/tts/$task_name/run.sh
```

## A complete run of TTS experiment can be broken down into following stages:

1) Data preparation: you can run `examples/tts/$task_name/run.sh`, it will download the corresponding dataset and store it in `examples/tts/$task_name/data`. The script `examples/tts/$task_name/local/prepare_data.py` would generate the desired csv file decripting the dataset

2) Data normalization: With the generated csv file, we should compute the cmvn file firstly like this `python athena/cmvn_main.py examples/tts/$task_name/configs/t2.json examples/tts/$task_name/data/train.csv`. We can also directly run the training command like this `python athena/main.py examples/tts/$task_name/configs/t2.json`. It will automatically compute the cmvn file before training if the cmvn file does not exist.

4) Acoustic model training: You can train a Tacotron2 model using json file `examples/tts/$task_name/configs/t2.json`

6) Synthesis: Currently, we provide a simple synthesis process using GriffinLim vocoder. To test your training performance, run `python athena/synthesize_main.py examples/tts/$task_name/t2.json`
