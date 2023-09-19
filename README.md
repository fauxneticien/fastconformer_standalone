# fastconformer_standalone

This is a development repository to extract the NeMo FastConformer model class out of the NeMo repository (which has a lot of dependencies) in order to get acquainted with the model at a low level and also give us a clean, dependency-free PyTorch model class for subsequent adaptation/experimentation with the model.

## Setup

```
# Create new conda environment in ./env
conda create -y --prefix ./env python=3.10 --no-default-packages

# Activate envionrment
conda activate ./env

# Install (minimal) dependencies
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install Cython && pip install gdown jupyterlab nemo_toolkit['all']

# Download checkpoint from HF into tmp/ folder
wget https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large/resolve/main/stt_en_fastconformer_ctc_large.nemo -P tmp/

# Download test file into tmp/ folder
gdown 15HIXKBA_0AeZOnaV9h0lciYs_I8VOaem -O tmp/cat.wav

# Test that we can transcribe tmp/cat.wav (should output 'cat')
python test-nemo-mwe.py
```

## Usage

### Initial (seemingly) working model class

```python
# See code in test-standalone.py
import torch
import torchaudio
from FastConformer.model import FastConformer

fc = FastConformer(num_labels=1024)
# Load weights unpacked from https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large/blob/main/stt_en_fastconformer_ctc_large.nemo
fc.load_pretrained_weights("tmp/nemo_unpacked/model_weights.ckpt", from_nemo=True)
fc.to('cuda')

samples, sample_rate = torchaudio.load("tmp/cat.wav")

audio_samples = samples
audio_lens    = torch.tensor(samples.shape[1]).unsqueeze(0)

standalone_log_probs = fc(audio_samples.to('cuda'), audio_lens.to('cuda'))
```