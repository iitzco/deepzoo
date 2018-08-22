# Deep Speech [EN]

> Convert speech audio to text
> **This work is just an example demonstration from [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech)**

![img](https://uploads-ssl.webflow.com/5985ca0c9abf440001d1f4b0/5a68a52180efb200017181cf_transcription_icon_v2_EN.png)

## [Link to model](https://github.com/iitzco/deepzoo/releases/download/model-upload-9/deepspeech-0.1.1-models.tar.gz)

## Limitations

* **This demo currently works only with mono 16kHZ `wav` files.**
* The provided model is only for English speech. Go to the original wiki [here](https://github.com/mozilla/DeepSpeech/wiki) to train other languages.

## Requirements

Run `pip install -r requirements.txt`

This will install `deepspeech`, `numpy` and `scipy`, which are it's only dependencies.

## How to run

Use `SpeechToText` class from `speech_to_text.py` file. 

The class can be used as shown in the following example:

```python
import sys
from speech_to_text import SpeechToText
import scipy.io.wavfile as wav

# This variable should point to models/
MODEL_PATH = "/path/to/downloaded/models"

# Remember, only WAV mono 16kHZ
AUDIO_PATH = "/path/to/wav/audio"

stt = SpeechToText(MODEL_PATH)
fs, audio = wav.read(AUDIO_PATH)

# Speech to text
text = stt.run(audio, fs)

print(text)
```

## Model info

Models were found in it's original repo [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

