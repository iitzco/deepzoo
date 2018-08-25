# YOLO v2 trained on COCO

![img](imgs/result.png)

> Locate and classify all objects that are present in an image

<p align="left">
  <a href="https://github.com/iitzco/deepzoo/releases/download/model-upload-5/yolov2_coco.zip">
    <img src="../imgs/download-button.png" height=100/>
  </a>
</p>

## Requirements

1. Run `pip install -r requirements.txt`
1. Run `pip install git+https://github.com/thtrieu/darkflow`

## How to run

```python
from darkflow.net.build import TFNet
import cv2
import sys

# Path to downloaded .weights file
MODEL_PATH = /path/to/weights/file

# Path to configuration .cfg file
MODEL_CONFIG = /path/to/cfg/file

# Path to labels.txt downloaded file
LABELS = /path/to/labels

options = {"model": MODEL_PATH, "load": MODEL_CONFIG, "labels": LABELS}

tfnet = TFNet(options)

imgcv = cv2.imread(sys.argv[1])

# return_predict receives a [H, W, C] numpy image
result = tfnet.return_predict(imgcv)

# result is a map object
print(result)
```

Example output:

```python
[{
    'label': 'person',
    'confidence': 0.0,
    'topleft': {
        'x': 323,
        'y': 321
    
    },
    'bottomright': {
        'x': 489,
        'y': 670
    
    }

}, {
    'label': 'person',
    'confidence': 0.7112027,
    'topleft': {
        'x': 318,
        'y': 328
    
    },
    'bottomright': {
        'x': 501,
        'y': 719
    
    }

}, {
    'label': 'elephant',
    'confidence': 0.82942057,
    'topleft': {
        'x': 444,
        'y': 231
    
    },
    'bottomright': {
        'x': 719,
        'y': 433
    
    }
 ...
}]
```

## Customize

Deal with the returning `map` object to do whatever you want!

## Model info

* This is based on yolo's tensorflow port called [darflow](https://github.com/thtrieu/darkflow).
* Weight ando configuration files extracted from [YOLO official site](https://pjreddie.com/darknet/yolo/)
* The uploaded model is version 2. The newest version (v3) is still not supported by darkflow.
