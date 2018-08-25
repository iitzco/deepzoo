# SSD MobileNet trained on COCO

![img](imgs/result.png)

<p align="left">
  <a href="https://github.com/iitzco/deepzoo/releases/download/model-upload-1/ssd_mobilenet_v2_coco_2018_03_29.zip">
    <img src="../imgs/download-button.png" height=100/>
  </a>
</p>

## Requirements

* tensorflow
* numpy

Install them by running `pip install -r requirements.txt`

## How to run

Use `ObjectDetector` class from `detector.py`. 

The class can be used as shown in the following example:

```python
img_path = "/path/to/my/image"
img = np.asarray(Image.open(img_path), dtype=np.uint8)

# Provide the .pb model file path
graph_path = "/path/to/downloaded/model"

model = ObjectDetector(graph_path)
out = model.run(img)

print(out)
```

> *NOTE*: the `run` method receives a numpy array with shape [H, W, C]. If you use the Image library to open the image, remember to also install `pillow`.

Example output:

```json
{
  "objects": [
    {
      "x0": 0.6019,
      "y0": 0.5202,
      "x1": 0.8349,
      "y1": 0.7085,
      "object": "car",
      "probability": 0.8041
    },
    {
      "y0": 0.4706,
      "x0": 0.2178,
      "y1": 0.6827,
      "x1": 0.2551,
      "object": "person",
      "probability": 0.7793
    },
    {
      "y0": 0.5152,
      "x0": 0.8042,
      "y1": 0.9988,
      "x1": 0.959,
      "object": "person",
      "probability": 0.7651
    },
	...
  ]
}
```

## Customize

If the json format does not suit your needs, just implement other handlers!

## Model info

Provided by tensorflow in it's model zoo. Link [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
