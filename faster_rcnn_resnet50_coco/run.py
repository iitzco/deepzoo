from detector import ObjectDetector
import sys
from PIL import Image
import numpy as np


img_path = sys.argv[1]
img = np.asarray(Image.open(img_path), dtype=np.uint8)

# Provide the .pb model file path
graph_path = "./faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"

model = ObjectDetector(graph_path)
out = model.run(img)

print(out)
