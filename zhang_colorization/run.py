import numpy as np
import os
import cv2
import skimage.color as color
import matplotlib.pyplot as plt
import sys

from colorant import Colorant

img = cv2.imread(sys.argv[1])

# Specify the paths for the model files 
PROTO = "./zhang_colorization/colorization_deploy_v2.prototxt"
MODEL = "./zhang_colorization/colorization_release_v2.caffemodel"
HULL = "./zhang_colorization/pts_in_hull.npy"

c = Colorant(PROTO, MODEL, HULL)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = c.run(img)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.png", img)
