# Adapted from https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py

import cv2
import numpy as np
import sys
import random

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


class PoseDetector():

    def __init__(self, proto_path, model_path):
        self.SIZE = 368
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def run(self, img, thresh=0.1):
        img_h, img_w, _ = img.shape
        inp = cv2.dnn.blobFromImage(img, 1.0 / 255, (self.SIZE, self.SIZE), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (img_w * point[0]) / out.shape[3]
            y = (img_h * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > thresh else None)

        return points
