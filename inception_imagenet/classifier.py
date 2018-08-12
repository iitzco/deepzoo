import argparse
import re
import sys

import numpy as np
import tensorflow as tf
import json


class Classifier():

    def __init__(self, graph_path, label_map):
        classification_graph = tf.Graph()
        with classification_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.classification_graph = classification_graph
        self.softmax_tensor = self.classification_graph.get_tensor_by_name('softmax:0')

        self.sess = tf.Session(graph=self.classification_graph)

        with open(label_map, "r") as f:
            self.name_map = json.load(f)

    def run(self, img_path):
        img = tf.gfile.FastGFile(img_path, 'rb').read()
        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': img})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        ret = {}

        for node_id in top_k:
            human_string = self.name_map.get(str(node_id), "")
            score = predictions[node_id]
            ret[human_string] = score

        return ret
