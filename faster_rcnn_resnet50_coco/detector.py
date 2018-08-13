import numpy as np
import tensorflow as tf
import json
from json import encoder

from labels import LABEL_MAP

encoder.FLOAT_REPR = lambda o: format(o, '.2f')


class ObjectDetector(object):

    def __init__(self, graph_path):
        self.load(graph_path)

    def load(self, graph_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.graph = detection_graph
        self.sess = tf.Session(graph=self.graph)

        self.load_nodes()

    def load_nodes(self):
        ops = self.graph.get_operations()

        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = self.graph.get_tensor_by_name(
                        tensor_name)

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

    def _run_inference(self, image):
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        return output_dict

    def json_handler(self, image, output_dict):
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']

        d = {}
        l = []
        for i, v in enumerate(scores):
            if v > 0:
                aux = {}
                aux['x0'] = float("{0:.4f}".format(boxes[i][1]))
                aux['y0'] = float("{0:.4f}".format(boxes[i][0]))
                aux['x1'] = float("{0:.4f}".format(boxes[i][3]))
                aux['y1'] = float("{0:.4f}".format(boxes[i][2]))
                aux['object'] = LABEL_MAP[classes[i]]
                aux['probability'] = float("{0:.4f}".format(scores[i]))
                l.append(aux)

        d['objects'] = l
        return json.dumps(d)

    def run(self, image):
        return self.json_handler(image, self._run_inference(image))
