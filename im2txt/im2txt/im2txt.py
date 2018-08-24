import math
import os
import sys
import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

class Im2Txt():
    def __init__(self, checkpoint_path, vocab_file):

        # Build the inference graph.
        self.g = tf.Graph()
        with self.g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)
        self.g.finalize()

        self.vocab = vocabulary.Vocabulary(vocab_file)
        self.sess = tf.Session(graph=self.g)
        restore_fn(self.sess)
        self.generator = caption_generator.CaptionGenerator(model, self.vocab)

    def run(self, image_path):
        with tf.gfile.GFile(image_path, "rb") as f:
            image = f.read()
            captions = self.generator.beam_search(self.sess, image)
            ret = []
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                ret.append((math.exp(caption.logprob), sentence))

            return ret

