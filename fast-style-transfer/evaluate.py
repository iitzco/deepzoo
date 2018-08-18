import sys
import transform
import numpy as np
import os
import tensorflow as tf
import cv2


def run(img, checkpoint_dir):
    sess = tf.Session()
    img = np.expand_dims(img, axis=0)
    img_placeholder = tf.placeholder(tf.float32, shape=img.shape, name='img_placeholder')
    preds = transform.net(img_placeholder)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_dir)

    _pred = sess.run(preds, feed_dict={img_placeholder: img})
    return _pred.squeeze()


if __name__ == '__main__':
    MODEL = "./models/wave.ckpt"
    OUTPUT = "./output"

    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = run(img, MODEL)
    img = np.clip(img, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
