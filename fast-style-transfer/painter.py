import sys
import numpy as np
import os
import tensorflow as tf
import cv2


class PainterNetwork:

    def _net(image):
        conv1 = PainterNetwork._conv_layer(image, 32, 9, 1)
        conv2 = PainterNetwork._conv_layer(conv1, 64, 3, 2)
        conv3 = PainterNetwork._conv_layer(conv2, 128, 3, 2)
        resid1 = PainterNetwork._residual_block(conv3, 3)
        resid2 = PainterNetwork._residual_block(resid1, 3)
        resid3 = PainterNetwork._residual_block(resid2, 3)
        resid4 = PainterNetwork._residual_block(resid3, 3)
        resid5 = PainterNetwork._residual_block(resid4, 3)
        conv_t1 = PainterNetwork._conv_tranpose_layer(resid5, 64, 3, 2)
        conv_t2 = PainterNetwork._conv_tranpose_layer(conv_t1, 32, 3, 2)
        conv_t3 = PainterNetwork._conv_layer(conv_t2, 3, 9, 1, relu=False)
        preds = tf.nn.tanh(conv_t3) * 150 + 255./2
        return preds

    def _conv_layer(net, num_filters, filter_size, strides, relu=True):
        weights_init = PainterNetwork._conv_init_vars(net, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = PainterNetwork._instance_norm(net)
        if relu:
            net = tf.nn.relu(net)

        return net

    def _conv_tranpose_layer(net, num_filters, filter_size, strides):
        weights_init = PainterNetwork._conv_init_vars(net, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = PainterNetwork._instance_norm(net)
        return tf.nn.relu(net)

    def _residual_block(net, filter_size=3):
        tmp = PainterNetwork._conv_layer(net, 128, filter_size, 1)
        return net + PainterNetwork._conv_layer(tmp, 128, filter_size, 1, relu=False)

    def _instance_norm(net, train=True):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift

    def _conv_init_vars(net, out_channels, filter_size, transpose=False):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]

        weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32)
        return weights_init


    def run(img, checkpoint_dir):
        sess = tf.Session()
        img = np.expand_dims(img, axis=0)
        img_placeholder = tf.placeholder(tf.float32, shape=img.shape, name='img_placeholder')
        preds = PainterNetwork._net(img_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)

        _pred = sess.run(preds, feed_dict={img_placeholder: img})
        return _pred.squeeze()


if __name__ == '__main__':
    MODEL = "./models/wave.ckpt"
    OUTPUT = "./output"

    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = PainterNetwork.run(img, MODEL)
    img = np.clip(img, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
