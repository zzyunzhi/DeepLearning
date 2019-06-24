import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from layers import conv2d_mask
from utils import *
from datetime import datetime
import matplotlib.pyplot as plt


def residual_block(layer_in, hidden_dim, scope, reuse=None):
    assert hidden_dim % 2 == 0
    with tf.variable_scope(scope, reuse=reuse):
        nn = conv2d_mask(layer_in, n_filters=hidden_dim//2, kernel_shape=[1, 1], mask_type="B",
                    scope="residual_block_0", activation=tf.nn.relu)
        nn = conv2d_mask(nn, n_filters=hidden_dim//2, kernel_shape=[3, 3], mask_type="B",
                    scope="residual_block_1", activation=tf.nn.relu)
        nn = conv2d_mask(nn, n_filters=hidden_dim, kernel_shape=[1, 1], mask_type="B",
                    scope="residual_block_2", activation=tf.nn.relu)
        nn = tf.add(layer_in, nn)
        layer_out = tf.contrib.layers.layer_norm(nn, scope="residual_block_out_normalized")
    return layer_out


class PixelCNN(object):
    def __init__(
            self,
            img_size,
            color_dim,
            visualize_batch,
    ):
        
        self.height, self.width, self.n_channels = img_size
        self.color_dim = color_dim
        self.visualize_batch = visualize_batch
        self.learning_rate = 1e-3
        self.grad_clip = 1
        self.hidden_dim = 16
        self.out_hidden_dim = 16
        self.n_residual_blocks = 12

        print('building training graph...')
        self.inputs = tf.placeholder(tf.float32, (None,) + img_size, name="inputs")
        self.loss, self.train_op, logits = self.build_graph()

        print('building inference graph...')
        with tf.variable_scope('inference'):
            probs = tf.nn.softmax(logits, axis=-1)
            self.samples_det = tf.argmax(probs, axis=-1)
            samples_flattened = tf.random.categorical(logits=tf.reshape(logits, [-1, self.color_dim]), num_samples=1)
            self.samples = tf.reshape(samples_flattened, [-1, self.height, self.width, self.n_channels])

    def build_graph(self):
        nn = conv2d_mask(
            self.inputs, n_filters=self.hidden_dim, kernel_shape=[7, 7], mask_type="A",
            scope="conv_0",
        )
        for idx in range(self.n_residual_blocks):
            nn = residual_block(nn, self.hidden_dim, "residual_block_{}".format(idx))
        for idx in range(2):
            nn = conv2d_mask(nn, n_filters=self.out_hidden_dim, kernel_shape=[1, 1], mask_type="B",
                             scope="out_recurr_{}".format(idx), activation=tf.nn.relu)

        '''
        labels = self.inputs = [None, 28, 28, 3] -> [None, 28, 28, 3, 4] (one-hot encoded)
        logits = self.logits = [None, 28, 28, 3*4] -> [None, 28, 28, 3, 4] (reshaped)
        '''
        logits = conv2d_mask(nn, n_filters=self.n_channels * self.color_dim, kernel_shape=[1, 1], mask_type="B",
                             scope="logits")
        logits = tf.reshape(logits, [-1, self.height, self.width, self.n_channels, self.color_dim])
        inputs_one_hot = tf.one_hot(tf.cast(self.inputs, tf.int32), self.color_dim, axis=-1,
                                    name='inputs_one_hot')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=inputs_one_hot, logits=logits, name='loss',
        ))

        train_op = tf.train.RMSPropOptimizer(self.learning_rate)
        grads_vars = train_op.compute_gradients(loss)
        grads_vars_clipped = [
            (tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in grads_vars
        ]
        train_op = train_op.apply_gradients(grads_vars_clipped)
        return loss, train_op, logits

    def train_step(self, batch):
        sess = tf.get_default_session()
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.inputs: batch})
        return loss

    def test_step(self, batch):
        sess = tf.get_default_session()
        loss = sess.run(self.loss, feed_dict={self.inputs: batch})
        return loss

    def extra_step(self, save_dir=None, prefix=''):
        images = self.reconstruct_images(self.visualize_batch)
        display_images(images/self.color_dim, 1, len(images))
        if save_dir is not None:
            save_images(images/self.color_dim, prefix, save_dir)

    def reconstruct_images(self, batch):
        sess = tf.get_default_session()
        images = sess.run(self.samples_det, feed_dict={self.inputs: batch})
        return images

    def generate_images(self, n_samples):
        sess = tf.get_default_session()
        if n_samples == 1:
            run_ph = self.samples_det
        else:
            run_ph = self.samples
        images = np.zeros((n_samples, self.height, self.width, self.n_channels), dtype='float32')
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.n_channels):
                    images[:, i, j, k] = sess.run(
                        run_ph,
                        feed_dict={self.inputs: images}
                    )[:, i, j, k]
        return images

