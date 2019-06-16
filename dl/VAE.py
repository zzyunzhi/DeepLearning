import tensorflow as tf
import numpy as np


def gated_shortcut_connection(inputs, scope='gsc'):
    with tf.variable_scope(scope):
        a = tf.layers.dense(inputs, tf.shape(inputs)[1:])
        b = tf.layers.dense(inputs, tf.shape(inputs)[1:], activation=tf.nn.sigmoid)
        nn = tf.multiply(a, b)
    return nn


def residual_stack(inputs):
    nn = inputs
    with tf.variable_scope('residual_stack'):
        for idx in range(5):
            nn = tf.nn.relu(nn)
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', activation=tf.nn.relu)
            nn = tf.layers.conv2d(inputs=nn, filters=128, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same')
            nn = gated_shortcut_connection(nn, scope='gsc_{}'.format(idx))
    nn = tf.nn.relu(nn)
    return nn


class VAE(object):

    def __init__(
            self,
            x_size=(32, 32, 3)
    ):
        self.x_size = x_size

        with tf.variable_scope('vae'):
            self.x_sym = tf.placeholder(tf.float32, [None, self.x_size], name='x_sym')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            self.loss, self.op, self.x_encoded_mean = self.build_loss()
            with tf.variable_scope('decoder', reuse=True):
                self.z_sym = tf.placeholder(tf.float32, [None, 1], name='z_sym')
                self.z_decoded = self.decode_sym(self.z_sym).sample()

    def encode_sym(self, x_sym):
        nn = tf.layers.conv2d(inputs=x_sym, filters=128, kernel_size=(4, 4), strides=(2, 2),
                              padding='same', activation=tf.nn.relu, name='conv_relu_0')
        nn = tf.layers.conv2d(inputs=nn, filters=256, kernel_size=(4, 4), strides=(2, 2),
                              padding='same', activation=tf.nn.relu, name='conv_relu_1')
        nn = tf.layers.conv2d(inputs=nn, filters=256, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', name='conv_2')
        nn = residual_stack(nn)
        log = tf.layers.dense(nn, 1)
        scale = tf.layers.dense(nn, 1)
        return tf.distributions.Normal(log=log, scale=scale)

    def decode_sym(self, z_sym):
        nn = tf.layers.dense(z_sym, np.prod(self.x_size))
        nn = tf.reshape(nn, [-1, self.x_size[0], self.x_size[1], self.x_size[2]])
        nn = tf.layers.conv2d_transpose(inputs=z_sym, filters=256, kernel_size=(3, 3), strides=(1, 1),
                                        padding='same', name='conv_0')
        nn = residual_stack(nn)
        nn = tf.layers.conv2d_transpose(input=nn, filters=128, kernel_size=(4, 4), strides=(2, 2),
                                        padding='same', activation=tf.nn.relu, name='conv_relu_1')
        nn = tf.layers.conv2d_transpose(input=nn, filter=self.x_size[2], kernel_size=(4, 4), strides=(2, 2),
                                        padding='same', name='conv_2')
        log = tf.layers.dense(nn, self.x_size)
        scale_diag = tf.layers.dense(nn, self.x_size)
        return tf.distributions.MultivariateNormalDiag(log=log, scale_diag=scale_diag)

    def build_loss(self):
        x_encoded, x_encoded_mean = self.encode_sym(self.x_sym)
        with tf.variable_scope('decoder'):
            x_encoded_decoded = self.decode_sym(x_encoded.sample())
        prior = tf.distributions.Normal(loc=0, scale=1)
        kl = tf.distributions.kl_divergence(x_encoded, prior)
        entropy = -x_encoded_decoded.log_prob(self.x_sym, axis=[1, 2, 3])
        loss = tf.reduce_mean(entropy, axis=0) + kl
        op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, op, x_encoded_mean

    def train_step(self, batch, lr=1e-4):
        sess = tf.get_default_session()
        loss, _ = sess.run([self.loss, self.op], {
            self.x_sym: batch,
            self.lr: lr,
        })

    def encode(self, x):
        sess = tf.get_default_session()
        x_encoded = sess.run(self.x_encoded_mean, feed_dict={self.x_sym: x})
        return x_encoded

    def decode(self, z):
        sess = tf.get_default_session()
        z_decoded = sess.run(self.z_decoded, feed_dict={self.z_sym: z})
        return z_decoded
