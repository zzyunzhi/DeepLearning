import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utils import pprint


def gated_shortcut_connection(inputs, scope):
    with tf.variable_scope(scope):
        nn_a = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(4, 4), strides=(1, 1),
                                padding='same')
        nn_b = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(4, 4), strides=(1, 1),
                                padding='same', activation=tf.nn.sigmoid)
        b = tf.get_variable('b', shape=[32], dtype=tf.float32)
        nn_a = tf.nn.bias_add(nn_a, b)
        c = tf.get_variable('c', shape=[32], dtype=tf.float32)
        nn_b = tf.nn.bias_add(nn_b, c)
        nn = nn_a * tf.nn.sigmoid(nn_b)
    return nn


def residual_stack(inputs, scope, n_blocks=2):
    print('building residual stack...')
    nn = inputs
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            nn = tf.nn.relu(nn)
            nn = tf.layers.conv2d(inputs=nn, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', activation=tf.nn.relu)
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same')
            nn = gated_shortcut_connection(nn, 'gsc_{}'.format(idx))
        nn = tf.nn.relu(nn)
    return nn


class VAE(object):
    def __init__(self):
        self.x_size = [32, 32, 3]
        self.z_size = 2
        self.verbose = False
        # use weights normalization
        self.weights_norm = False

        self.x_sym, self.lr, self.z_sym = self.build_ph()
        self.loss, self.op, self.x_encoded_mean = self.build_loss()
        self.z_decoded = self.decode_sym(self.z_sym, reuse=True).probs

    def build_ph(self):
        x_sym = tf.placeholder(tf.float32, [None, *self.x_size], name='x_sym')
        lr = tf.placeholder(tf.float32, [], name='lr')
        with tf.variable_scope('decoder'):
            z_sym = tf.placeholder(tf.float32, [None, self.z_size], name='z_sym')
        return x_sym, lr, z_sym

    def encode_sym(self, x_sym, reuse=None, scope='encode_sym'):
        with tf.variable_scope(scope, reuse=reuse):
            print('before encoding sym', x_sym.get_shape().as_list())
            nn = tf.layers.conv2d(inputs=x_sym, filters=32, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation=tf.nn.relu, name='conv_relu_0')
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation=tf.nn.relu, name='conv_relu_1')
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='conv_2')
            nn = residual_stack(nn, 'res_stack')
            nn = tf.layers.dense(tf.layers.flatten(nn), self.z_size*2, name='dense_3')
            loc, scale_diag = tf.split(nn, 2, axis=-1)
            scale_diag = tf.nn.softplus(scale_diag) + 1e-6
            print('after encoding sym', loc.get_shape().as_list())
            assert int(loc.get_shape()[1]) == self.z_size
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag), loc

    def decode_sym(self, z_sym, reuse=None, scope='decode_sym'):
        with tf.variable_scope(scope, reuse=reuse):
            print('before decoding sym', z_sym.get_shape().as_list())
            nn = tf.layers.dense(z_sym, 128, name='dense_0')
            nn = tf.reshape(nn, [-1, 8, 8, 2])
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='conv_0')
            nn = residual_stack(nn, 'res_stack')
            nn = tf.layers.conv2d_transpose(inputs=nn, filters=32, kernel_size=(4, 4), strides=(2, 2),
                                            padding='same', activation=tf.nn.relu, name='conv_relu_1')
            nn = tf.layers.conv2d_transpose(inputs=nn, filters=3, kernel_size=(4, 4), strides=(2, 2),
                                            padding='same', name='conv_2')
            nn = tf.nn.tanh(nn)
            nn = tf.clip_by_value(nn, 1e-8, 1-1e-8)
            nn = pprint(nn)
        return tfp.distributions.Multinomial(total_count=255., logits=nn)
        # return tfp.distributions.Bernoulli(logits=nn)

    def build_loss(self):
        x_encoded, x_encoded_mean = self.encode_sym(self.x_sym)
        x_encoded_decoded = self.decode_sym(x_encoded.sample())
        prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.z_size), scale_diag=tf.ones(self.z_size))
        kl = tfp.distributions.kl_divergence(x_encoded, prior)
        kl = tf.squeeze(kl)
        log_prob = x_encoded_decoded.log_prob(self.x_sym)
        log_prob = pprint(log_prob)
        #entropy = -tf.reduce_sum(x_encoded_decoded.log_prob(self.x_sym), axis=[1, 2, 3])
        entropy = -tf.reduce_sum(log_prob, axis=[1, 2, 3])

        if self.verbose:
            entropy = pprint(entropy, "entropy")
            kl = pprint(kl, "kl")

        loss = tf.reduce_mean(entropy + kl)
        op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, op, x_encoded_mean

    def train_step(self, batch, lr=1e-4):
        feed_dict = {self.x_sym: batch, self.lr: lr}
        sess = tf.get_default_session()
        loss, _ = sess.run([self.loss, self.op], feed_dict=feed_dict)
        return loss

    def test_step(self, batch):
        sess = tf.get_default_session()
        loss = sess.run(self.loss, feed_dict={
            self.x_sym: batch,
        })
        return loss

    def encode(self, x):
        sess = tf.get_default_session()
        x_encoded = sess.run(self.x_encoded_mean, feed_dict={self.x_sym: x})
        return x_encoded

    def decode(self, z):
        assert z.shape[1] == self.z_size
        sess = tf.get_default_session()
        z_decoded = sess.run(self.z_decoded, feed_dict={self.z_sym: z})
        return z_decoded
