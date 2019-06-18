import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from layers import *


class VAE(object):
    def __init__(self):
        self.x_size = [32, 32, 3]
        self.verbose = True
        # use weights normalization
        self.weights_norm = False
        self.initialized = False

        self.x_sym, self.init_batch_size, self.lr, self.z_sym = self.build_ph()
        self.build_graph()


    def build_ph(self):
        x_sym = tf.placeholder(tf.float32, [None, *self.x_size], name='x_sym')
        init_batch_size = tf.placeholder(tf.int32, [], name='init_batch_size')
        lr = tf.placeholder(tf.float32, [], name='lr')
        with tf.variable_scope('decoder'):
            z_sym = tf.placeholder(tf.float32, [None, 1], name='z_sym')
        return x_sym, init_batch_size, lr, z_sym

    def build_graph(self):
        self.loss, self.op, self.x_encoded_mean = self.build_loss()
        with tf.variable_scope('decoder', reuse=True):
            self.z_decoded_flatten = self.decode_sym(self.z_sym).sample()
            self.z_decoded = tf.reshape(self.z_decoded_flatten, [-1, np.prod(self.x_size)])

    def encode_sym(self, x_sym):
        print('before encoding sym', x_sym.get_shape().as_list())
        if self.weights_norm:
            nn = conv2d_wn(x_sym, 32, (4, 4), (2, 2), 'conv_relu_0', tf.nn.relu, self.initialized)
            nn = conv2d_wn(nn, 64, (4, 4), (2, 2), 'conv_relu_1', tf.nn.relu, self.initialized)
            nn = conv2d_wn(nn, 64, (3, 3), (1, 1), 'conv_2', None, self.initialized)
            nn = residual_stack(nn, 'encode_sym_res_stack', self.initialized, weights_norm=True)
        else:
            nn = tf.layers.conv2d(inputs=x_sym, filters=32, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation=tf.nn.relu, name='conv_relu_0')
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation=tf.nn.relu, name='conv_relu_1')
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='conv_2')
            nn = residual_stack(nn, 'encode_sym_res_stack', self.initialized, weights_norm=False)
        nn = tf.layers.flatten(nn)
        loc = tf.layers.dense(nn, 1, name='encode_sym_mean')
        scale = tf.exp(tf.layers.dense(nn, 1, name='encode_sym_std'))
        print('after encoding sym', loc.get_shape().as_list())
        return tfp.distributions.Normal(loc=loc, scale=scale), loc

    def decode_sym(self, z_sym):
        print('before decoding sym', z_sym.get_shape().as_list())
        nn = tf.reshape(z_sym, [-1, 1, 1, 1])
        if self.weights_norm:
            nn = conv2d_wn(nn, 64, (3, 3), (1, 1), 'conv_0', None, self.initialized)
            nn = residual_stack(nn, 'decode_sym_res_stack', self.initialized, weights_norm=True)
            nn = conv2d_wn_transpose(nn, 32, (4, 4), (2, 2), 'conv_relu_1', tf.nn.relu, self.initialized,
                                     shape=[self.init_batch_size, *nn.get_shape().as_list()[1:]])
            nn = conv2d_wn_transpose(nn, self.x_size[2], (4, 4), (2, 2), 'conv_2', None, self.initialized,
                                     shape=[self.init_batch_size, *nn.get_shape().as_list()[1:]])
        else:
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='conv_0')
            nn = residual_stack(nn, 'decode_sym_res_stack', self.initialized, weights_norm=False)
            nn = tf.layers.conv2d_transpose(inputs=nn, filters=32, kernel_size=(4, 4), strides=(2, 2),
                                            padding='same', activation=tf.nn.relu, name='conv_relu_1')
            nn = tf.layers.conv2d_transpose(inputs=nn, filters=self.x_size[2], kernel_size=(4, 4), strides=(2, 2),
                                            padding='same', name='conv_2')
        print('after conv layers...', nn.get_shape().as_list())
        nn = tf.layers.flatten(nn)
        loc = tf.layers.dense(nn, np.prod(self.x_size), name='decode_sym_mean')
        scale_diag = tf.exp(tf.layers.dense(nn, np.prod(self.x_size), name='decode_sym_std'))
        print('after decoding sym', loc.get_shape().as_list())
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    def build_loss(self):
        x_encoded, x_encoded_mean = self.encode_sym(self.x_sym)
        with tf.variable_scope('decoder'):
            x_encoded_decoded = self.decode_sym(x_encoded.sample())
        prior = tfp.distributions.Normal(loc=0., scale=1.)
        kl = tfp.distributions.kl_divergence(x_encoded, prior)
        kl = tf.squeeze(kl)
        x_flattened = tf.reshape(self.x_sym, [-1, np.prod(self.x_size)])
        entropy = -x_encoded_decoded.log_prob(x_flattened)

        if self.verbose:
            from utils import pprint
            entropy = pprint(entropy, "entropy")
            kl = pprint(kl, "kl")

        loss = tf.reduce_mean(entropy + kl)
        op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, op, x_encoded_mean

    def train_step(self, batch, lr=1e-4):
        print('calling train_step with self.initialized', self.initialized)
        feed_dict = {self.x_sym: batch, self.lr: lr}
        sess = tf.get_default_session()
        if not self.initialized:
            feed_dict[self.init_batch_size] = len(batch)
            init_op = tf.initializers.global_variables()
            _ = sess.run(init_op, feed_dict=feed_dict)
            _, loss = sess.run([self.op, self.loss], feed_dict=feed_dict)
            self.initialized = True
        else:
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
        sess = tf.get_default_session()
        z_decoded = sess.run(self.z_decoded, feed_dict={self.z_sym: z})
        return z_decoded
