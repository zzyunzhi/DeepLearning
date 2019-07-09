import tensorflow as tf
import numpy as np


def upsample_conv2d(inputs, filter_size, n_filters):
    nn = tf.concat([inputs, inputs, inputs, inputs], axis=-1)
    nn = tf.depth_to_space(nn, block_size=2)
    nn = tf.layers.conv2d(nn, filters=n_filters, kernel_size=filter_size, padding='same')
    return nn


def downsample_conv2d(inputs, filter_size, n_filters):
    nn = tf.space_to_depth(inputs, block_size=2)
    nn = tf.add_n(tf.split(nn, 4, axis=-1))/4
    nn = tf.layers.conv2d(nn, filters=n_filters, kernel_size=filter_size, padding='same')
    return nn


def res_block_up(inputs, filter_size, n_filters, scope='up_block'):
    with tf.variable_scope(scope):
        nn = tf.layers.batch_normalization(inputs)
        nn = tf.nn.relu(nn)
        nn = tf.layers.conv2d(nn, filters=n_filters, kernel_size=filter_size, padding='same')
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.relu(nn)
        residual = upsample_conv2d(nn, filter_size=filter_size, n_filters=n_filters)
        shortcut = upsample_conv2d(nn, filter_size=(1, 1), n_filters=n_filters)
        return shortcut + residual


def res_block_down(inputs, filter_size, n_filters, scope='down_block'):
    with tf.variable_scope(scope):
        nn = tf.layers.batch_normalization(inputs)
        nn = tf.nn.relu(nn)
        nn = tf.layers.conv2d(nn, filters=n_filters, kernel_size=filter_size, padding='same')
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.relu(nn)
        residual = downsample_conv2d(nn, filter_size=filter_size, n_filters=n_filters)
        shortcut = downsample_conv2d(nn, filter_size=(1, 1), n_filters=n_filters)
        return shortcut + residual


def res_block(inputs, filter_size, n_filters, scope='res_block'):
    with tf.variable_scope(scope):
        residual = tf.layers.conv2d(inputs, filters=n_filters, kernel_size=filter_size, padding='same')
        shortcut = tf.layers.conv2d(inputs, filters=n_filters, filter_size=(1, 1), padding='same')
        return shortcut + residual


class SNGAN(object):
    def __init__(self, img_size=(32, 32, 3)):
        self.learning_rate = 1e-3
        self.latent_size = 128
        self.inputs = tf.placeholder(tf.float32, (None,) + img_size, name='inputs')
        # self.loss, self.train_op = self.build_graph()

    def build_generator_graph(self, n_samples):
        with tf.variable_scope('generator'):
            z = tf.random_normal([n_samples, self.latent_size])
            nn = tf.layers.dense(z, 4*4*256)
            nn = tf.reshape(nn, [-1, 4, 4, 256])
            for idx in range(3):
                nn = res_block_up(nn, filter_size=(3, 3), n_filters=256,
                                scope='res_up_{}'.format(idx))
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)
            nn = tf.layers.conv2d(nn, filters=3, kernel_size=(3, 3), padding='same')
            nn = tf.nn.tanh(nn)
        return nn

    def build_discriminator_graph(self):
        with tf.variable_scope('discriminator'):
            nn = self.inputs
            for idx in range(2):
                nn = res_block_down(nn, filter_size=(3, 3), n_filters=256,
                                  scope='res_down_{}'.format(idx))
            for idx in range(2):
                nn = res_block(nn, filter_size=(3, 3), n_filters=256,
                              scope='res_{}'.format(idx))
            nn = tf.nn.relu(nn)
            nn = tf.nn.pool(nn, window_shape=(), pooling_type='AVG', padding='SAME') * 4
            nn = tf.layers.flatten(nn)
            nn = tf.layers.dense(nn, 1)
        return nn
