import tensorflow as tf
import numpy as np
from utils import *


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
        nn = tf.layers.batch_normalization(inputs, axis=[1, 2, 3])
        nn = tf.nn.relu(nn)
        nn = tf.layers.conv2d(nn, filters=n_filters, kernel_size=filter_size, padding='same')
        nn = tf.layers.batch_normalization(nn, axis=[1, 2, 3])
        nn = tf.nn.relu(nn)
        residual = upsample_conv2d(nn, filter_size=filter_size, n_filters=n_filters)
        shortcut = upsample_conv2d(inputs, filter_size=(1, 1), n_filters=n_filters)
        return shortcut + residual


def res_block_down(inputs, filter_size, n_filters, scope='down_block'):
    with tf.variable_scope(scope):
        nn = tf.layers.batch_normalization(inputs)
        nn = tf.nn.relu(nn)
        nn = tf.layers.conv2d(nn, filters=n_filters, kernel_size=filter_size, padding='same')
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.relu(nn)
        residual = downsample_conv2d(nn, filter_size=filter_size, n_filters=n_filters)
        shortcut = downsample_conv2d(inputs, filter_size=(1, 1), n_filters=n_filters)
        return shortcut + residual


def res_block(inputs, filter_size, n_filters, scope='res_block'):
    with tf.variable_scope(scope):
        residual = tf.layers.conv2d(inputs, filters=n_filters, kernel_size=filter_size, padding='same')
        shortcut = inputs
        return shortcut + residual


class SNGAN(object):
    def __init__(
            self,
            lr=2e-4,
            beta1=.0,
            beta2=.9,
    ):

        self.img_size=(32, 32, 3)
        self.latent_size=128
        self.x = tf.placeholder(tf.float32, (None,) + self.img_size, name='x')
        self.z = tf.placeholder(tf.float32, (None, self.latent_size), name='z')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # D(x), discriminating real samples
        d_x = self.build_discriminator_graph(self.x, reuse=None)
        print(f'D(x) has shape {d_x.get_shape().as_list()}')
        # G(z)
        g_z = self.build_generator_graph(self.z, reuse=None)
        print(f'G(z) has shape {g_z.get_shape().as_list()}')
        # D(G(z)), discriminating fake samples
        d_g_z = self.build_discriminator_graph(g_z, reuse=True)
        print(f'D(G(z) has shape {d_g_z.get_shape().as_list()}')

        self.generated_images = tf.cast(tf.clip_by_value((g_z + 1.0) * 127.5, 0.0, 255.0), dtype=tf.int32)

        # build loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_x, labels=tf.ones_like(d_x)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g_z, labels=tf.zeros_like(d_g_z)))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g_z, labels=tf.ones_like(d_g_z)))

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps=10, decay_rate=0.99)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=beta1, beta2=beta2)
        d_gvs = optimizer.compute_gradients(self.d_loss, var_list=d_vars)
        g_gvs = optimizer.compute_gradients(self.g_loss, var_list=g_vars)
        self.d_opt = optimizer.apply_gradients(d_gvs)
        self.g_opt = optimizer.apply_gradients(g_gvs)

    def build_discriminator_graph(self, inputs, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            nn = inputs
            for idx in range(2):
                nn = res_block_down(nn, filter_size=(3, 3), n_filters=256,
                                    scope=f'res_down{idx}')
            for idx in range(2):
                nn = res_block(nn, filter_size=(3, 3), n_filters=256,
                               scope=f'res_{idx}')
            nn = tf.nn.relu(nn)
            nn = tf.nn.avg_pool(value=nn, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME') * 4
            nn = tf.layers.flatten(nn)
            nn = tf.layers.dense(nn, 1)
        return nn

    def build_generator_graph(self, inputs, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            nn = inputs
            nn = tf.layers.dense(nn, 4*4*256)
            nn = tf.reshape(nn, [-1, 4, 4, 256])
            for idx in range(3):
                nn = res_block_up(nn, filter_size=(3, 3), n_filters=256,
                                  scope=f'res_up_{idx}')
            nn = tf.layers.batch_normalization(nn, axis=[1, 2, 3])
            nn = tf.nn.relu(nn)
            nn = tf.layers.conv2d(nn, filters=3, kernel_size=(3, 3), padding='same')
            nn = tf.nn.tanh(nn)
        return nn

    def train_d_step(self, batch_x, batch_z):
        sess = tf.get_default_session()
        d_loss, _ = sess.run([self.d_loss, self.d_opt], feed_dict={self.x: batch_x, self.z: batch_z})
        self.global_step = self.global_step + 1
        return d_loss

    def train_g_step(self, batch_z):
        sess = tf.get_default_session()
        g_loss, _ = sess.run([self.g_loss, self.g_opt], feed_dict={self.z: batch_z})
        return g_loss

    def show_progress(self, n_samples=2, save_dir='./assets/SNGAN/'):
        z = np.random.normal(0, 1, [n_samples, self.latent_size])
        sess = tf.get_default_session()
        images = sess.run(self.generated_images, feed_dict={self.z: z})

        display_images(images, 1, len(images))
        if save_dir is not None:
            save_images(images, prefix='', save_dir=save_dir)

        # inception score
        # is_mean, is_std = compute_is(images)
        # return is_mean, is_std

    def display_meta_info(self):
        sess = tf.get_default_session()
        lr, gs = sess.run([self.learning_rate, self.global_step])
        print(f'lr = {lr} at global step {gs}')
