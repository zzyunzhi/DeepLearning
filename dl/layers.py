import tensorflow as tf
import numpy as np


def conv2d_mask(
        inputs,
        n_filters,
        kernel_shape, # [kernel_height, kernel_width]
        mask_type, # None, "A" or "B"
        scope,
        activation=None,
        strides=(1, 1), # [column_wise_stride, row_wise_stride]
        reuse=None,
        verbose=False,
    ):
    with tf.variable_scope(scope, reuse=reuse):
        n_channels = int(inputs.get_shape()[-1])
        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        assert kernel_h % 2 == 1 and kernel_w % 2 == 1

        center_h = kernel_h // 2
        center_w = kernel_w // 2

        weights = tf.get_variable("weights", [kernel_h, kernel_w, n_channels, n_filters],
                                  tf.float32, tf.contrib.layers.xavier_initializer())

        mask = np.ones((kernel_h, kernel_w, n_channels, n_filters), dtype=np.float32)

        mask[center_h, center_w+1:, :, :] = 0.
        mask[center_h+1:, :, :, :] = 0.

        if mask_type == 'A':
            mask[center_h, center_w, :, :] = 0.
        else:
            assert mask_type == 'B'
        weights.assign(weights * tf.constant(mask, dtype=tf.float32))

        nn = tf.nn.conv2d(input=inputs, filter=weights,
                          strides=[1, stride_h, stride_w, 1], padding='SAME')
        biases = tf.get_variable("biases", [n_filters,], tf.float32, tf.zeros_initializer())
        nn = tf.nn.bias_add(nn, biases)
        if activation is not None:
            nn = activation(nn)

    return nn 


def mlp_wn(inputs, output_size, scope, activation, initialized):
    with tf.variable_scope(scope):
        if not initialized:
            weights = tf.get_variable(
                'weights',
                shape=[int(inputs.get_shape()[1]), output_size],
                dtype=tf.float32,
                initializer=tf.initializers.random_normal(0, 0.05),
                trainable=True,
            )
            weights = tf.nn.l2_normalize(weights.initialized_value(), [0])
            nn = tf.matmul(inputs, weights)
            nn_mean, nn_var = tf.nn.moments(nn, [0])
            g_init = 1/tf.sqrt(nn_var + 1e-10)
            b_init = -nn_mean * g_init
            _ = tf.get_variable(
                'g',
                dtype=tf.float32,
                initializer=g_init,
                trainable=True,
            )
            print('after getting g')
            _ = tf.get_variable(
                'b',
                dtype=tf.float32,
                initializer=b_init,
                trainable=True,
            )
            print('after gettting b')
            nn = tf.reshape(g_init, [1, output_size]) * nn
            nn = tf.nn.bias_add(nn, b_init)
        else:
            print('hitting initialized')
            weights = tf.get_variable('weights')
            g = tf.get_variable('g')
            b = tf.get_variable('b')
            tf.assert_variables_initialized([weights, g, b])
            weights = weights * (g / tf.nn.l2_normalize(weights, [0]))[None]
            nn = tf.matmul(inputs, weights)
            nn = tf.nn.bias_add(nn, b)

        if activation is not None:
            nn = activation(nn)
        return nn


def conv2d_wn(inputs, n_filters, kernel_size, strides, scope, activation, initialized, shape=None):
    if shape is None:
        shape = inputs.get_shape().as_list()
    with tf.variable_scope(scope):
        if not initialized:
            weights = tf.get_variable(
                'weights',
                shape=[kernel_size[0], kernel_size[1], shape[3], n_filters],
                dtype=tf.float32,
                initializer=tf.initializers.random_normal(0, 0.05),
                trainable=True,
            )
            weights = tf.nn.l2_normalize(weights.initialized_value(), [0, 1, 2])
            nn = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, strides[0], strides[1], 1],
                padding='SAME',
            )
            nn_mean, nn_var = tf.nn.moments(nn, [0, 1, 2])
            g_init = 1/tf.sqrt(nn_var+1e-8)
            b_init = -nn_mean * g_init
            print('before getting variable')
            _ = tf.get_variable(
                'g',
                dtype=tf.float32,
                initializer=g_init,
                trainable=True,
            )
            print('after getting g')
            _ = tf.get_variable(
                'b',
                dtype=tf.float32,
                initializer=b_init,
                trainable=True,
            )
            print('after getting b')
            nn = tf.reshape(g_init, [1, 1, 1, n_filters]) * nn
            nn = tf.nn.bias_add(nn, b_init)

            if activation is not None:
                nn = activation(nn)
        else:
            print('hitting initialized')
            weights = tf.get_variable('weights')
            g = tf.get_variable('g')
            b = tf.get_variable('b')
            tf.assert_variables_initialized([weights, g, b])
            weights = tf.reshape(g, [1, 1, 1, n_filters]) / tf.nn.l2_normalize(weights, [0, 1, 2])
            nn = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, strides[0], strides[1], 1],
                padding='SAME',
            )
            nn = tf.nn.bias_add(nn, b)

            if activation is not None:
                nn = activation(nn)
    return nn


def conv2d_wn_transpose(inputs, n_filters, kernel_size, strides, scope, activation, initialized, shape):
    output_shape = [shape[0], strides[0]*shape[1], strides[1]*shape[2], n_filters]
    with tf.variable_scope(scope):
        if not initialized:
            weights = tf.get_variable(
                'weights',
                shape=[kernel_size[0], kernel_size[1], n_filters, shape[3]],
                dtype=tf.float32,
                initializer=tf.initializers.random_normal(0, 0.05),
                trainable=True
            )
            weights = tf.nn.l2_normalize(weights.initialized_value(), [0, 1, 3])
            nn = tf.nn.conv2d_transpose(
                value=inputs,
                filter=weights,
                output_shape=output_shape,
                strides=[1, strides[0], strides[1], 1],
                padding='SAME',
            )
            nn_mean, nn_var = tf.nn.moments(nn, [0, 1, 2])
            g_init = 1/tf.sqrt(nn_var+1e-8)
            b_init = -nn_mean * g_init
            _ = tf.Variable(g_init, trainable=True, name='g', dtype=tf.float32)
            _ = tf.Variable(b_init, trainable=True, name='b', dtype=tf.float32)
            print('before getting variable')
            _ = tf.get_variable(
                'g',
                dtype=tf.float32,
                initializer=g_init,
                trainable=True,
            )
            print('after getting g')
            _ = tf.get_variable(
                'b',
                dtype=tf.float32,
                initializer=b_init,
                trainable=True,
            )
            nn = tf.reshape(g_init, [1, 1, 1, n_filters]) * (nn + tf.reshape(b_init, [1, 1, 1, n_filters]))

            if activation is not None:
                nn = activation(nn)
        else:
            weights = tf.get_variable('weights')
            g = tf.get_variable('g')
            b = tf.get_variable('b')
            tf.assert_variables_initialized([weights, g, b])
            weights = tf.reshape(g, [1, 1, 1, n_filters]) / tf.nn.l2_normalize(weights, [0, 1, 3])
            nn = tf.nn.conv2d_transpose(
                value=inputs,
                filter=weights,
                output_shape=output_shape,
                strides=[1, strides[0], strides[1], 1],
            )
            nn = tf.nn.bias_add(nn, b)

            if activation is not None:
                nn = activation(nn)
    return nn


