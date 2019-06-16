import tensorflow as tf


def conv2d_wn(inputs, n_filters, kernel_size, strides, scope, activation=None, init=False, transpose=False):
    with tf.variable_scope(scope):
        if not transpose:
            weights = tf.get_variable(
                'filter',
                shape=[kernel_size[0], kernel_size[1], int(inputs.get_shape()[3]), n_filters],
                dtype=tf.float32)
            weights = tf.nn.l2_normalize(weights, [0, 1, 2])
            nn = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, strides[0], strides[1], 1],
                padding='SAME',
            )
        else:
            weights = tf.get_variable(
                'filter',
                shape=[kernel_size[0], kernel_size[1], n_filters, int(inputs.get_shape()[3])],
                dtype=tf.float32)
            weights = tf.nn.l2_normalize(weights, [0, 1, 3])

            shape = tf.shape(inputs)
            output_shape = tf.stack([shape[0], shape[1]*strides[0], shape[2]*strides[1], shape[3]])
            nn = tf.nn.conv2d_transpose(
                value=inputs,
                filter=weights,
                output_shape=output_shape,
                strides=strides,
            )

        # data dependent weights normalization
        if init:
            nn_mean, nn_var = tf.nn.moments(nn, [0, 1, 2])
            g_initializer = tf.reciprocal(tf.sqrt(nn_var+1e-8))
            b_initializer = -nn_mean * g_initializer
            print(g_initializer.get_shape())
            print(b_initializer.get_shape())
            g = tf.get_variable('g', dtype=tf.float32, initializer=g_initializer, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=b_initializer, trainable=True)
        else:
            g = tf.get_variable('g')
            b = tf.get_variable('b')
            tf.assert_variables_initialized([g, b])

        nn = tf.reshape(g, [1, 1, 1, n_filters]) * (nn + tf.reshape(b, [1, 1, 1, n_filters]))

        if activation is not None:
            nn = activation(nn)
    return nn


def gated_shortcut_connection(inputs, scope='gsc', weights_norm=True):
    with tf.variable_scope(scope):
        if weights_norm:
            nn_a = conv2d_wn(inputs, 64, (4, 4), (1, 1), 'nn_a', None, True)
            nn_b = conv2d_wn(inputs, 64, (4, 4), (1, 1), 'nn_b', tf.nn.sigmoid, True)
        else:
            nn_a = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(4, 4), strides=(1, 1),
                                    padding='same')
            nn_b = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(4, 4), strides=(1, 1),
                                    padding='same', activation=tf.nn.sigmoid)
        b = tf.get_variable('b', shape=[64], dtype=tf.float32)
        nn_a = tf.nn.bias_add(nn_a, b)
        c = tf.get_variable('c', shape=[64], dtype=tf.float32)
        nn_b = tf.nn.bias_add(nn_b, c)
        nn = nn_a * tf.nn.sigmoid(nn_b)
        print(nn.get_shape())
    return nn


def residual_stack(inputs):
    nn = inputs
    print(nn.get_shape())
    with tf.variable_scope('residual_stack'):
        for idx in range(5):
            nn = tf.nn.relu(nn)
            nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', activation=tf.nn.relu)
            nn = tf.layers.conv2d(inputs=nn, filters=128, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same')
            nn = gated_shortcut_connection(nn, scope='gsc_{}'.format(idx))
    nn = tf.nn.relu(nn)
    print(nn.get_shape())
    return nn
