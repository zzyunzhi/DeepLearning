import tensorflow as tf


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


