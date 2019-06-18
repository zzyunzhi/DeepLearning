import tensorflow as tf


def conv2d_wn(inputs, n_filters, kernel_size, strides, scope, activation, initialized, shape=None):
    if shape is None:
        shape = inputs.get_shape().as_list()
    if not initialized:
        with tf.variable_scope(scope):
            weights = tf.get_variable(
                'weights',
                shape=[kernel_size[0], kernel_size[1], shape[3], n_filters],
                dtype=tf.float32,
            )
            weights = tf.nn.l2_normalize(weights.initialized_value(), [0, 1, 2])
            nn = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, strides[0], strides[1], 1],
                padding='SAME',
            )
            print('data based initialization...')
            nn_mean, nn_var = tf.nn.moments(nn, [0, 1, 2])
            g_init = 1/tf.sqrt(nn_var+1e-8)
            b_init = -nn_mean * g_init
            print('before getting variable')
            _ = tf.Variable(g_init, trainable=True, name='g', dtype=tf.float32)
            print('after getting g')
            _ = tf.Variable(b_init, trainable=True, name='b', dtype=tf.float32)
            print('after getting b')
            nn = tf.reshape(g_init, [1, 1, 1, n_filters]) * nn
            nn = tf.nn.bias_add(nn, b_init)

            if activation is not None:
                nn = activation(nn)
    else:
        with tf.variable_scope(scope, reuse=True):
            print('hitting initialized')
            weights = tf.get_variable('weights')
            g = tf.get_variable('g')
            b = tf.get_variable('b')
            tf.assert_variables_initialized([weights, g, b])
            weights = tf.reshape(g, [1, 1, 1, n_filters]) * tf.nn.l2_normalize(weights, [0, 1, 2])
            nn = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, strides[0], strides[1], tf.reshape(b, [1, 1, 1, n_filters])],
                padding='SAME',
            )
            nn = tf.nn.bias_add(nn, b)

            if activation is not None:
                nn = activation(nn)
    return nn


def conv2d_wn_transpose(inputs, n_filters, kernel_size, strides, scope, activation, initialized, shape):
    output_shape = [shape[0], strides[0]*shape[1], strides[1]*shape[2], n_filters]
    if not initialized:
        with tf.variable_scope(scope):
            weights = tf.get_variable(
                'weights',
                shape=[kernel_size[0], kernel_size[1], n_filters, shape[3]],
                dtype=tf.float32)
            weights = tf.nn.l2_normalize(weights.initialized_value(), [0, 1, 3])
            nn = tf.nn.conv2d_transpose(
                value=inputs,
                filter=weights,
                output_shape=output_shape,
                strides=[1, strides[0], strides[1], 1],
            )
            nn_mean, nn_var = tf.nn.moments(nn, [0, 1, 2])
            g_init = 1/tf.sqrt(nn_var+1e-8)
            b_init = -nn_mean * g_init
            _ = tf.Variable(g_init, trainable=True, name='g', dtype=tf.float32)
            _ = tf.Variable(b_init, trainable=True, name='b', dtype=tf.float32)
            nn = tf.reshape(g_init, [1, 1, 1, n_filters]) * (nn + tf.reshape(b_init, [1, 1, 1, n_filters]))

            if activation is not None:
                nn = activation(nn)
    else:
        with tf.variable_scope(scope, reuse=True):
            weights = tf.get_variable('weights')
            g = tf.get_variable('g')
            b = tf.get_variable('b')
            tf.assert_variables_initialized([weights, g, b])
            weights = tf.reshape(g, [1, 1, 1, n_filters]) * tf.nn.l2_normalize(weights, [0, 1, 3])
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


def gated_shortcut_connection(inputs, scope, initialized, weights_norm=True):
    with tf.variable_scope(scope):
        if weights_norm:
            nn_a = conv2d_wn(inputs, 32, (4, 4), (1, 1), 'nn_a', None, initialized)
            nn_b = conv2d_wn(inputs, 32, (4, 4), (1, 1), 'nn_b', tf.nn.sigmoid, initialized)
            print('built', nn_a.name)
        else:
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


def residual_stack(inputs, scope, initialized, weights_norm, n_layers=2):
    print('building residual stack...')
    nn = inputs
    with tf.variable_scope(scope):
        if weights_norm:
            for idx in range(n_layers):
                nn = tf.nn.relu(nn)
                nn = conv2d_wn(nn, 32, (3, 3), (1, 1), '{}_conv_relu_0'.format(idx), tf.nn.relu, initialized)
                nn = conv2d_wn(nn, 64, (3, 3), (1, 1), '{}_conv_1'.format(idx), tf.nn.relu, initialized)
                nn = gated_shortcut_connection(nn, 'gsc_{}'.format(idx), initialized, weights_norm=True)
        else:
            for idx in range(n_layers):
                nn = tf.nn.relu(nn)
                nn = tf.layers.conv2d(inputs=nn, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same', activation=tf.nn.relu)
                nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same')
                nn = gated_shortcut_connection(nn, 'gsc_{}'.format(idx), initialized, weights_norm=True)
        nn = tf.nn.relu(nn)
    return nn
