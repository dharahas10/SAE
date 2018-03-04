import math

import tensorflow as tf


def get_layer_variables(inputs, shape,
                        weights_init='random_uniform',
                        activation_fn='tanh',
                        dropout=False,
                        keep_prob=1,
                        name=None):

    layer_name = "layer_"
    if name is not None:
        layer_name += name

    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as scope:

        local_weights = get_weights(shape, init_type=weights_init)
        local_bias = get_biases(shape)

        layer = tf.add(tf.matmul(inputs, local_weights), local_bias)

        if activation_fn == 'tanh':
            activated_layer = tf.tanh(layer, name=scope.name)
        elif activation_fn == 'sigmoid':
            activated_layer = tf.sigmoid(layer, name=scope.name)

        layer = activated_layer
        if dropout:
            layer = tf.nn.dropout(layer, keep_prob, name='dropout_layer')

    return layer, [local_weights, local_bias]


def get_weights(shape, init_type='random_uniform', name=None):

    if init_type == 'random_uniform':
        init_op = tf.random_uniform(shape,
                                    -1/math.sqrt(shape[0]),
                                    1/math.sqrt(shape[0]))

    elif init_type == 'random_normal':
        init_op = tf.random_normal(shape)

    elif init_type == 'truncated_normal':
        init_op = tf.truncated_normal(shape)

    var_name = 'weights'
    if name is not None:
        var_name += '_'+name

    return tf.get_variable(var_name,
                           dtype=tf.float32,
                           initializer=init_op)


def get_biases(shape, name=None):
    var_name = 'biases'
    if name is not None:
        var_name += '_'+name

    return tf.get_variable(var_name,
                           dtype=tf.float32,
                           initializer=tf.ones(shape[1], tf.float32),
                           trainable=True)
