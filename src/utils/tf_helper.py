import tensorflow as tf


def corruptData(indices, values, shape, noiseRatio=0.1, name=None):
    if name is None:
        name = 'corrupt_input'

    with tf.variable_scope(name) as scope:
        rand_val = tf.random_uniform(tf.shape(values))
        rand_val_less_noise = tf.greater_equal(rand_val,
                                               tf.constant(noiseRatio, dtype=tf.float32))
        rand_val_noisy = tf.cast(rand_val_less_noise, tf.float32)

        noisy_values = tf.multiply(values, rand_val_noisy)
        noisy_X = tf.sparse_to_dense(indices,
                                     shape,
                                     noisy_values,
                                     name=scope.name)

    return noisy_X


def get_global_step(name=None):
    if name is None:
        name = 'temp'
    with tf.variable_scope(name):
        global_step_tensor = tf.get_variable('global_step_tensor',
                                             initializer=1,
                                             trainable=False)
    return global_step_tensor


def nonzeroLikeTensor(a, b):
    '''
        a= [[1 2 0  4]
            [0 1 0 4]]
        b= [[1 0 1  4]
            [0 1 0 4]]

        all zero indices in a are made 0 in b
        returns b
    '''

    nonzero_a = tf.not_equal(a, tf.constant(0, tf.float32))
    ones_a = tf.cast(nonzero_a, tf.float32)
    b_ = tf.multiply(b, ones_a)

    return b_
