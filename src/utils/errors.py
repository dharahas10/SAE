import tensorflow as tf

from src.utils.tf_helper import nonzeroLikeTensor


def error_fn(test, predict):

    with tf.variable_scope('error'):

        predict_ = nonzeroLikeTensor(test, predict)
        abs_diff = tf.abs(tf.subtract(test, predict_))

        mse = tf.reduce_sum(tf.square(abs_diff))
        mae = tf.reduce_sum(abs_diff)

        # for testing nonzero count
        nonzero_count = tf.count_nonzero(predict_)

        error = {'mse': mse, 'mae': mae, 'nonzero_count': nonzero_count}

    return error
