import tensorflow as tf

from src.utils.tf_helper import nonzeroLikeTensor


def loss_fn(correct,
            predict,
            noisy_input=None,
            isNoisy=False,
            noisyConfig=None,
            l2_reg=False,
            l2_beta=0,
            trainable_vars=None,
            l2_loss=0):

    if isNoisy:
        loss = get_noisy_loss(correct, predict, noisy_input, noisyConfig)
    else:
        loss = get_loss(correct, predict)

    if l2_reg and trainable_vars is not None:
        l2_loss += get_l2_loss(trainable_vars, l2_beta)

    loss = loss + l2_loss

    return loss


def get_loss(correct, predict):
    with tf.variable_scope('loss') as scope:
        predict_ = nonzeroLikeTensor(correct, predict)
        cross_entropy = tf.squared_difference(predict_, correct)
        cost = tf.reduce_sum(cross_entropy)

        loss = tf.truediv(cost,
                          tf.count_nonzero(correct, dtype=tf.float32),
                          name=scope.name)

    return loss


def get_noisy_loss(correct, predict, noisy_input, noisyConfig):

    assert noisyConfig is not None, "ERROR: noisyConfig is None"

    alpha = noisyConfig['alpha']
    beta = noisyConfig['beta']

    with tf.variable_scope('noisy_loss'):
        # beta cost
        predict_beta = nonzeroLikeTensor(noisy_input, predict)
        cross_entropy_beta = tf.squared_difference(predict_beta, noisy_input)
        cost_beta = tf.reduce_sum(cross_entropy_beta, name='beta_cost')
        loss_beta = tf.truediv(cost_beta,
                               tf.count_nonzero(noisy_input, dtype=tf.float32),
                               name="beta_loss")

        # alpha cost
        alpha_input = tf.subtract(correct, noisy_input)
        predict_alpha = nonzeroLikeTensor(alpha_input, predict)
        cross_entropy_alpha = tf.squared_difference(predict_alpha, alpha_input)
        cost_alpha = tf.reduce_sum(cross_entropy_alpha, name='alpha_cost')
        loss_alpha = tf.truediv(cost_alpha,
                                tf.count_nonzero(
                                    alpha_input, dtype=tf.float32),
                                name='alpha_loss')

        loss = tf.add(tf.multiply(alpha, loss_alpha),
                      tf.multiply(beta, loss_beta),
                      name='loss')

    return loss


def get_l2_loss(alist, beta=0.5):
    l2_loss = 0
    for var in alist:
        l2_loss += tf.nn.l2_loss(var)

    return beta*l2_loss
