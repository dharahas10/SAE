import tensorflow as tf


def optimize_fn(config, loss, train_vars=None, global_step=None):

    optimizer_name = config['name'].lower()

    if optimizer_name == 'gradientdescentoptimizer':
        optimizer = gradientDescentOptimizer(config, global_step)
    elif optimizer_name == 'adamoptimizer':
        optimizer = adamOptimizer(config, global_step=global_step)
    else:
        print("===> ERROR:: Optimizer not found")

    return optimizer.minimize(loss,
                              global_step=global_step,
                              var_list=train_vars)


def gradientDescentOptimizer(config, global_step=None):

    learning_rate = config['learning_rate']

    if config['decay'] and global_step is not None:
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   config['decay_steps'],
                                                   config['decay_rate'],
                                                   staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer


def adamOptimizer(config, global_step=None):

    learning_rate = config['learning_rate']

    if config['decay'] and global_step is not None:
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   config['decay_steps'],
                                                   config['decay_rate'],
                                                   staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer
