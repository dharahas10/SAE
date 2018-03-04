import os

import tensorflow as tf


def loadCheckpointSaver(path, name, config, saver, sess):

    folder = generate_path(path, config)

    if not os.path.isdir(folder):
        print('\n\nWARN:: Checkpoint directory not found')
    else:
        print('\n------------------------------------------')
        print('Restoring Checkpoint Saver ....')
        saver.restore(sess, os.path.join(folder, name))
        # saver.restore(sess, folder)
        print('Checkpoint Successfully Loaded')
        print('------------------------------------------')
    return True


def saveCheckpoint(path, name, config, saver, sess):

    folder = generate_path(path, config)

    print('\n------------------------------------------')
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print('{} is Created.'.format(folder))

    save_path = saver.save(sess, os.path.join(folder, name))
    print("Model saved successfully in file: %s" % save_path)
    print('------------------------------------------')
    return True


def generate_path(path, config):

    dir_str = config['name']

    if config['normalization'] == 0:
        dir_str += '_'+'sigmoid'
    else:
        dir_str += '_'+'tanh'

    if config['regularization']['l2_regularization']:
        dir_str += '_'+'l2-reg'
    if config['dropout']['bool']:
        dir_str += '_'+'dropout'
    
    if config['optimizer']['name'] == 'gradientdescentoptimizer':
        dir_str += '_'+'gD'
    else:
        dir_str += '_'+'adam'
    
    hidden_str = '('
    for val in config['hidden_neurons']:
        hidden_str += str(val)+','
    hidden_str += ')'

    # '_'+ str(config['hidden_neurons'])+
    dir_str += '_'+hidden_str+'_'+str(len(config['hidden_neurons']))

    path = os.path.join(path, dir_str)
    return path
