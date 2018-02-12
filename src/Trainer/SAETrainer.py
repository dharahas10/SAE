import tensorflow as tf
from src.Network.Model import Model
from src.helper import *
import math

class SAETrainer(object):

    def __init__(self, config, info):
        print("----------------Stacked Auto-Encoder Trainer----------------")
        self._type = config['type']

        if self._type == 'U':
            self._input_neurons = info['nV']
        else:
            self._input_neurons = info['nU']

        self._output_nuerons = self._input_neurons
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._nLayers = config['hidden_layers']
        self._config = config

        self._optimizer = self._get_optimizer_fn(config['optimizer'])


    def _get_optimizer_fn(self, op_config):
        if op_config['name'] == 'GradientDescentOptimizer':
            print('--------------- Trainer using GradientDescentOptimizer -----------')
            self._learning_rate = op_config['learning_rate']
            optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)

        return optimizer


    def optimize_fn(self, loss, trainable_weights):
        return self._optimizer.minimize(loss,
                                        var_list=trainable_weights)



    def train(self, train_data):

        self._train = train_data[self._type]['data']

        model = Model(self._config)

        with tf.Graph().as_default():

            train_indices = tf.placeholder(tf.int32, name='train_indices')
            train_values = tf.placeholder(tf.float32, name='train_values')
            shape = [self._batch_size, self._input_neurons]
            # Dense train input data
            X = tf.sparse_to_dense(train_indices, shape, train_values, name='X')

            pretrain_layers_weights, predict_layer = model.inference(self._input_neurons,X)

            loss_op_layers = []
            optimize_op_layers = []

            for layer_dict in pretrain_layers_weights:
                curr_predict = layer_dict['output_layer']
                curr_input = layer_dict['input_tensor']
                curr_trainable_weights = layer_dict['var_list']
                curr_loss_op = model.loss_fn(curr_input, curr_predict)
                loss_op_layers.append(curr_loss_op)

                curr_optimize_op = self.optimize_fn(curr_loss_op, curr_trainable_weights)
                optimize_op_layers.append(curr_optimize_op)

            # predict_layer optimization
            loss_op_predict = model.loss_fn(X, predict_layer)
            loss_op_layers.append(curr_loss_op)
            optimiz_op_predict = self.optimize_fn(loss_op_predict, None)
            optimize_op_layers.append(optimiz_op_predict)


            mae_op, rms_op = model._error_mini_batch(X, predict_layer)

            init = tf.global_variables_initializer()

            # saver model
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(init)
                self._loadCheckpointSaver(self._config, saver, sess)

                print("------- Pre-Training Strating ----------")
                layer_counter = 1
                for optim_layer in optimize_op_layers:
                    print('\n------Current Per-Training Layer-{} ------'.format(layer_counter))

                    for epoch in range(self._epochs):
                        print("\n Current Running Epoch is {}/{}"
                          .format(epoch+1, self._epochs))

                        for indices, ratings in iterate_mini_batch(self._train,
                                                                  self._batch_size):

                            _ = sess.run(optim_layer,
                                         {train_indices: indices,
                                          train_values: ratings})


                    self._saveCheckpoint(self._config, saver, sess)
                    layer_counter += 1




                print("\n\n-----------------Calculating the error-------")
                tmp_mae = 0
                tmp_rms = 0
                tmp_count = 0
                counter = 0
                for indices, ratings in iterate_mini_batch(self._train,
                                                          self._batch_size):

                    curr_mae, curr_rms = sess.run([mae_op, rms_op],
                                                  {train_indices: indices,
                                                   train_values: ratings})

                    tmp_mae += curr_mae
                    tmp_rms += curr_rms
                    tmp_count += len(ratings)
                    counter +=1
                    print("Current Counter {} and MAE: {} and RMS: {}".format(counter, tmp_mae, tmp_rms))

            print("-----------------------------------------------------------")
            print("----------------Training Completed------------------")
            print("Total MAE: {} and RMS: {}".format(tmp_mae*2/tmp_count, math.sqrt(tmp_rms*2/tmp_count)))


    def _loadCheckpointSaver(self, config, saver, sess):

        model_dir = config['save_model']['path']+self._type+'/'+str(self._nLayers)+'_'+str(config['hidden_neurons'])+'/'
        if find_dir(model_dir):
            # Found saved model
            print("Restoring From the previously Found Model .......")
            saver.restore(sess, model_dir+config['save_model']['name'])
            print("Previous Model Found and restored Succesfully")
        else:
            print("No previously saved model found")


    def _saveCheckpoint(self, config, saver, sess):
        model_dir = config['save_model']['path']+self._type+'/'+str(self._nLayers)+'_'+str(config['hidden_neurons'])+'/'
        # Save the variables to disk.
        if not find_dir(model_dir):
            make_dir(model_dir)

        save_path = saver.save(sess, model_dir+config['save_model']['name'])
        print("Model saved in file: %s" % save_path)
