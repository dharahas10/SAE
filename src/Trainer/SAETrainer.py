import math
import time
from pprint import pprint

import tensorflow as tf

from src.helper import duration, iterate_mini_batchOne, iterate_mini_batchTwo
from src.Network.Model import Model
from src.utils.errors import error_fn
from src.utils.losses import loss_fn
from src.utils.optimizer import optimize_fn
from src.utils.saver import loadCheckpointSaver, saveCheckpoint
from src.utils.tf_helper import corruptData, get_global_step


class SAETrainer(object):

    def __init__(self, config, info):

        print("\n----------------Stacked Auto-Encoder Trainer----------------")
        self._input_neurons = info['nItems']+1
        self._output_nuerons = self._input_neurons
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._nLayers = len(config['hidden_neurons'])

        # noisy hyperparameters
        self._isNoisy = config['noise']['isNoisy']
        self._noiseConfig = None
        if self._isNoisy:
            self._noiseRatio = config['noise']['noiseRatio']
            self._noiseConfig = {}
            self._noiseConfig['alpha'] = config['noise']['alpha']
            self._noiseConfig['beta'] = config['noise']['beta']

        # regularization hyperparameter
        self._regularization = config['regularization']['l2_regularization']
        self._l2_beta = 0
        if self._regularization:
            self.l2_beta = config['regularization']['l2_beta']

        # optimization hyperparameters
        self._optimizer_config = config['optimizer']

        # normalization parameters
        self._normalization = config['normalization']

        # model saving details
        self._save_path = config['save_model']['path']
        self._save_name = config['save_model']['name']

        # errors
        self._bestRMSE = 999999
        self._bestMAE = 999999
        self._rmse = []
        self._mae = []
        # setting random seed for tensorflow
        tf.set_random_seed(config['seed'])
        self._config = config

    def build_graph(self):

        model = Model(self._config)
        graph = tf.Graph()

        with graph.as_default():
            # Train graph
            # input placeholders
            train_indices = tf.placeholder(tf.int32, name='train_indices')
            train_values = tf.placeholder(tf.float32, name='train_values')
            train_shape = [self._batch_size, self._input_neurons]

            # Sparse to Dense train_input
            X = tf.sparse_to_dense(train_indices,
                                   train_shape,
                                   train_values,
                                   name='X')
            Y = X  # input == output *autoencoder*

            if self._isNoisy:
                X = corruptData(train_indices, train_values, train_shape,
                                noiseRatio=self._noiseRatio,
                                name='noisy_X')

            # Get all layers for training
            pretrain_layers = model.inference(X, self._input_neurons)

            optimize_ops_list = []
            # loss_ops_list = []

            layer_counter = 1
            for layer_info in pretrain_layers:

                layer = layer_info['layer']
                train_vars = layer_info['train_vars']

                global_step_tensor = get_global_step(
                    name='layer_'+str(layer_counter))
                layer_counter += 1

                loss_op = loss_fn(Y, layer,
                                  noisy_input=X,
                                  isNoisy=self._isNoisy,
                                  noisyConfig=self._noiseConfig,
                                  l2_reg=self._regularization,
                                  l2_beta=self._l2_beta,
                                  trainable_vars=train_vars)

                optimize_op = optimize_fn(self._optimizer_config, loss_op,
                                          train_vars=train_vars,
                                          global_step=global_step_tensor)

                # loss_ops_list.append(loss_op)
                optimize_ops_list.append(optimize_op)
            # pprint(optimize_ops_list)

            # test graph
            test_indices = tf.placeholder(tf.int32, name="test_indices")
            test_values = tf.placeholder(tf.float32, name='test_values')
            test_shape = test_shape = [self._batch_size, self._input_neurons]
            # test_input
            test_tensor = tf.sparse_to_dense(
                test_indices, test_shape, test_values, name='test_tensor')

            # since Y is original non-corrupt input from above
            X = Y
            pretrain_test_layers = model.inference(X, self._input_neurons)
            # pprint(pretrain_test_layers)

            error_ops_list = []
            for layer_info in pretrain_test_layers:

                predict_ = layer_info['layer']
                error_op = error_fn(test_tensor, predict_)
                error_ops_list.append(error_op)
            # pprint(error_ops_list)

        return graph, optimize_ops_list, error_ops_list, {'train_indices': train_indices,
                                                          'train_values': train_values,
                                                          'test_indices': test_indices,
                                                          'test_values': test_values}

    def execute(self, train_data, test_data):

        self._train = train_data
        self._test = test_data

        # Build Graph
        graph, optimize_ops_list, error_ops_list, feed_dict = self.build_graph()
        # check lengths of layers
        assert len(optimize_ops_list) == len(
            error_ops_list), "ERROR:: lengths of optimize_ops_list and error_ops_list are not equal"

        # start session
        with tf.Session(graph=graph) as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # Load graph if already saved
            loadCheckpointSaver(self._save_path,
                                self._save_name,
                                self._config,
                                saver, sess)

            # Train/Optimize each layer
            for i in range(len(optimize_ops_list)):
                print(
                    "\n===================== Training Layer-{} ===============".format(i+1))
                training_layer_start = time.time()

                optimize_op = optimize_ops_list[i]
                error_op = error_ops_list[i]

                for epoch in range(1, self._epochs+1):
                    print(
                        "===> [ current epoch: {}/{} ]".format(epoch, self._epochs))

                    # train with mini-batches
                    batch_counter = 1
                    for batch in iterate_mini_batchOne(self._train, self._batch_size):
                        # print("[current Bactch counter : {}".format(batch_counter))
                        batch_counter += 1

                        _ = sess.run(optimize_op, {
                            feed_dict['train_indices']: batch['indices'],
                            feed_dict['train_values']: batch['values']
                        })

                # testing each layer for rmse and mae
                mse = 0
                mae = 0
                noRatings = 0
                for test_batch, train_batch in iterate_mini_batchTwo(self._test, self._train, self._batch_size):

                    local_mse, local_mae = sess.run([error_op['mse'], error_op['mae']], {
                        feed_dict['train_indices']: train_batch['indices'],
                        feed_dict['train_values']: train_batch['values'],
                        feed_dict['test_indices']: test_batch['indices'],
                        feed_dict['test_values']: test_batch['values']
                    })
                    mse += local_mse
                    mae += local_mae
                    # assert local_nonzero_count == len(test_batch['values']), "ERROR:: testing nonzero_counts of values are not equal values: {} and sess: {}".format(len(test_batch['values']), local_nonzero_count)
                    noRatings += len(test_batch['values'])

                if self._normalization == 0:
                    rmse = math.sqrt(mse/noRatings) * 5
                    mae = (mae/noRatings) * 5
                else:
                    rmse = math.sqrt(mse/noRatings) * 2
                    mae = (mae/noRatings) * 2

                self._rmse.append(rmse)
                self._mae.append(mae)

                training_layer_end = time.time()
                print("[RMSE => {} and MAE => {} and time : {}]".format(
                    rmse, mae, duration(training_layer_start, training_layer_end)))

                # save model for later
                saveCheckpoint(self._save_path, self._save_name,
                               self._config, saver, sess)

        # Best RMSE and MAE
        self._bestMAE = min(self._mae)
        self._bestRMSE = min(self._rmse)
        print("\n\n================================================")
        print("\n\n\t The Best RMSE: {0:.4f}  and Best MAE: {1:.4f} ".format(
            self._bestRMSE/1.0000, self._bestMAE/1.0000))
        print("\n Successfully Trained!!!!")
