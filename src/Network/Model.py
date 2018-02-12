import tensorflow as tf
import math

class Model():
    def __init__(self, config):
        self._nLayers = config["hidden_layers"]
        self._hidden_neurons = config["hidden_neurons"]
        self._batch_size = config["batch_size"]
        self._optimizer = config['optimizer']
        self._activate_fn = config['activate_fn']

        self._pretrain_layer_weights = []


    def _create_weights(self, shape, name=""):
        return tf.get_variable("Weights_"+name,
                               dtype=tf.float32,
                               initializer=tf.random_uniform(shape,
                                             minval=-1.0,
                                             maxval=1.0),
                               trainable=True)


    def _create_bias(self, shape, name=""):
        return tf.get_variable("Bias_"+name,
                               dtype=tf.float32,
                               initializer=tf.constant(1.0, shape=[shape[1]]),
                               trainable=False)


    def _compute_layer(self, input_tensor, weights, biases, activate_fn, name):

        if activate_fn == 'tanh':
            return tf.tanh(
                tf.add(tf.matmul(input_tensor, weights), biases), name=name)
        else:
            return tf.sigmoid(
                tf.add(tf.matmul(input_tensor, weights), biases), name=name)



    def inference(self,input_neurons, input_tensor):

        self._input_neurons = input_neurons
        self._output_neurons = self._input_neurons


        for layer in range(self._nLayers):

            # Computing Shapes of Weights and Biases for each hidden-layer
            if layer == 0:
                shape = [self._input_neurons, self._hidden_neurons[layer]]
                # pretrain_shape = [self._hidden_neurons[layer], self._input_neurons]
                layer_name = "Hidden-Layer-{}".format(layer+1)

            else:
                shape = [self._hidden_neurons[layer-1], self._hidden_neurons[layer]]
                layer_name = "Hidden-Layer-{}".format(layer+1)

            print(shape)

            # Creating layers for model
            with tf.variable_scope(layer_name) as scope:
                weights = self._create_weights(shape, layer_name)
                biases = self._create_bias(shape, layer_name)

                layer_output = self._compute_layer(input_tensor, weights,
                                                   biases,
                                                   self._activate_fn,
                                                   scope.name)

                # Temparary weights and biases used for pre=training the hidden layers
                tmp_weights = self._create_weights([shape[1], shape[0]], "pretrain_weight")
                tmp_biases = self._create_bias([shape[1], shape[0]], 'pretrian_bias')
                pretrain_layer = self._compute_layer(layer_output, tmp_weights,
                                                     tmp_biases,
                                                     self._activate_fn,
                                                     scope.name+'-pretrain')
                self._pretrain_layer_weights.append({ 'output_layer': pretrain_layer,
                                                     'input_tensor': input_tensor,
                                                     'var_list': [weights, tmp_weights]})

                input_tensor = layer_output


        output_shape = [self._hidden_neurons[-1], self._output_neurons]
        layer_name = 'Output-Layer'
        print(output_shape)
        with tf.variable_scope(layer_name) as scope:
            weights = self._create_weights(output_shape, layer_name)
            biases = self._create_bias(output_shape, layer_name)

            output_layer = self._compute_layer(input_tensor, weights,
                                               biases,
                                               self._activate_fn,
                                               scope.name)


        return self._pretrain_layer_weights, output_layer


    def loss_fn(self, input_tensor, predict_tensor):
        with tf.variable_scope('loss') as scope:
            nonzero_bool = tf.not_equal(input_tensor, tf.constant(0, tf.float32))
            nonzero_mat = tf.cast(nonzero_bool, tf.float32)
            predict_nonzero = tf.multiply(predict_tensor, nonzero_mat)
            cross_entropy = tf.squared_difference(predict_nonzero, input_tensor)
            cost = tf.reduce_mean(cross_entropy, name=scope.name)

        return cost


    def optimizer(self,):
        pass


    def error(self, output, l_out):
        with tf.variable_scope("Error") as scope:
            nonzero_matrix = tf.cast(
                tf.not_equal(output, tf.constant(0, tf.float32)),
                tf.float32
            )

            l_out_nonzero = tf.multiply(l_out, nonzero_matrix)
            nonzero_count = tf.reduce_sum(nonzero_matrix)
            with tf.variable_scope("MAE") as sub_scope:
                mae = tf.multiply(
                        tf.truediv(
                            tf.reduce_sum(
                                tf.abs(
                                    tf.subtract(l_out_nonzero,
                                                output))),
                            nonzero_count),
                        tf.constant(2, tf.float32),
                        name=sub_scope.name)

            with tf.variable_scope("RMS") as sub_scope:
                rms = tf.multiply(
                        tf.truediv(
                            tf.reduce_sum(
                                tf.square(
                                    tf.subtract(l_out_nonzero,
                                                output))),
                            nonzero_count),
                        tf.constant(2, tf.float32),
                        name=sub_scope.name)

            return mae, rms


    def _error_mini_batch(self, X, Y):
        with tf.variable_scope("error_mini_batch") as scope:
            nonzero_mat = tf.cast(
                tf.not_equal(X, tf.constant(0, tf.float32)),
                tf.float32
            )
            Y_nonzero = tf.multiply(Y, nonzero_mat)

            with tf.variable_scope("mae") as sub_scope:
                mae = tf.reduce_sum(
                    tf.abs(
                        tf.subtract(Y_nonzero, X)
                    )
                , name=sub_scope.name)

            with tf.variable_scope("rms") as sub_scope:
                rms = tf.reduce_sum(
                    tf.squared_difference(Y_nonzero, X)
                , name=sub_scope.name)

        return mae, rms
