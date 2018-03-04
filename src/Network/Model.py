import math

import tensorflow as tf

from src.utils.layers import get_layer_variables


class Model():
    def __init__(self, config):

        self._hidden_neurons = config["hidden_neurons"]
        self._nLayers = len(self._hidden_neurons)
        self._input_neurons = None

        if config['normalization'] == 0:
            self._activation_fn = 'sigmoid'
        elif config['normalization'] == -1:
            self._activation_fn = 'tanh'

        # dropout-neurons
        self._dropout = config['dropout']['bool']
        self._keep_prob = 1
        if self._dropout:
            self._keep_prob = config['dropout']['keep_prob']

        # list to store all trainable variables
        self.global_trainable_vars = []

    def inference(self, inputs, input_size):

        if self._input_neurons is None:
            self._input_neurons = input_size

        pretrain_layers = []

        layer_shapes = self.get_layer_shapes(self._input_neurons,
                                             self._hidden_neurons)

        layer_input = inputs
        for i in range(self._nLayers):

            hidden_layer, hidden_weights_bias = get_layer_variables(layer_input, layer_shapes[i],
                                                                    activation_fn=self._activation_fn,
                                                                    dropout=self._dropout,
                                                                    keep_prob=self._keep_prob,
                                                                    name=str(i+1))
            layer_input = hidden_layer
            self.global_trainable_vars += hidden_weights_bias

            if i < self._nLayers-1:
                # another layer for pre-training each layer
                extra_layer_shape = [layer_shapes[i][1], self._input_neurons]
                extra_layer, extra_weights_bias = get_layer_variables(hidden_layer, extra_layer_shape,
                                                                      activation_fn=self._activation_fn,
                                                                      name=str(i+1)+"_decoder")

                trainable_vars = []
                trainable_vars += hidden_weights_bias
                trainable_vars += extra_weights_bias

                pretrain_layers.append({
                    'layer': extra_layer,
                    'train_vars': trainable_vars
                })

        # Final Layer : a bit different includes all weights and biases for training
        final_layer, final_weights_bias = get_layer_variables(layer_input, layer_shapes[self._nLayers],
                                                              activation_fn=self._activation_fn,
                                                              name="output")
        self.global_trainable_vars += final_weights_bias

        trainable_vars = []
        trainable_vars += hidden_weights_bias
        trainable_vars += final_weights_bias
        # For training only last layer
        pretrain_layers.append({
            'layer': final_layer,
            'train_vars': trainable_vars
        })

        # For trianing complete layer
        if self._nLayers > 1:
            pretrain_layers.append({
                'layer': final_layer,
                'train_vars': self.global_trainable_vars
            })

        return pretrain_layers

    def get_layer_shapes(self, input_neurons, hidden_neurons):

        nLayers = len(hidden_neurons)
        shapes = []
        for i in range(nLayers+1):
            if i == 0:
                shapes.append([input_neurons, hidden_neurons[i]])
            elif i == nLayers:
                shapes.append([hidden_neurons[i-1], input_neurons])
            else:
                shapes.append([hidden_neurons[i-1], hidden_neurons[i]])

        return shapes
