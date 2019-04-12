from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.d = n_inputs
        self.hidden = n_hidden
        self.classes = n_classes
        self.loss = 0
        self.accuracy = 0
        depth = len(self.hidden)
        layers = list()
        for i in range(depth):
            if i == 0:
                h_layer = layer(self.d, self.hidden[i])
            else:
                h_layer = layer(self.hidden[i-1], self.hidden[i])
            layers.append(h_layer)
        output_layer = layer(self.hidden[-1], self.classes)
        output_layer.activation = SoftMax()
        output_layer.name = 'Out'
        layers.append(output_layer)
        self.layers = layers
        self.loss = CrossEntropy()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        layer_input = x
        for layer in self.layers:
            # print(layer.name)
            layer_input = layer.forward(layer_input)
        return layer_input

    def backward(self, dout, index, args):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        n = len(self.layers)
        layer_d = dout
        for i in range(n-1, -1, -1):
            layer = self.layers[i]
            # print(layer.name)
            if i == n - 1:
                layer_d = layer.backward(layer_d, args, index=index)
            else:
                layer_d = layer.backward(layer_d, args)
        return


class layer:
    def __init__(self, d, h):
        self.name = 'H'
        self.input = None
        self.linear = Linear(d, h)
        self.activation = ReLU()
        self.output = None

    def forward(self, x):
        linear_output = self.linear.forward(x)
        activation_output = self.activation.forward(linear_output)
        return activation_output

    def backward(self, dout, args, index=None):
        if index is not None:
            activation_d = self.activation.backward(dout, index)
            linear_d = self.linear.backward(activation_d)
            n = linear_d.shape[0]
            batch = np.zeros(self.linear.w.shape)
            for i in range(n):
                batch = batch + linear_d[i]
            self.linear.w = self.linear.w - args.learning_rate * batch / n
        else:
            activation_d = self.activation.backward(dout)
            linear_d = self.linear.backward(activation_d)
            n = linear_d.shape[0]
            batch = np.zeros(self.linear.w.shape)
            for i in range(n):
                batch = batch + linear_d[i]
            self.linear.w = self.linear.w - args.learning_rate * batch / n
        return activation_d