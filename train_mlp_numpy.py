from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp import MLP
from modules import CrossEntropy
import sklearn.datasets
import sklearn.preprocessing

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets, mlp):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    d, num = predictions.shape
    hit = 0
    result = mlp.forward(predictions)
    print("loss", mlp.loss.forward(result, targets))
    result_T = result.T
    targets_T = targets.T
    for i in range(num):
        est = np.argmax(result_T[i])
        real = np.argmax(targets_T[i])
        if est == real:
            hit += 1
    accuracy = hit/num * 100
    return accuracy


def train(mlp, x, y, args):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    sample = x
    label = y
    forward_output = mlp.forward(sample)
    loss = mlp.loss.forward(forward_output, label)
    # print(loss)
    loss_d, index = mlp.loss.backward(forward_output, label)
    mlp.backward(loss_d, index, args)
    return mlp


def main(args):
    """
    Main function
    """
    x, temp_y = sklearn.datasets.make_moons(n_samples=1000)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(temp_y)
    onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y = onehot_encoder.fit_transform(integer_encoded)

    predictions, temp_y = sklearn.datasets.make_moons(n_samples=200)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(temp_y)
    onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    targets = onehot_encoder.fit_transform(integer_encoded)

    hiddens_temp = args.dnn_hidden_units.split(",")
    hiddens = list()
    for value in hiddens_temp:
        hiddens.append(int(value))
    mlp = MLP(2, hiddens, 2)
    for i in range(args.max_steps):
        mlp = train(mlp, x.T, y.T, args)
        # print("train")
        # for layer in mlp.layers:
        #     print(layer.linear.w.shape)
        if i % args.eval_freq == 0:
            print(accuracy(predictions.T, targets.T, mlp))
    return


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args)
