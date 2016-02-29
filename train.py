#!/usr/bin/env python
"""Predicting poker hand's strength with artificial neural networks in Python"""

from __future__ import print_function
from argparse import ArgumentParser as Parser
from itertools import izip

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.supervised.trainers import RPropMinusTrainer

TRAIN_METHODS = {'rprop': RPropMinusTrainer,
                 }


def get_parser():
    """Parse command-line arguments"""
    parser = Parser(description='Train neural network to classify poker hands')
    parser.add_argument('-bs', '--batch-size', type=int, nargs='?', default=5,
                        help='size of training batches')
    parser.add_argument('-e', '--epochs', help='# of training iterations',
                        type=int, nargs='?', default=100)
    parser.add_argument('-hu', '--hidden', help='# of hidden units',
                        type=int, nargs='?', default=10)
    parser.add_argument('-m', '--method', help='training method',
                        type=str, nargs='?', default='rprop')
    parser.add_argument('-nt', '--num-testing', help='# of testing inputs',
                        type=int, nargs='?', default='25000')
    parser.add_argument('-v', '--verbose', help='print status messages',
                        action='store_true')
    return parser


def preprocess_load_data(loaded_lines):
    """Separate target column data and perform bit vector transformation"""
    def bit_vec_transform(num):
        """Transform target column values from 0-9 to a bit vector"""
        vec = [0]*10
        vec[int(num)] = 1
        return vec

    input_data = []
    target_data = []
    for line in loaded_lines:
        input_data.append([int(x) for x in line[:-1]])
        target_data.append(bit_vec_transform(line[-1]))

    if not input_data or not target_data:
        raise ValueError('Input and/or target data not found!')

    return input_data, target_data


def build_dataset(args, input_data, target_data):
    """Build SupervisedDataSet by adding data samples

       Keyword arguments:
       input_data -- suit and rank combinations (list)
       target_data -- different poker hands (list)
    """
    dataset = SupervisedDataSet(len(input_data[0]), len(target_data[0]))
    for in_data, tg_data in izip(input_data, target_data):
        dataset.addSample(in_data, tg_data)

    if args['verbose']:
        print('Dataset built.')

    return dataset


def load_training_data(args):
    """Load training data and perform preprocessing"""
    if args['verbose']:
        print('Loading training dataset...')

    with open('data/poker-hand-training-true.data', 'r') as training_file:
        training_lines = [line.split(',') for line in training_file.readlines()]

    # Separate target column data and perform bit vector transformation
    training_data, target_data = preprocess_load_data(training_lines)

    # Build SupervisedDataSet by adding data samples
    return build_dataset(args, training_data, target_data)


def load_testing_data(args):
    """Load testing data up to num_testing lines (default: 100K)"""
    if args['verbose']:
        print('Loading testing dataset...')

    with open('data/poker-hand-testing.data', 'r') as testing_file:
        testing_lines = [testing_file.readline().split(',')
                         for _ in range(args['num_testing'])]

    # Separate target column data and perform bit vector transformation
    testing_data, target_data = preprocess_load_data(testing_lines)

    # Build SupervisedDataSet by adding data samples
    return build_dataset(args, testing_data, target_data)


def load_data(args):
    """Load training and test data"""
    return load_training_data(args), load_testing_data(args)


def train(args, training_ds):
    """Build and train feed-forward neural network

       Keyword arguments:
       args -- program arguments (dict)
       training_ds -- suit, ranks, and target hands (SupervisedDataSet)
    """
    # Build a feed-forward network with x hidden units
    if args['verbose']:
        print('Building network...')
    ff_network = buildNetwork(training_ds.indim, args['hidden'],
                              training_ds.outdim)
    if args['verbose']:
        print('Network built.')

    # Train using user-specified method and training data for n epochs
    if args['verbose']:
        print('Training network...')
    trainer = TRAIN_METHODS[args['method']](ff_network, dataset=training_ds,
                                            verbose=args['verbose'])
    batch_size = args['batch_size']
    max_epochs = args['epochs']
    for i in range(0, max_epochs, batch_size):
        if args['verbose']:
            print('Training batch {0} of {1}.'.format(i, max_epochs))
        trainer.trainEpochs(batch_size)

    return trainer


def evaluate(trainer, training_ds, testing_ds):
    """Use the trainer to evaluate the network on the training and test data"""
    training_result = percentError(trainer.testOnClassData(),
                                   training_ds['class'])
    testing_result = percentError(trainer.testOnClassData(dataset=testing_ds),
                                  training_ds['class'])
    print("epoch: %4d" % trainer.totalepochs)
    print("train error: %5.2f%%" % training_result)
    print("test error: %5.2f%%" % testing_result)


def command_line_runner():
    """Handle command-line interaction"""
    parser = get_parser()
    args = vars(parser.parse_args())

    # Load training and test data
    training_ds, testing_ds = load_data(args)

    # Build and train feed-forward neural network
    trainer = train(args, training_ds, args['method'])

    # Use the trainer to evaluate the network on the training and test data
    evaluate(trainer, training_ds, testing_ds)


if __name__ == '__main__':
    command_line_runner()
