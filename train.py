#!/usr/bin/env python
"""Predicting poker hand's strength with artificial neural networks in Python"""

from __future__ import absolute_import, print_function
from argparse import ArgumentParser as Parser
from itertools import izip

import numpy as np
from pybrain.structure.modules import LinearLayer, SoftmaxLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import Validator

from load import load_data


TRAIN_METHODS = {'gdm': BackpropTrainer,
                 'scg': None,
                 'rp': RPropMinusTrainer}

ACTIVATION_FNS = {'purelin': LinearLayer,
                  'tansig': TanhLayer}


def get_parser():
    """Parse command-line arguments"""
    parser = Parser(description='Train neural network to classify poker hands')
    parser.add_argument('-a', '--activation', type=str,
                        nargs='?', default='tansig',
                        help='hidden layer activation fn (default: tansig)')
    parser.add_argument('-me', '--max_epochs', type=int, nargs='?', default=1000,
                        help='# of training iterations (default: 1000)')
    parser.add_argument('-hn', '--hidden-neurons', type=int, nargs='?', default=10,
                        help='# of hidden neuron units (default: 10)')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        nargs='?', default=0.2,
                        help='controls size of weight changes (default: 0.2)')
    parser.add_argument('-m', '--method', type=str, nargs='?', default='rp',
                        help='training method (default: rp)')
    parser.add_argument('-nt', '--num-testing', type=int,
                        nargs='?', default='25000',
                        help='# of testing inputs (default: 25000)')
    parser.add_argument('-v', '--verbose', help='print status messages',
                        action='store_true')
    return parser


def train(args, training_ds):
    """Build and train feed-forward neural network

       Keyword arguments:
       args -- program arguments (dict)
       training_ds -- suit, ranks, and target hands (list)
    """
    # Build a feed-forward network with x hidden units
    if args['verbose']:
        print('\nBuilding network:')
        print('\tinput neurons: {}'.format(training_ds.indim))
        print('\thidden neurons: {}'.format(args['hidden_neurons']))
        print('\toutput neurons: {}'.format(training_ds.outdim))
        print('\thidden layer activation fn: {}'.format(args['activation']))
        print('\toutput layer activation fn: softmax')

    ff_network = buildNetwork(training_ds.indim,
                              args['hidden_neurons'],
                              training_ds.outdim,
                              hiddenclass=ACTIVATION_FNS[args['activation']],
                              outclass=SoftmaxLayer)

    if args['verbose']:
        print('Network built.')

    # Train using user-specified method and training data for n epochs
    if args['verbose']:
        print('\nTraining network:')
        print('\tmax epochs: {}'.format(args['max_epochs']))
        print('\ttraining method: {}'.format(args['method']))
        if args['method'] == 'gdm':
            momentum = 0.7
        else:
            momentum = 0.0
        print('\tmomentum: {}'.format(momentum))
        print('\tlearning rate: {}'.format(args['learning_rate']))

    trainer = TRAIN_METHODS[args['method']](ff_network, dataset=training_ds,
                                            verbose=args['verbose'],
                                            momentum=momentum,
                                            learningrate=args['learning_rate'])

    try:
        trainer.trainEpochs(args['max_epochs'])
    except (KeyboardInterrupt, EOFError):
        pass

    return trainer, ff_network


def evaluate(args, trainer, ff_network, training_ds, testing_ds):
    """Evaluate the networks hit rate and MSE on training and testing"""
    if args['verbose']:
        print('\nEvaluating the networks hit rate and MSE:')
    print('\tTotal epochs: %4d' % trainer.totalepochs)

    def print_dataset_eval(dataset):
        """Print dataset hit rate and MSE"""
        predicted = [ff_network.activate(x) for x in dataset['input']]
        hits = 0
        mse = 0

        for pred, targ in izip(predicted, dataset['target']):
            hits += Validator.classificationPerformance(pred, targ)
            mse += Validator.MSE(pred, targ)
        total_hits = hits / len(predicted)
        total_mse = mse / len(predicted)

        print('\t\tHit rate: {}'.format(total_hits))
        print('\t\tMSE: {}'.format(total_mse))

    print('\n\tTraining set:')
    print_dataset_eval(training_ds)

    print('\n\tTesting set:')
    print_dataset_eval(testing_ds)


def run_simulation(args):
    """Run ANN simulation"""
    # Load training and test data
    training_ds, testing_ds = load_data(args)

    # Build and train feed-forward neural network
    trainer, ff_network = train(args, training_ds)

    # Use the trainer to evaluate the network on the training and test data
    evaluate(args, trainer, ff_network, training_ds, testing_ds)


def command_line_runner():
    """Handle command-line interaction"""
    parser = get_parser()
    args = vars(parser.parse_args())

    run_simulation(args)


if __name__ == '__main__':
    command_line_runner()
