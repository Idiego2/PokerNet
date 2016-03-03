#!/usr/bin/env python
"""Predicting poker hand's strength with artificial neural networks in Python"""

from __future__ import absolute_import, print_function
from argparse import ArgumentParser as Parser
from itertools import izip

import numpy as np

from load import load_data
from activation_functions import softmax_function, sigmoid_function
from neuralnet import NeuralNet


def gdm_trainer(ff_network, training_ds, max_iterations, learning_rate,
                momentum_factor):
    """Gradient descent with momentum trainer which uses back-propagation"""
    if momentum_factor == 0:
        raise ValueError('You must set the momentum for gradient descent!')
    return ff_network.backpropagation(training_ds,
                                      max_iterations=max_iterations,
                                      learning_rate=learning_rate,
                                      momentum_factor=momentum_factor)


def scg_trainer(ff_network, training_ds, max_iterations, learning_rate,
                momentum_factor):
    """Scaled conjugate gradient trainer"""
    return ff_network.scg(training_ds, max_iterations=max_iterations)


def rp_trainer(ff_network, training_ds, max_iterations, learning_rate,
               momentum_factor):
    """Resilient back-propagation trainer"""
    return ff_network.resilient_backpropagation(training_ds,
                                                max_iterations=max_iterations)


TRAIN_METHODS = {'gdm': gdm_trainer,
                 'scg': scg_trainer,
                 'rp': rp_trainer}

ACTIVATION_FNS = {'softmax': softmax_function,
                  'sigmoid': sigmoid_function}


def get_parser():
    """Parse command-line arguments"""
    parser = Parser(description='Train neural network to classify poker hands')
    parser.add_argument('-ah', '--activation-hidden', type=str,
                        nargs='?', default='sigmoid',
                        help='hidden layer activation fn (default: sigmoid)')
    parser.add_argument('-ao', '--activation-output', type=str,
                        nargs='?', default='softmax',
                        help='output layer activation fn (default: softmax)')
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=1000,
                        help='# of training iterations (default: 1000)')
    parser.add_argument('-hi', '--hidden', type=int, nargs='?', default=10,
                        help='# of hidden units (default: 10)')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        nargs='?', default=0.2,
                        help='controls size of weight changes (default: 0.2)')
    parser.add_argument('-m', '--method', type=str, nargs='?', default='rp',
                        help='training method (default: rp)')
    parser.add_argument('-mo', '--momentum', type=float, nargs='?', default=0,
                        help='training momentum (default: 0)')
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
    feature_dim = len(training_ds[0].features)
    target_dim = len(training_ds[0].targets)
    settings = {"n_inputs": feature_dim,
                "layers": [(feature_dim + target_dim + args['hidden'],
                            ACTIVATION_FNS[args['activation_hidden']]),
                           (target_dim,
                            ACTIVATION_FNS[args['activation_output']])]}

    if args['verbose']:
        print('\nBuilding network:')
        print('\tinput neurons: {}'.format(feature_dim))
        print('\thidden neurons: {}'.format(args['hidden']))
        print('\toutput neurons: {}'.format(target_dim))

    ff_network = NeuralNet(settings)

    if args['verbose']:
        print('Network built.')

    # Train using user-specified method and training data for n epochs
    if args['verbose']:
        print('\nTraining network:')
        print('\tmax epochs: {}'.format(args['epochs']))
        print('\tmethod: {}'.format(args['method']))
        print('\tmomentum: {}'.format(args['momentum']))
        print('\tlearning rate: {}'.format(args['learning_rate']))
    
    try:
        TRAIN_METHODS[args['method']](ff_network, training_ds,
                                      max_iterations=args['epochs'],
                                      learning_rate=args['learning_rate'],
                                      momentum_factor=args['momentum'])
    except (KeyboardInterrupt, EOFError):
        pass

    return ff_network


def evaluate(args, ff_network, training_ds, testing_ds):
    """Evaluate the networks overall results and MSE on training and testing"""
    if args['verbose']:
        print('\nEvaluating the networks overall results and MSE:')

    print('\n\tTraining set:')
    tr_mse, tr_res = ff_network.test(training_ds)

    print('\n\tTesting set:')
    te_mse, te_res = ff_network.test(testing_ds)


def run_simulation(args):
    """Run ANN simulation"""
    # Load training and test data
    training_ds, testing_ds = load_data(args)

    # Build and train feed-forward neural network
    ff_network = train(args, training_ds)

    # Use the trainer to evaluate the network on the training and test data
    evaluate(args, ff_network, training_ds, testing_ds)


def command_line_runner():
    """Handle command-line interaction"""
    parser = get_parser()
    args = vars(parser.parse_args())

    run_simulation(args)


if __name__ == '__main__':
    command_line_runner()
