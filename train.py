#!/usr/bin/env python
"""Predicting poker hand's strength with artificial neural networks in Python"""

from __future__ import absolute_import, print_function
from argparse import ArgumentParser as Parser

from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.supervised.trainers import RPropMinusTrainer

from load import load_data

TRAIN_METHODS = {'rprop': RPropMinusTrainer,
                 }


def get_parser():
    """Parse command-line arguments"""
    parser = Parser(description='Train neural network to classify poker hands')
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=1000,
                        help='# of training iterations (default: 1000)')
    parser.add_argument('-hu', '--hidden', type=int, nargs='?', default=10, 
                        help='# of hidden units (default: 10)')
    parser.add_argument('-m', '--method', type=str, nargs='?', default='rprop',
                        help='training method (default: rprop)')
    parser.add_argument('-nt', '--num-testing', type=int, nargs='?', default='25000',
                        help='# of testing inputs (default: 25000')
    parser.add_argument('-v', '--verbose', help='print status messages',
                        action='store_true')
    return parser


def train(args, training_ds):
    """Build and train feed-forward neural network

       Keyword arguments:
       args -- program arguments (dict)
       training_ds -- suit, ranks, and target hands (SupervisedDataSet)
    """
    # Build a feed-forward network with x hidden units
    if args['verbose']:
        print('\nBuilding network...')
    ff_network = buildNetwork(training_ds.indim, args['hidden'],
                              training_ds.outdim)
    if args['verbose']:
        print('Network built.')

    # Train using user-specified method and training data for n epochs
    if args['verbose']:
        print('\nTraining network for {0} epochs...'.format(args['epochs']))
    trainer = TRAIN_METHODS[args['method']](ff_network, dataset=training_ds,
                                            verbose=args['verbose'])

    try:
        trainer.trainEpochs(args['epochs'])
    except (KeyboardInterrupt, EOFError):
        pass

    return trainer


def evaluate(args, trainer, training_ds, testing_ds):
    """Use the trainer to evaluate the network on the training and test data"""
    if args['verbose']:
        print('\nEvaluating the network...')
    print('Total epochs: %4d' % trainer.totalepochs)
    training_result = percentError(trainer.testOnClassData(training_ds),
                                   training_ds)
    print('Training error: %5.2f%%' % training_result)
    testing_result = percentError(trainer.testOnClassData(testing_ds),
                                  testing_ds)
    print('Testing error: %5.2f%%' % testing_result)


def command_line_runner():
    """Handle command-line interaction"""
    parser = get_parser()
    args = vars(parser.parse_args())

    # Load training and test data
    training_ds, testing_ds = load_data(args)

    # Build and train feed-forward neural network
    trainer = train(args, training_ds)

    # Use the trainer to evaluate the network on the training and test data
    evaluate(args, trainer, training_ds, testing_ds)


if __name__ == '__main__':
    command_line_runner()
