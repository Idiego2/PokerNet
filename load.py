"""Load and preprocess poker testing and training datasets"""

from __future__ import print_function
from itertools import izip

from pybrain.datasets import SupervisedDataSet


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
                         for _ in xrange(args['num_testing'])]

    # Separate target column data and perform bit vector transformation
    testing_data, target_data = preprocess_load_data(testing_lines)

    # Build SupervisedDataSet by adding data samples
    return build_dataset(args, testing_data, target_data)


def load_data(args):
    """Load training and test data"""
    return load_training_data(args), load_testing_data(args)
