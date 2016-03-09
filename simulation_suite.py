#!/usr/bin/env python
"""PokerNet simulation suite"""

from __future__ import print_function
from multiprocessing import Pool

from train import run_simulation, get_parser


# Initialize args with program defaults
INIT_ARGS = vars(get_parser().parse_args())


def table_one():
    """Table 1: Training results for different neuron numbers"""
    simulation_num = 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'learning_rate', 'max_epochs',
                    'activation', 'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'activation': 'purelin',
                 'max_epochs': 100,
                 'learning_rate': 0.02})
    for i in (10, 30, 50):
        args['hidden_neurons'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)


def table_two():
    """Table 2: Effect of the change in learning rate value"""
    simulation_num = 2
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'activation', 'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'activation': 'purelin',
                 'max_epochs': 100,
                 'learning_rate': 0.2})
    for i in (10, 30, 50):
        args['hidden_neurons'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)


def table_three():
    """Table 3: Effect of different epoch limits for 10 neurons"""
    simulation_num = 3
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'activation': 'purelin',
                 'learning_rate': 0.2,
                 'hidden_neurons': 10})
    for i in (100, 200, 300):
        args['epochs'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)


def table_four():
    """Table 4: Effect of different epoch limits for 30 neurons"""
    simulation_num = 4
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'activation': 'purelin',
                 'learning_rate': 0.2,
                 'hidden_neurons': 30})
    for i in (100, 200, 300):
        args['epochs'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)


def table_five():
    """Table 5: Effect of different epoch limits for 50 neurons"""
    simulation_num = 5
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'activation': 'purelin',
                 'learning_rate': 0.2,
                 'hidden_neurons': 50})
    for i in (100, 200, 300):
        args['epochs'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)


def table_six():
    """Table 6: Effect of transfer function for resilient back-propagation method for 100 iteration"""
    simulation_num = 6
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'max_epochs': 100,
                 'hidden_neurons': 10})
    for n_num in (30, 50):
        args['hidden_neurons'] = n_num

        for act_fn in ('purelin', 'tansig'):
            args['activation'] = act_fn

            for lrn_rate in (0.2, 0.02):
                args['learning_rate'] = lrn_rate
                run_simulation(args, sim_num=simulation_num, header=table_header)


def table_seven():
    """Table 7: Effect of transfer function for resilient back-propagation method for 200 iteration"""
    simulation_num = 7
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'max_epochs': 200,
                 'hidden_neurons': 10})
    for n_num in (10, 30, 50):
        args['hidden_neurons'] = n_num

        for act_fn in ('purelin', 'tansig'):
            args['activation'] = act_fn

            for lrn_rate in (0.2, 0.02):
                args['learning_rate'] = lrn_rate
                run_simulation(args, sim_num=simulation_num, header=table_header)


def table_eight():
    """Table 8: Take the best results for various learning rates"""
    simulation_num = 8
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'learning_rate', 'max_epochs',
                    'hits', 'mse']

    args = INIT_ARGS.copy()
    args.update({'method': 'rp',
                 'hidden_neurons': 50,
                 'max_epochs': 200})
    for lrn_rate in (0.02, 0.2):
        args['learning_rate'] = lrn_rate
        run_simulation(args, sim_num=simulation_num, header=table_header)


def run_simulations():
    """Run ANN training simulations with various methods and parameters"""
    # Run processes asynchronously
    pool = Pool(processes=8)

    p1 = pool.apply_async(table_one())
    p2 = pool.apply_async(table_two())
    p3 = pool.apply_async(table_three())
    p4 = pool.apply_async(table_four())
    p5 = pool.apply_async(table_five())
    p6 = pool.apply_async(table_six())
    p7 = pool.apply_async(table_seven())

    # Run three times to later take best results
    p8 = pool.apply_async(table_eight())
    p8 = pool.apply_async(table_eight())
    p8 = pool.apply_async(table_eight())

    pool.close()
    pool.join()


if __name__ == '__main__':
    run_simulations()
