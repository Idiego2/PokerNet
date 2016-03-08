#!/usr/bin/env python
"""PokerNet simulation suite"""

from train import run_simulation, get_parser


def run_simulations():
    """Run ANN training simulations with various methods and parameters"""
    # Initialize args with program defaults
    init_args = vars(get_parser().parse_args())

    # Table 1: Training results for different neuron numbers
    simulation_num = 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'learning_rate', 'max_epochs',
                    'activation', 'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'gdm',
                 'activation': 'purelin',
                 'max_epochs': 1000,
                 'learning_rate': 0.02})
    for i in (10, 30, 50):
        args['hidden_neurons'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 2: Effect of the change in learning rate value
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'activation', 'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'gdm',
                 'activation': 'purelin',
                 'max_epochs': 1000,
                 'learning_rate': 0.2})
    for i in (10, 30, 50):
        args['hidden_neurons'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 3: Effect of different epoch limits for 10 neurons
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'gdm',
                 'activation': 'purelin',
                 'learning_rate': 0.2,
                 'hidden_neurons': 10})
    for i in (1000, 2000, 3000):
        args['epochs'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 4: Effect of different epoch limits for 30 neurons
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'gdm',
                 'activation': 'purelin',
                 'learning_rate': 0.2,
                 'hidden_neurons': 30})
    for i in (1000, 2000, 3000):
        args['epochs'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 5: Effect of different epoch limits for 50 neurons
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'gdm',
                 'activation': 'purelin',
                 'learning_rate': 0.2,
                 'hidden_neurons': 50})
    for i in (1000, 2000, 3000):
        args['epochs'] = i
        run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 6: comparison of three methods for 10 neurons
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'method', 'learning_rate', 'max_epochs',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'activation': 'purelin',
                 'max_epochs': 1000,
                 'hidden_neurons': 10})
    for mthd in ('gdm', 'scg', 'rp'):
        args['method'] = mthd

        for lrn_rate in (0.2, 0.02):
            args['learning_rate'] = lrn_rate
            run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 7: Effect of transfer function for resilient back-propagation method for 1000 iteration
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'method', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'rp',
                 'max_epochs': 1000,
                 'hidden_neurons': 10})
    for n_num in (30, 50):
        args['hidden_neurons'] = n_num

        for act_fn in ('purelin', 'tansig'):
            args['activation'] = act_fn

            for lrn_rate in (0.2, 0.02):
                args['learning_rate'] = lrn_rate
                run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 8: Effect of transfer function for resilient back-propagation method for 2000 iteration
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'method', 'max_epochs', 'learning_rate',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'rp',
                 'max_epochs': 2000,
                 'hidden_neurons': 10})
    for n_num in (10, 30, 50):
        args['hidden_neurons'] = n_num

        for act_fn in ('purelin', 'tansig'):
            args['activation'] = act_fn

            for lrn_rate in (0.2, 0.02):
                args['learning_rate'] = lrn_rate
                run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 9: Effect of transfer function for scaled conjugate gradient method
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'method', 'learning_rate', 'max_epochs',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'method': 'scg',
                 'max_epochs': 1000,
                 'hidden_neurons': 10})
    for act_fn in ('purelin', 'tansig'):
        args['activation'] = act_fn

        for lrn_rate in (0.2, 0.02):
            args['learning_rate'] = lrn_rate
            run_simulation(args, sim_num=simulation_num, header=table_header)

    return  # Returning here because have yet to figure out validation limit for table 10
            # as well as the best train results for table 11

    # Table 10: Effect of transfer function for scaled conjugate gradient method
    # Vary validation limit (what is that?)... also need train stop criteria
    #                                           (validation limit reached or max epoch reached)
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'validation_limit', 'learning_rate', 'max_epochs',
                    'hits', 'stop_criteria']

    args = init_args.copy()
    args.update({'method': 'scg',
                 'hidden_neurons': 50})
    validation_limit = 100
    # set validation limit to 100 here
    args['max_epochs'] = 1000
    args['learning_rate'] = 0.2
    run_simulation(args, sim_num=simulation_num, header=table_header)

    # set validation limit to 1000 here
    validation_limit = 1000
    args['max_epochs'] = 1000
    args['learning_rate'] = 0.2
    run_simulation(args, sim_num=simulation_num, header=table_header)

    args['max_epochs'] = 5000
    for lrn_rate in (0.2, 0.02):
        args['learning_rate'] = lrn_rate
        run_simulation(args, sim_num=simulation_num, header=table_header)

    # Table 11: Test results for various networks
    # takes the best train result. how many runs to examine?
    simulation_num += 1
    print('\n** Running simulation {} **\n'.format(simulation_num))
    table_header = ['hidden_neurons', 'method', 'learning_rate', 'max_epochs',
                    'hits', 'mse']

    args = init_args.copy()
    args.update({'hidden_neurons': 50,
                 'max_epochs': 2000})
    for lrn_rate in (0.02, 0.2):
        args['learning_rate'] = lrn_rate
        for mthd in ('scg', 'rp'):
            args['method'] = mthd
            run_simulation(args, sim_num=simulation_num, header=table_header)


if __name__ == '__main__':
    run_simulations()
