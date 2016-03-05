"""PokerNet simulation suite"""

from train import run_simulation, get_parser


def run_simulations():
    """Run ANN training simulations with various methods and parameters"""
    # Initialize args with program defaults
    init_args = vars(get_parser().parse_args())

    # Table 1: Training results for different neuron numbers
    args = init_args.copy()
    args.update({'method': 'gdm',
            'activation': 'purelin',
            'max_epoch': 1000,
            'learning_rate': 0.02})
    for i in (10, 30, 50):
        args['hidden_neurons'] = i
        run_simulation(args)


    # Table 2: Effect of the change in learning rate value
    args = init_args.copy()
    args.update({'method': 'gdm',
            'activation': 'purelin',
            'max_epoch': 1000,
            'learning_rate': 0.2})
    for i in (10, 30, 50):
        args['hidden_neurons'] = i
        run_simulation(args)


    # Table 3: Effect of different epoch limits for 10 neurons
    args = init_args.copy()
    args.update({'method': 'gdm',
            'activation': 'purelin',
            'learning_rate': 0.2,
            'hidden_neurons': 10})
    for i in (1000, 2000, 3000):
        args['epochs'] = i
        run_simulation(args)


    # Table 4: Effect of different epoch limits for 30 neurons
    args = init_args.copy()
    args.update({'method': 'gdm',
            'activation': 'purelin',
            'learning_rate': 0.2,
            'hidden_neurons': 30})
    for i in (1000, 2000, 3000):
        args['epochs'] = i
        run_simulation(args)


    # Table 5: Effect of different epoch limits for 50 neurons
    args = init_args.copy()
    args.update({'method': 'gdm',
            'activation': 'purelin',
            'learning_rate': 0.2,
            'hidden_neurons': 50})
    for i in (1000, 2000, 3000):
        args['epochs'] = i
        run_simulation(args)


    # Table 6: comparison of three methods for 10 neurons
    args = init_args.copy()
    args.update({'activation': 'purelin',
            'max_epoch': 1000,
            'hidden_neurons': 10})
    for mthd in ('gdm', 'scg', 'rp'):
        args['method'] = mthd

        for lrn_rate in (0.2, 0.02):
            args['learning_rate'] = lrn_rate
            run_simulation(args)


    # Table 7: Effect of transfer function for resilient back-propagation method for 1000 iteration
    args = init_args.copy()
    args.update({'method': 'rp',
            'max_epoch': 1000,
            'hidden_neurons': 10})
    for n_num in (30, 50):
        args['hidden_neurons'] = n_num

        for act_fn in ('purelin', 'tansig'):
            args['activation'] = act_fn

            for lrn_rate in (0.2, 0.02):
                args['learning_rate'] = lrn_rate
                run_simulation(args)


    # Table 8: Effect of transfer function for resilient back-propagation method for 2000 iteration
    args = init_args.copy()
    args.update({'method': 'rp',
            'max_epoch': 2000,
            'hidden_neurons': 10})
    for n_num in (10, 30, 50):
        args['hidden_neurons'] = n_num

        for act_fn in ('purelin', 'tansig'):
            args['activation'] = act_fn

            for lrn_rate in (0.2, 0.02):
                args['learning_rate'] = lrn_rate
                run_simulation(args)


    # Table 9: Effect of transfer function for scaled conjugate gradient method
    args = init_args.copy()
    args.update({'method': 'scg',
            'max_epoch': 1000,
            'hidden_neurons': 10})
    for act_fn in ('purelin', 'tansig'):
        args['activation'] = act_fn

        for lrn_rate in (0.2, 0.02):
            args['learning_rate'] = lrn_rate
            run_simulation(args)


    # Table 10: Effect of transfer function for scaled conjugate gradient method
    # Vary validation limit (what is that?)
    args = init_args.copy()
    args.update({'method': 'scg',
            'hidden_neurons': 50})
    validation_limit = 100
    # set validation limit to 100 here
    args['max_epoch'] = 1000
    args['learning_rate'] = 0.2
    run_simulation(args)

    # set validation limit to 1000 here
    validation_limit = 1000
    args['max_epoch'] = 1000
    args['learning_rate'] = 0.2
    run_simulation(args)

    args['max_epoch'] = 5000
    for lrn_rate in (0.2, 0.02):
        args['learning_rate'] = lrn_rate
        run_simulation(args)

    # Table 11: Test results for various networks
    args = init_args.copy()




    

            


if __name__ == '__main__':
    run_simulations()
