args_sors_physionet = {
    'no_cuda': False,
    'cuda_device': '0',
    'seed': 1,
    'model': 'sorsnet',
    'signal_type': 'fz',
    'dataset': 'physionet',
    'classes': ['W', 'N1', 'N2', 'N3', 'R'],
    # training params
    'val_ratio': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv robustness params
    'robust_training': True,
    'train_epsilon': 10.0,
    'test_epsilon': 6.0,
    'step_size': 0.01,
    'nb_iter': 10,
    'orthogonality': False,
    'ortho_lambda': 0.003,
    'spectral_normalization': False,
    #######
    # compression params
    'sparsity': 0.8,
    'min_layerwise_sparsity': 0.1,
    'gamma_lambda': 0.00001,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'get_hypnogram':False,
    'logging_comment': 'SleepEDF Epsilon Sampled',
    'logging_run': 1,
    #######
    'verbose': 2,
    # corruption params
    'gaussian_training': False,
    'train_corruption_strength': 2,
    'test_corruption': 'none',
    'test_corruption_strength': 1,
    # noise reduction params
    'noise_removal': False,
    'l_min': 0.5,
    'l_max': 40.0
}


args_sors_shhs_hypnogram = {
    'no_cuda': False,
    'cuda_device': '0',
    'seed': 1,
    'model': 'sorsnet',
    'signal_type': 'fz',
    'dataset': 'shhs_hypnogram',
    'classes': ['W', 'N1', 'N2', 'N3', 'R'],
    # training params
    'val_ratio': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv robustness params
    'robust_training': True,
    'train_epsilon': 10.0,
    'test_epsilon': 6.0,
    'step_size': 0.01,
    'nb_iter': 10,
    'orthogonality': False,
    'ortho_lambda': 0.003,
    'spectral_normalization': False,
    #######
    # compression params
    'sparsity': 0.8,
    'min_layerwise_sparsity': 0.1,
    'gamma_lambda': 0.00001,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'get_hypnogram':False,
    'logging_comment': 'SleepEDF Epsilon Sampled',
    'logging_run': 1,
    #######
    'verbose': 2,
    # corruption params
    'gaussian_training': False,
    'train_corruption_strength': 2,
    'test_corruption': 'none',
    'test_corruption_strength': 1,
    # noise reduction params
    'noise_removal': False,
    'l_min': 0.5,
    'l_max': 40.0
}

args_sors_shhs = {
    'no_cuda': False,
    'cuda_device': '0',
    'seed': 1,
    'model': 'sorsnet',
    'signal_type': 'fz',
    'dataset': 'shhs',
    'classes': ['W', 'N1', 'N2', 'N3', 'R'],
    # training params
    'val_ratio': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv robustness params
    'robust_training': True,
    'train_epsilon': 10.0,
    'test_epsilon': 6.0,
    'step_size': 0.01,
    'nb_iter': 10,
    'orthogonality': False,
    'ortho_lambda': 0.003,
    'spectral_normalization': False,
    #######
    # compression params
    'sparsity': 0.8,
    'min_layerwise_sparsity': 0.1,
    'gamma_lambda': 0.00001,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'get_hypnogram':False,
    'logging_comment': 'SleepEDF Epsilon Sampled',
    'logging_run': 1,
    #######
    'verbose': 2,
    # corruption params
    'gaussian_training': False,
    'train_corruption_strength': 2,
    'test_corruption': 'none',
    'test_corruption_strength': 1,
    # noise reduction params
    'noise_removal': False,
    'l_min': 0.5,
    'l_max': 40.0
}

args_residual_physionet = {
    'no_cuda': False,
    'cuda_device': '0',
    'seed': 1,
    'model': 'deep_residual',
    'signal_type': 'fz',
    'dataset': 'physionet',
    'classes': ['W', 'N1', 'N2', 'N3', 'R'],
    # training params
    'val_ratio': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv robustness params
    'robust_training': True,
    'train_epsilon': 10.0,
    'test_epsilon': 6.0,
    'step_size': 0.01,
    'nb_iter': 10,
    'orthogonality': False,
    'ortho_lambda': 0.003,
    'spectral_normalization': False,
    #######
    # compression params
    'sparsity': 0.8,
    'min_layerwise_sparsity': 0.1,
    'gamma_lambda': 0.00001,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'get_hypnogram':False,
    'logging_comment': 'SleepEDF Epsilon Sampled',
    'logging_run': 1,
    #######
    'verbose': 2,
    # corruption params
    'gaussian_training': False,
    'train_corruption_strength': 2,
    'test_corruption': 'none',
    'test_corruption_strength': 1,
    # noise reduction params
    'noise_removal': False,
    'l_min': 0.5,
    'l_max': 40.0
}

args_biswal_physionet = {

    'no_cuda': False,
    'cuda_device': '0',
    'seed': 1,
    'model': 'biswalnet',
    'signal_type': 'fz',
    'dataset': 'physionet',
    'classes': ['W', 'N1', 'N2', 'N3', 'R'],
    # training params
    'val_ratio': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv robustness params
    'robust_training': True,
    'train_epsilon': 10.0,
    'test_epsilon': 6.0,
    'step_size': 0.01,
    'nb_iter': 10,
    'orthogonality': False,
    'ortho_lambda': 0.003,
    'spectral_normalization': False,
    #######
    # compression params
    'sparsity': 0.8,
    'min_layerwise_sparsity': 0.1,
    'gamma_lambda': 0.00001,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'get_hypnogram':False,
    'logging_comment': 'SleepEDF Epsilon Sampled',
    'logging_run': 1,
    #######
    'verbose': 2,
    # corruption params
    'gaussian_training': False,
    'train_corruption_strength': 2,
    'test_corruption': 'none',
    'test_corruption_strength': 1,
    # noise reduction params
    'noise_removal': False,
    'l_min': 0.5,
    'l_max': 40.0
}

args_residual_shhs = {
    'no_cuda': False,
    'cuda_device': '0',
    'seed': 1,
    'model': 'deep_residual',
    'signal_type': 'fz',
    'dataset': 'shhs',
    'classes': ['W', 'N1', 'N2', 'N3', 'R'],
    # training params
    'val_ratio': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv robustness params
    'robust_training': True,
    'train_epsilon': 10.0,
    'test_epsilon': 6.0,
    'step_size': 0.01,
    'nb_iter': 10,
    'orthogonality': False,
    'ortho_lambda': 0.003,
    'spectral_normalization': False,
    #######
    # compression params
    'sparsity': 0.8,
    'min_layerwise_sparsity': 0.1,
    'gamma_lambda': 0.00001,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'get_hypnogram':False,
    'logging_comment': 'SleepEDF Epsilon Sampled',
    'logging_run': 1,
    #######
    'verbose': 2,
    # corruption params
    'gaussian_training': False,
    'train_corruption_strength': 2,
    'test_corruption': 'none',
    'test_corruption_strength': 1,
    # noise reduction params
    'noise_removal': False,
    'l_min': 0.5,
    'l_max': 40.0
}
