from joblib import Parallel, delayed
from tqdm import tqdm
import os
import time
import itertools

from main import AdversarialCompression
from config import *


class Experiments:
    def __init__(self, args):
        self.args = args

    def run_search_gaussian(self, index, train_corruption_strength):
        GPU_IDS = [0,1,2,3,4,5,6,7]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        DEVICE_ID = GPU_IDS[index % len(GPU_IDS)]

        self.args['cuda_device'] = str(DEVICE_ID)
        self.args['train_corruption_strength'] = train_corruption_strength

        self.gtc(loader='val_loader')

    def line_search_gaussian(self):
        train_corruption_strengths = [1,2,3]

        num_processes = len(train_corruption_strengths)

        Parallel(n_jobs=num_processes)(
            delayed(self.run_search_gaussian)(index, train_corruption_strength)
            for (index, (train_corruption_strength)) in tqdm(enumerate(train_corruption_strengths)))

    def run_search_epsilon(self, index, epsilon):
        GPU_IDS = [0,1,2,3,4,5,6, 7]

        # to avoid all args being logged at the same time
        time.sleep(int(index / len(GPU_IDS)) * len(GPU_IDS) + GPU_IDS[index % len(GPU_IDS)])

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        DEVICE_ID = GPU_IDS[index % len(GPU_IDS)]

        self.args['cuda_device'] = str(DEVICE_ID)
        self.args['train_epsilon'] = epsilon

        self.at(loader='val_loader')

    def line_search_epsilon(self):
        epsilons = [24.0,26.0,28.0,30.0]

        num_processes = len(epsilons)

        Parallel(n_jobs=num_processes)(
            delayed(self.run_search_epsilon)(index, epsilon)
            for (index, epsilon) in tqdm(enumerate(epsilons)))

    def run_search_sparsity(self, index, sparsity):
        GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

        # to avoid all args being logged at the same time
        time.sleep(int(index / len(GPU_IDS)) * len(GPU_IDS) + GPU_IDS[index % len(GPU_IDS)])

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        DEVICE_ID = GPU_IDS[index % len(GPU_IDS)]

        self.args['cuda_device'] = str(DEVICE_ID)
        self.args['sparsity'] = sparsity

        self.atc_ortho(loader='test_loader')

    def line_search_sparsity(self):
        sparsity = [0.80,0.85,0.90,0.95]

        num_processes = len(sparsity)

        Parallel(n_jobs=num_processes)(
            delayed(self.run_search_sparsity)(index, s)
            for (index, s) in tqdm(enumerate(sparsity)))



    def run_search(self, index, ortho_lambda, gamma_lambda):
        GPU_IDS = [2,5]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        DEVICE_ID = GPU_IDS[index % len(GPU_IDS)]
        self.args['cuda_device'] = str(DEVICE_ID)

        self.args['ortho_lambda'] = ortho_lambda
        self.args['gamma_lambda'] = gamma_lambda

        self.atc_ortho(loader='val_loader')

    def grid_search(self):
        ortho_lambdas = [0.001]
        gamma_lambdas = [0.0001, 0.00001]

        # get every combination of ortho lambdas and gamma lambdas
        parameter_combinations = list(itertools.product(ortho_lambdas, gamma_lambdas))

        num_processes = len(parameter_combinations)

        Parallel(n_jobs=num_processes)(
            delayed(self.run_search)(index, ortho_lambda, gamma_lambda)
            for (index, (ortho_lambda, gamma_lambda)) in tqdm(enumerate(parameter_combinations)))

    def bt(self):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # no compression
        self.args['gamma_lambda'] = 0

        # no noise removal
        self.args['noise_removal'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model)
        # ac.visualize_eigenvalues(model)

    def btc(self):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # assert some compression
        assert self.args['gamma_lambda'] != 0

        # no noise removal
        self.args['noise_removal'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_small_model()
        ac.print_metrics(model, type='small')

    def btnr(self):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # no compression
        self.args['gamma_lambda'] = 0

        # noise removal
        self.args['noise_removal'] = True

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model)
        # ac.visualize_eigenvalues(model)

    def btcnr(self):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # assert some compression
        assert self.args['gamma_lambda'] != 0

        # no noise removal
        self.args['noise_removal'] = True

        ac = AdversarialCompression(self.args)
        model = ac.get_small_model()
        ac.print_metrics(model, type='small')

    def gt(self):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # some noise corruption
        self.args['gaussian_training'] = True
        assert self.args['train_corruption_strength'] in [1,2,3,4,5]

        # no compression
        self.args['gamma_lambda'] = 0

        # no noise removal
        self.args['noise_removal'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model)

    def gtc(self, loader='test_loader'):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # some noise corruption
        self.args['gaussian_training'] = True
        assert self.args['train_corruption_strength'] in [1, 2, 3, 4, 5]

        # assert some compression
        assert self.args['gamma_lambda'] != 0

        # no noise removal
        self.args['noise_removal'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_small_model()
        ac.print_metrics(model, type='small', loader=loader)

    def at(self, loader='test_loader'):
        # no adversarial robustness
        self.args['robust_training'] = True
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # no compression
        self.args['gamma_lambda'] = 0

        # no noise removal
        self.args['noise_removal'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model,loader=loader)

    def atc(self):
        # only adversarial robustness
        self.args['robust_training'] = True
        self.args['orthogonality'] = False
        self.args['spectral_normalization'] = False
        self.args['ortho_lambda'] = 0

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # assert some compression
        assert self.args['gamma_lambda'] != 0

        # no noise removal
        self.args['noise_removal'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_small_model()
        ac.print_metrics(model, type='small')

    def bt_ortho(self):
        # no adversarial robustness
        self.args['robust_training'] = False
        self.args['orthogonality'] = True
        self.args['spectral_normalization'] = False

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # no compression
        self.args['gamma_lambda'] = 0

        # no noise removal
        self.args['noise_removal'] = False

        # assert some ortho lambda
        assert self.args['ortho_lambda'] != 0

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model)

        # ac.visualize_eigenvalues(model)
        # tf_model = covert_to_tensorflow(self.args, model)
        # print(tf_model)

    def at_ortho(self):
        # no adversarial robustness
        self.args['robust_training'] = True
        self.args['orthogonality'] = True
        self.args['spectral_normalization'] = False

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # no compression
        self.args['gamma_lambda'] = 0

        # no noise removal
        self.args['noise_removal'] = False

        # assert some ortho lambda
        assert self.args['ortho_lambda'] != 0

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model)

        # ac.visualize_eigenvalues(model)
        # tf_model = covert_to_tensorflow(self.args, model)
        # print(tf_model)

    def atc_ortho(self, loader='test_loader'):
        # no adversarial robustness
        self.args['robust_training'] = True
        self.args['orthogonality'] = True
        self.args['spectral_normalization'] = False

        # no noise corruption
        self.args['gaussian_training'] = False
        self.args['train_corruption_strength'] = 1
        self.args['test_corruption'] = 'none'
        self.args['test_corruption_strength'] = 1

        # no noise removal
        self.args['noise_removal'] = False

        # assert some compression + some ortho
        assert self.args['gamma_lambda'] != 0
        assert self.args['ortho_lambda'] != 0

        ac = AdversarialCompression(self.args)
        model = ac.get_small_model()
        ac.print_metrics(model, type='small', loader=loader)
        # ac.visualize_eigenvalues(model)


def main():
    args = args_sors_physionet

    args['enable_logging'] = True
    args['enable_saving'] = True
    args['get_hypnogram'] = False # ensure if this is True then enable logging is False

    args['seed'] = 1

    args['cuda_device'] = '0'


    args['logging_comment'] = 'Sors_Physionet'

    ex = Experiments(args)

    ex.bt() # evaluates Sors model
    # ex.btc() # evaluates the Liu model
    # ex.btcnr() # evaluates the Blanco model
    # ex.gtc() # evaluates the Ford model
    # ex.atc() # evaluates the REST(A) model
    # ex.atc_ortho() # evaluates the REST(A+S) model


if __name__ == '__main__':
    main()


