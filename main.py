import os
import collections
import torch
import copy
import numpy as np
from tensorboardX import SummaryWriter


from utils import set_random_seed, test, load_model, weights_init
from utils import make_idx_dict, get_layer_from_idx
from models.model_builder import Model_Builder
from spectral_compression import SpectralCompression
from metrics import print_model_param_nums, print_model_param_flops

from config import *


class AdversarialCompression:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_device']
        use_cuda = not args['no_cuda'] and torch.cuda.is_available()
        args['device'] = torch.device("cuda" if use_cuda else "cpu")

        prune_layers = {
           'sleepnet_spectral': [0, 4, 7, 11, 14],
           'sorsnet': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
           'biswalnet': [0, 7, 13, 19, 26, 32, 38, 44, 51, 57, 63, 69, 75, 81, 88, 94, 100],
           'deep_residual': [3, 9, 18, 24, 33, 39, 48, 54, 63, 69, 78, 84, 93, 99, 108, 114]
        }

        conv_feature_size = {
            'sleepnet_spectral': 1,  # TODO : needs to change to a dict
            'sorsnet': {
                'physionet': 3,
                'shhs': 4},
            'biswalnet': {
                'physionet': 1,
                'shhs': 1},
            'deep_residual': {
                'physionet': 11,
                'shhs': 14},  # TODO
        }

        self.slimming_params = {}
        self.spectral_params = {}

        # get current working dir
        args['dir'] = os.getcwd()

        # set logging params
        args['run'] = args['logging_run']
        args['log_dir'] = args['dir'] + '/logs{}/{}_{}'.format(args['seed'], args['dataset'], args['model'])

        # set pruning params depending on model type
        args['prune_layers'] = prune_layers[args['model']]
        args['conv_feature_size'] = conv_feature_size[args['model']][args['dataset']]

        # define paths for model persistence
        args['chkpnt_dir'] = args['dir'] + '/checkpoints{}/{}/{}/'.format(args['seed'],args['dataset'], args['model'])

        # decide folder structure
        if args['gaussian_training']:
            params = 'gt_{}/cs_{}_ortho_{}/ol_{}_sp_{}_gl_{}/'.format(args['gaussian_training'], args['train_corruption_strength'],
                                                                       args['orthogonality'], args['ortho_lambda'],
                                                                       args['sparsity'], args['gamma_lambda'])
        elif args['orthogonality']:
            params = 'rt_{}/eps_{}_ortho_{}/ol_{}_sp_{}_gl_{}/'.format(args['robust_training'], args['train_epsilon'],
                                                                         args['orthogonality'], args['ortho_lambda'],
                                                                         args['sparsity'], args['gamma_lambda'])
        elif args['spectral_normalization']:
            params = 'rt_{}/eps_{}_spn_{}/sp_{}_gl_{}/'.format(args['robust_training'], args['train_epsilon'],
                                                                         args['spectral_normalization'],
                                                                         args['sparsity'], args['gamma_lambda'])
        else:
            params = 'rt_{}/eps_{}_sp_{}_gl_{}/'.format(args['robust_training'], args['train_epsilon'],
                                                            args['sparsity'], args['gamma_lambda'])

        os.makedirs(args['chkpnt_dir']+'/'+'/'.join(params.split('/')[:-1])+'/', exist_ok=True)
        args['full_model_path'] = args['chkpnt_dir'] + params + 'spec_large.pt'
        args['small_model_path'] = args['chkpnt_dir'] + params + 'spec_small.pt'

        # paths for model def and data
        args['model_dir'] = args['dir'] + '/models/'
        args['data_dir'] = args['dir'] + '/data'

        self.args = args

    def set_params(self, model_builder):
        train_loader, val_loader, test_loader = model_builder.get_loaders()

        if self.args['dataset'] == 'physionet' or self.args['dataset'] == 'shhs':
            clip_min, clip_max = model_builder.get_bounds()
        else:
            clip_min, clip_max = 0, 1

        self.slimming_params = {
                                    'prune_layers': self.args['prune_layers'],
                                    'gamma_lambda': self.args['gamma_lambda'],
                                    'conv_feature_size': self.args['conv_feature_size'],
                                    'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader,
                                }

        self.spectral_params = {
                                    # compression params
                                    'prune_layers': self.args['prune_layers'],
                                    'gamma_lambda': self.args['gamma_lambda'],
                                    'conv_feature_size': self.args['conv_feature_size'],
                                    ####
                                    # adv robustness params
                                    'orthogonality': self.args['orthogonality'],
                                    'ortho_lambda': self.args['ortho_lambda'],
                                    'robust_training': self.args['robust_training'],
                                    'attack': 'pgdInf',
                                    'train_epsilon': self.args['train_epsilon'],
                                    'test_epsilon': self.args['test_epsilon'],
                                    'nb_iter': self.args['nb_iter'],
                                    'eps_iter': self.args['step_size'],
                                    'clip_min': clip_min,
                                    'clip_max': clip_max,
                                    ####
                                    'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader,
                                    'gaussian_training': self.args['gaussian_training']
                                }

    def get_full_model(self, index=0):
        # set seed for reproducability
        set_random_seed(self.args['seed'])

        if os.path.exists(self.args['full_model_path']):
            full_model = load_model(self.args['full_model_path'])
        else:
            model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args)
            self.set_params(model_builder)
            full_model = model_builder.get_model()

            spc = SpectralCompression(self.args, self.spectral_params)

            # log args
            if self.args['enable_logging'] and (not index):
                args_log = {k: v for (k, v) in self.args.items() if
                            not (isinstance(v, str) or isinstance(v, list) or isinstance(v, torch.device))}
                writer = SummaryWriter(self.args['log_dir'])
                writer.add_scalars('{}_large_{}/args'.format(self.args['logging_comment'], self.args['run']), args_log, 1)

            if self.args['verbose'] > 0: print('\ttraining large model')
            full_model = spc.train(full_model)

        return full_model

    def get_small_model(self, index=0):
        # set seed for reproducability
        set_random_seed(self.args['seed'])

        if os.path.exists(self.args['small_model_path']):
            small_adv_model = load_model(self.args['small_model_path'])
        else:
            model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args)
            self.set_params(model_builder)
            spc = SpectralCompression(self.args, self.spectral_params)

            # log args
            if self.args['enable_logging'] and (not index):
                args_log = {k: v for (k, v) in self.args.items() if
                            not (isinstance(v, str) or isinstance(v, list) or isinstance(v, torch.device))}
                writer = SummaryWriter(self.args['log_dir'])
                writer.add_scalars('{}_small_{}/args'.format(self.args['logging_comment'], self.args['run']), args_log, 1)

            if self.args['verbose'] > 0: print('\tobtaining full model')
            full_adv_model = self.get_full_model(index)
            if self.args['verbose'] > 0: print('\tpruning model')
            pruned_model = spc.prune_model(full_adv_model)
            if self.args['verbose'] > 0: print('\tretraining pruned_model')
            small_adv_model = spc.train(pruned_model, retrain=True)

        return small_adv_model

    def robustify(self, model):
        #### need to look at full model path
        model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args)
        self.set_params(model_builder)
        spc = SpectralCompression(self.args, self.spectral_params)

        robust_model = spc.train(model)

    def reinit(self, model):
        model.apply(weights_init)
        if self.args['device'] == torch.device('cuda'):
            model = model.cuda()
        return model


    def print_metrics(self, model, type='large', loader='test_loader'):
        # set seed for reproducability
        set_random_seed(self.args['seed'])

        # get no. of params and no. of flops
        input_res = {
            'physionet': [3000, 1],
            'shhs': [3750, 1]
        }
        if self.args['model'] == 'sorsnet':
            input_res['physionet'] = [12000, 1]
            input_res['shhs'] = [15000, 1]

        num_params = print_model_param_nums(model)  # unit of Mega
        num_flops = print_model_param_flops(model.cpu(), input_res=input_res[self.args['dataset']])  # unit of Giga
        if self.args['device'] == torch.device('cuda'):
            model = model.cuda()

        # test each model on all types of noise
        noise_types = {
                       'physionet': ['gaussian_noise', 'shot_noise', 'none'],
                       'shhs': ['gaussian_noise', 'shot_noise', 'none']
                      }
        corruption_strengths = {'physionet': [1, 2, 3],
                                'shhs': [1, 2, 3]}

        noise_ta = collections.defaultdict(dict)
        noise_f1 = collections.defaultdict(dict)
        noise_cfm = collections.defaultdict(dict)
        noise_metrics = collections.defaultdict(dict)
        for noise in noise_types[self.args['dataset']]:
            self.args['test_corruption'] = noise
            for cs in corruption_strengths[self.args['dataset']]:
                print('Testing on {} noise with strength {}'.format(noise, cs))
                self.args['test_corruption_strength'] = cs
                model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args, get_hypnogram=self.args['get_hypnogram'])
                self.set_params(model_builder)
                _, metrics, cfm = test(self.args, model, self.args['device'], self.spectral_params[loader], type=type)
                noise_ta[noise][cs] = metrics['ben_acc']
                noise_f1[noise][cs] = metrics['macro avg']['f1-score']
                noise_cfm[noise][cs] = cfm
                noise_metrics[noise][cs] = metrics

        # test model on all types of epsilon
        epsilons = {'physionet': [2.0, 6.0, 12.0],
                    'shhs': [2.0, 6.0, 12.0]}  # TODO

        adv_ta = collections.defaultdict(dict)
        adv_f1 = collections.defaultdict(dict)
        adv_cfm = collections.defaultdict(dict)
        adv_metrics = collections.defaultdict(dict)
        for epsilon in epsilons[self.args['dataset']]:
            print('Adversarial testing with epsilon {}'.format(epsilon))
            self.args['test_epsilon'] = epsilon
            model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args, get_hypnogram=self.args['get_hypnogram'])
            self.set_params(model_builder)
            spc = SpectralCompression(self.args, self.spectral_params)
            _, _, ben_metrics, a_metrics, a_cfm = spc.test_robustness(model, self.spectral_params['test_loader'], type=type)
            adv_ta[epsilon] = a_metrics['adv_acc']
            adv_f1[epsilon] = a_metrics['macro avg']['f1-score']
            adv_cfm[epsilon] = a_cfm
            adv_metrics[epsilon] = a_metrics

        result_str = 'Size & Flops\n'
        result_str += '\t Size {} M\n'.format(num_params)
        result_str += '\t Flops {} G\n'.format(num_flops)

        result_str += '\n\nNoise accuracy with {}\n'.format(loader)
        for cs in corruption_strengths[self.args['dataset']]:
            result_str += '\t========= corruption strength {} ============\n'.format(cs)
            for noise_type in noise_types[self.args['dataset']]:
                result_str += '\t{} : {:0.2f} (ACC), {:0.2f} (F1)\n'.format(noise_type,noise_ta[noise_type][cs],noise_f1[noise_type][cs])

        result_str += '\n\nAdversarial Accuracy with {}\n'.format(loader)
        for epsilon in epsilons[self.args['dataset']]:
            result_str += '\ttrain_epsilon {:0.2f} test_epsilon {:0.2f} : {:0.2f} (ACC), {:0.2f} (F1)\n'.format(self.args['train_epsilon'], epsilon, adv_ta[epsilon], adv_f1[epsilon])

        result_str += '\n\nNoise confusion matrices with {}\n'.format(loader)
        for cs in corruption_strengths[self.args['dataset']]:
            result_str += '\t========= corruption strength {} ============\n'.format(cs)
            for noise_type in noise_types[self.args['dataset']]:
                result_str += 'confusion matrix for noisy eeg ({} {})\n'.format(noise_type,cs)
                for i, row in enumerate(noise_cfm[noise_type][cs]):
                    result_str += '\t{} : {} {} {} {} {} : pre {:0.2f} rec {:0.2f} f1 {:0.2f}\n'.format(
                                                            self.args['classes'][i], row[0], row[1], row[2], row[3], row[4],
                                                            noise_metrics[noise_type][cs][self.args['classes'][i]]['precision'],
                                                            noise_metrics[noise_type][cs][self.args['classes'][i]]['recall'],
                                                            noise_metrics[noise_type][cs][self.args['classes'][i]]['f1-score'])

        result_str += '\n\nAdversarial confusion matrices with {}\n'.format(loader)
        for epsilon in epsilons[self.args['dataset']]:
            result_str += 'confusion matrix for adv eeg (train_eps : {})\n'.format(epsilon)
            for i, row in enumerate(adv_cfm[epsilon]):
                result_str += '\t{} : {} {} {} {} {} : pre {:0.2f} rec {:0.2f} f1 {:0.2f}\n'.format(
                                                        self.args['classes'][i], row[0], row[1], row[2], row[3], row[4],
                                                        adv_metrics[epsilon][self.args['classes'][i]]['precision'],
                                                        adv_metrics[epsilon][self.args['classes'][i]]['recall'],
                                                        adv_metrics[epsilon][self.args['classes'][i]]['f1-score'])

            # print
        if self.args['verbose'] > 0:
            print(result_str)

        if self.args['enable_logging']:
            # save results to file
            file_path = '{}/{}_{}_{}/result_info_sparsity_{}_train_noisestrength_{}_train_epsilon_{}_ortholambda_{}_gammalambda_{}.txt'.format( self.args['log_dir'],
                                                                                                                         self.args['logging_comment'],
                                                                                                                         type, self.args['run'],
                                                                                                                         self.args['sparsity'],
                                                                                                                         self.args['train_corruption_strength'],
                                                                                                                         self.args['train_epsilon'],
                                                                                                                         self.args['ortho_lambda'],
                                                                                                                         self.args['gamma_lambda'])

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write(result_str)

    def visualize_eigenvalues(self, model):
        # idx_dict maps layer_idx to the layer in t
        ctr, self.idx_dict = make_idx_dict(model, -1, [], {})

        for layer_idx in self.args['prune_layers']:
            layer = get_layer_from_idx(model, copy.deepcopy(self.idx_dict), layer_idx)
            weight = layer.weight.data.cpu().numpy()
            W = np.reshape(weight, [weight.shape[0], np.prod(weight.shape[1:])])

            eval, evec = np.linalg.eig(np.matmul(W.transpose(), W))

            print(eval)


def main():
    args = args_sors_physionet
    ac = AdversarialCompression(args)
    model = ac.get_full_model(0)
    ac.print_metrics(model, type='large')


if __name__ == '__main__':
    main()
