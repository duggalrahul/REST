import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import csv
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal

from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

import mne

from utils import make_idx_dict, get_layer_from_idx, set_layer_to_idx
from utils import test, save_model, print_results
from utils import adjust_learning_rate_cifar10, adjust_learning_rate_mnist, adjust_learning_rate_physionet, adjust_learning_rate_shhs
from utils import gaussian_noise


class SpectralCompression:
    def __init__(self, args, spectral_args):
        self.args = args
        self.spectral_args = spectral_args

        self.train_loader = spectral_args['train_loader']
        self.val_loader = spectral_args['val_loader']
        self.test_loader = spectral_args['test_loader']

        self.prune_layers = spectral_args['prune_layers']
        self.orthogonality = spectral_args['orthogonality']
        self.ortho_lambda = spectral_args['ortho_lambda']
        self.robust_training = spectral_args['robust_training']
        self.gaussian_training = spectral_args['gaussian_training']
        self.gamma_lambda = spectral_args['gamma_lambda']
        self.conv_feature_size = spectral_args['conv_feature_size']

    def compress(self, model):
        if self.args['verbose'] > 0: print('\ttraining with reg')
        model = self.train(model)
        if self.args['verbose'] > 0: print('\tpruning model')
        pruned_model = self.prune_model(model)
        if self.args['verbose'] > 0: print('\tretraining pruned_model')
        pruned_model = self.train(pruned_model, retrain=True)

        return pruned_model

    def init_adversary(self, model, test=False):

        # chose train or test epsilon
        epsilon = self.spectral_args['test_epsilon'] if test else self.spectral_args['train_epsilon']
        uniform_sample = False #True if not test else False

        return LinfPGDAttack(
                            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                            eps=epsilon, nb_iter=self.spectral_args["nb_iter"],
                            eps_iter=self.spectral_args["eps_iter"], rand_init=True, clip_min=self.spectral_args['clip_min'],
                            clip_max=self.spectral_args['clip_max'], targeted=False, uniform_sample=uniform_sample)

    def spectral_init(self, model):
        U = {}
        for l, m in enumerate(model.modules()):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                N = m.weight.shape[0]
                U_l = MultivariateNormal(torch.zeros(N), torch.eye(N)).sample()
                U_l = U_l.cuda() if self.args['device'] == torch.device('cuda') else U_l
                U[l] = U_l

        return U

    def spectral_normalize(self, model, U):

        for l, m in enumerate(model.modules()):

            flag = False

            if isinstance(m, nn.Linear):
                W = m.weight
                flag=1
            elif isinstance(m, nn.Conv2d):
                W = m.weight
                N, C, Kh, Kw = W.shape
                W = torch.reshape(W, (N, C*Kh*Kw))
                flag = 2
            elif isinstance(m, nn.Conv1d):
                W = m.weight
                N, C, L = W.shape
                W = torch.reshape(W, (N, C*L))
                flag=3

            if flag:

                for i in range(1):
                    V_l = torch.matmul(torch.t(W), U[l])
                    V_l = V_l / (torch.norm(V_l) + 1e-12)
                    U_l = torch.matmul(W, V_l)
                    U_l = U_l / (torch.norm(U_l) + 1e-12)
                    U[l].data = U_l

                U_l_t = torch.reshape(U_l, (1, U_l.shape[0]))
                rho_W = torch.matmul(U_l_t, torch.matmul(W, V_l))

                W = W / rho_W

                if flag == 2:
                    W = torch.reshape(W, (N, C, Kh, Kw))
                elif flag == 3:
                    W = torch.reshape(W, (N, C, L))

                m.weight.data = W

        return model, U

    def ortho_reg(self, model):
        reg_loss = 0

        for m in model.modules():

            if isinstance(m, nn.Conv2d):
                W = m.weight
                N, C, Kh, Kw = W.shape
                W_mat = torch.reshape(W, (N, C*Kh*Kw))

                I = torch.eye(C*Kh*Kw).cuda() if self.args['device'] == torch.device('cuda') else torch.eye(C*Kh*Kw)
                reg_loss += torch.norm(torch.matmul(torch.t(W_mat), W_mat) - I)
                
            if isinstance(m, nn.Linear):
                W_mat = m.weight

                L = W_mat.shape[1]
                I = torch.eye(L).cuda() if self.args['device'] == torch.device('cuda') else torch.eye(L)
                reg_loss += torch.norm(torch.matmul(torch.t(W_mat), W_mat) - I)

            if isinstance(m, nn.Conv1d):
                W = m.weight
                N, C, L = W.shape
                W_mat = torch.reshape(W, (N, C*L))
                I = torch.eye(C*L).cuda() if self.args['device'] == torch.device('cuda') else torch.eye(C*L)
                reg_loss += torch.norm(torch.matmul(torch.t(W_mat), W_mat) - I)

        return reg_loss

    def test_robustness(self, model, data_loader, gaussian_training=False, train_mean=None, train_std=None, type='large'):
        model.eval()
        # set_random_seed(self.args['seed'])
        data_len = len(data_loader.dataset)
        loss, loss_adv = 0, 0
        correct, correct_adv = 0, 0
        true_labels = []
        ben_pred_labels = []
        adv_pred_labels = []
        adversary = self.init_adversary(model, test=True)

        t = tqdm(iter(data_loader), leave=False, total=len(data_loader), disable=not self.args['verbose'] > 1)
        for batch_idx, (data, target) in (enumerate(t)):
            # run on benign data

            if self.args['noise_removal']:
                data = data.cpu().data.numpy()
                n_batch, n_channel, n_length = data.shape
                if self.args['dataset'] == 'physionet':
                    sample_freq = 100
                elif self.args['dataset'] == 'shhs':
                    sample_freq = 125

                info = mne.create_info(['eeg_ch1'] * n_batch, sample_freq, ch_types=['eeg'] * n_batch)
                raw = mne.io.RawArray(copy.copy(data[:, 0, :].reshape(n_batch, n_length)), info)
                raw.filter(self.args['l_min'], self.args['l_max'])
                data = raw._data.reshape(n_batch, n_channel, n_length).astype(np.float32)
                data = torch.from_numpy(data)

            data, target = data.to(self.args['device']), target.to(self.args['device'])

            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            true_labels.extend(target.data.cpu().numpy().flatten().tolist())
            ben_pred_labels.extend(pred.data.cpu().numpy().flatten().tolist())

            # run on perturbed data
            if gaussian_training:
                data = gaussian_noise(data.data.cpu().numpy(), train_mean, train_std,
                                      self.args['train_corruption_strength'], self.args['dataset'])
                data = torch.stack([torch.Tensor(i) for i in data])
                data = data.to(self.args['device'])
            else:
                data = adversary.perturb(data, target)

            output = model(data)
            loss_adv += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct_adv += pred.eq(target.view_as(pred)).sum().item()
            adv_pred_labels.extend(pred.data.cpu().numpy().flatten().tolist())

        loss_adv /= data_len

        ben_metrics = classification_report(true_labels, ben_pred_labels, target_names=self.args['classes'], output_dict=True)
        adv_metrics = classification_report(true_labels, adv_pred_labels, target_names=self.args['classes'], output_dict=True)
        adv_cfm = confusion_matrix(true_labels, adv_pred_labels)

        ben_metrics['ben_acc'] = 100. * correct / data_len
        adv_metrics['adv_acc'] = 100. * correct_adv / data_len

        ben_macro_f1 = ben_metrics['macro avg']['f1-score']
        adv_macro_f1 = adv_metrics['macro avg']['f1-score']

        if self.args['get_hypnogram']:
            file_name = '{}/{}_{}_{}/sparsity_{}_adv_train_eps_{}.csv'.format(self.args['log_dir'], self.args['logging_comment'],
                                                                              type, self.args['run'], self.args['sparsity'],
                                                                              self.args['test_epsilon'])

            with open(file_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(['True Label'], ['Predicted Label']))
                writer.writerows(zip(true_labels, adv_pred_labels))

        return ben_macro_f1, adv_macro_f1, ben_metrics, adv_metrics, adv_cfm

    def train(self, model, retrain=False):
        if self.args['enable_logging']:
            writer = SummaryWriter(self.args['log_dir'])

        if self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args['lr'])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'],
                                  momentum=self.args['momentum'], nesterov=self.args['nesterov'])

        adversary = self.init_adversary(model)

        if self.args['spectral_normalization']:
            U = self.spectral_init(model)

        # calculate train std and mean
        if self.args['gaussian_training']:
            if self.args['dataset'] == 'physionet' or self.args['dataset'] == 'shhs':
                train_mean = self.train_loader.dataset.tensors[0].data.numpy().mean().item()
                train_std = self.train_loader.dataset.tensors[0].data.numpy().std().item()
            else:
                train_mean = 0
                train_std = 0

        best_model = copy.deepcopy(model)
        best_val_f1 = 0
        for epoch in range(1, self.args['epochs']+1):
            model.train()

            train_loss = 0
            correct = 0
            true_labels = []
            pred_labels = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.args['device']), target.to(self.args['device'])

                if self.robust_training:
                    # when performing attack, the model needs to be in eval mode
                    # also the parameters should be accumulating gradients
                    with ctx_noparamgrad_and_eval(model):
                        data = adversary.perturb(data, target)

                elif self.gaussian_training:
                    data = gaussian_noise(data.data.cpu().numpy(), train_mean, train_std, self.args['train_corruption_strength'], self.args['dataset'])
                    data = torch.stack([torch.Tensor(i) for i in data])
                    data = data.to(self.args['device'])

                optimizer.zero_grad()
                output = model(data)

                if self.args['spectral_normalization']:
                    model, U = self.spectral_normalize(model, U)

                loss = F.cross_entropy(output, target, reduction='mean')

                if self.orthogonality:
                    loss += self.ortho_lambda * self.ortho_reg(model)

                loss.backward()

                # enable or disable regularization on gamma
                if not retrain:
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                            m.weight.grad.data.add_(self.args['gamma_lambda'] * torch.sign(m.weight.data))

                optimizer.step()
                train_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                true_labels.extend(target.data.cpu().numpy().flatten().tolist())
                pred_labels.extend(pred.data.cpu().numpy().flatten().tolist())

                if batch_idx % 100 == 0:
                    if self.args['verbose'] > 1: print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))


            train_loss /= len(self.train_loader.dataset)
            metrics = classification_report(true_labels, pred_labels, target_names=self.args['classes'], output_dict=True)
            metrics['acc_train'] = 100. * correct / len(self.train_loader.dataset)
            train_f1 = metrics['macro avg']['f1-score']

            if self.gaussian_training:
                vf1, nvf1, _, _, _ = self.test_robustness(model, self.val_loader, self.gaussian_training, train_mean, train_std)
                tf1, ntf1, _, _, _ = self.test_robustness(model, self.test_loader, self.gaussian_training, train_mean, train_std)
                vf1_nvf1_avg = (vf1 + nvf1) / 2.0
                vf1_avf1_avg = -10
                log_dict = {'nvf1': nvf1, 'vf1': vf1, 'tf1': tf1, 'ntf1': ntf1, 'vf1_nvf1_avg': vf1_nvf1_avg}
            else:
                vf1, avf1, _, _, _ = self.test_robustness(model, self.val_loader)
                tf1, atf1, _, _, _= self.test_robustness(model, self.test_loader)
                vf1_avf1_avg = (vf1+avf1)/2.0
                vf1_nvf1_avg = -10
                log_dict = {'avf1': avf1, 'vf1': vf1, 'tf1': tf1, 'atf1': atf1, 'vf1_avf1_avg': vf1_avf1_avg}

            # print stats
            if self.robust_training:
                print_results(self.args, adv_train_f1=train_f1,val_f1=vf1,test_f1=tf1,adv_val_f1=avf1, adv_test_f1=atf1, vf1_avf1_avg=vf1_avf1_avg,epoch=epoch)
                if vf1_avf1_avg >= best_val_f1:
                    if self.args['verbose'] > 0: print('Model improved vf1_avf1_avg acc {} -> {} '.format(best_val_f1, vf1_avf1_avg))
                    best_val_f1 = vf1_avf1_avg
            elif self.gaussian_training:
                print_results(self.args, noise_train_f1=train_f1, val_f1=vf1, test_f1=tf1, noise_val_f1=nvf1,
                                noise_test_f1=ntf1, vf1_nvf1_avg=vf1_nvf1_avg, epoch=epoch)
                if vf1_nvf1_avg >= best_val_f1:
                    if self.args['verbose'] > 0: print('Model improved vf1_nvf1_avg acc {} -> {} '.format(best_val_f1, vf1_nvf1_avg))
                    best_val_f1 = vf1_nvf1_avg
            else:
                print_results(self.args,train_f1=train_f1,val_f1=vf1,test_f1=tf1,adv_val_f1=avf1,adv_test_f1=atf1,vf1_avf1_avg=vf1_avf1_avg,epoch=epoch)
                if vf1 >= best_val_f1:
                    if self.args['verbose'] > 0: print('Model improved vf1 acc {} -> {} '.format(best_val_f1, vf1))
                    best_val_f1 = vf1

            # log stats
            if self.args['enable_logging']:
                if retrain:
                    writer.add_scalars('{}_small_{}/sparsity_{}_trainnoisestrength_{}_trainepsilon_{}_gammalambda_{}_ortholambda_{}'.format(
                                                                                          self.args['logging_comment'],
                                                                                          self.args['run'],
                                                                                          self.args['sparsity'],
                                                                                          self.args['train_corruption_strength'],
                                                                                          self.args['train_epsilon'],
                                                                                          self.args['gamma_lambda'],
                                                                                          self.args['ortho_lambda']),
                                                                                          log_dict, epoch)
                else:
                    writer.add_scalars('{}_large_{}/sparsity_{}_trainnoisestrength_{}_trainepsilon_{}_gammalambda_{}_ortholambda_{}'.format(
                                                                                          self.args['logging_comment'],
                                                                                          self.args['run'],
                                                                                          self.args['sparsity'],
                                                                                          self.args['train_corruption_strength'],
                                                                                          self.args['train_epsilon'],
                                                                                          self.args['gamma_lambda'],
                                                                                          self.args['ortho_lambda']),
                                                                                          log_dict, epoch)

            # save best model
            if self.args['enable_saving'] and ((best_val_f1 == vf1_avf1_avg) or (best_val_f1 == vf1_nvf1_avg) or (best_val_f1 == vf1)):
                best_model = copy.deepcopy(model)
                if self.args['verbose'] > 0:  print('Saving to file ...\n')
                if retrain:
                    save_model(model, self.args['small_model_path'], self.args)
                else:
                    save_model(model, self.args['full_model_path'], self.args)

            # reduce learning rate depending on dataset
            if self.args['optimizer'] == 'sgd':
                if self.args['dataset'] == 'cifar10':
                    adjust_learning_rate_cifar10(self.args, optimizer, epoch)
                elif self.args['dataset'] == 'mnist':
                    adjust_learning_rate_mnist(self.args, optimizer, epoch)
                elif self.args['dataset'] == 'physionet':
                    adjust_learning_rate_physionet(self.args, optimizer, epoch)
                elif self.args['dataset'] == 'shhs':
                    adjust_learning_rate_shhs(self.args, optimizer, epoch)
            elif self.args['optimizer'] == 'adam':
                scheduler.step()

        if self.args['test_corruption'] is 'none':
            tf1, atf1, _, _, _ = self.test_robustness(best_model, self.test_loader)
            print_results(self.args, test_f1=tf1, adv_test_f1=atf1, epoch=self.args['epochs'])
        else:
            ntf1, _, _ = test(best_model, self.args['device'], self.test_loader)
            print_results(self.args, noise_test_f1=ntf1, epoch=self.args['epochs'])

        return best_model

    def prune_model(self, model):
        pruned_model = copy.deepcopy(model)

        # idx_dict maps layer_idx to the layer in t
        ctr, self.idx_dict = make_idx_dict(pruned_model, -1, [], {})

        gamma_thresh = self.get_gamma_threshold(model)

        for layer_idx in sorted(self.prune_layers):
            pruned_model = self.prune_layer(pruned_model, layer_idx, gamma_thresh)

        if self.args['device'] == torch.device('cuda'):
            pruned_model = pruned_model.cuda()

        return pruned_model

    def prune_layer(self, pruned_model, layer_idx, gamma_thresh):
        # will prune a specific layer specific to the threshold

        layer1 = get_layer_from_idx(pruned_model, copy.deepcopy(self.idx_dict), layer_idx)
        layer2 = None
        batchnorm_idx, batchnorm_layer = None, None
        next_layer_idx = layer_idx
        while not (isinstance(layer2, nn.Linear) or isinstance(layer2, nn.Conv2d) or isinstance(layer2, nn.Conv1d)):
            next_layer_idx = next_layer_idx + 1
            layer2 = get_layer_from_idx(pruned_model, copy.deepcopy(self.idx_dict), next_layer_idx)

            if isinstance(layer2, nn.BatchNorm2d) or isinstance(layer2, nn.BatchNorm1d):
                batchnorm_idx = next_layer_idx
                batchnorm_layer = layer2

        # find index of surviving filters (don't kill more than 90% filters in a layer)
        bn = batchnorm_layer.weight.data
        gamma_thresh = np.min([torch.kthvalue(bn.cpu(),int(bn.shape[0]*(1-self.args['min_layerwise_sparsity'])))[0],gamma_thresh.cpu()])
        idx_surv = bn > gamma_thresh

        # Select filter weights for surviving filters
        W1, B1 = layer1.weight.data, layer1.bias.data if layer1.bias is not None else None
        gamma, beta = batchnorm_layer.weight.data, batchnorm_layer.bias.data
        W2, B2 = layer2.weight.data, layer2.bias.data if layer2.bias is not None else None
        B1_flag, B2_flag = True if B1 is not None else False, True if B2 is not None else False

        gamma_pruned, beta_pruned = gamma[idx_surv], beta[idx_surv]

        if isinstance(layer1, nn.Conv2d):
            W1_pruned = W1[idx_surv, :, :, :]
            layer1_pruned = nn.Conv2d(W1_pruned.shape[1], W1_pruned.shape[0], W1_pruned.shape[2], stride=layer1.stride, padding=layer1.padding, bias=B1_flag)
            bn_pruned = nn.BatchNorm2d(W1_pruned.shape[0])
        elif isinstance(layer1, nn.Conv1d):
            W1_pruned = W1[idx_surv, :, :]
            layer1_pruned = nn.Conv1d(W1_pruned.shape[1], W1_pruned.shape[0], W1_pruned.shape[2], stride=layer1.stride, padding=layer1.padding, bias=B1_flag)
            bn_pruned = nn.BatchNorm1d(W1_pruned.shape[0])
        elif isinstance(layer1, nn.Linear):
            W1_pruned = W1[idx_surv, :]
            layer1_pruned = nn.Linear(W1_pruned.shape[1], W1_pruned.shape[0])
            bn_pruned = nn.BatchNorm1d(W1_pruned.shape[0])

        if isinstance(layer2, nn.Conv2d):
            W2_pruned = W2[: , idx_surv, :, :]
            layer2_pruned = nn.Conv2d(W2_pruned.shape[1], W2_pruned.shape[0], W2_pruned.shape[2], stride=layer2.stride, padding=layer2.padding, bias=B2_flag)
        elif isinstance(layer2, nn.Conv1d):
            W2_pruned = W2[:, idx_surv, :]
            layer2_pruned = nn.Conv1d(W2_pruned.shape[1], W2_pruned.shape[0], W2_pruned.shape[2], stride=layer2.stride, padding=layer2.padding, bias=B2_flag)
        elif isinstance(layer2, nn.Linear):
            if isinstance(layer1, nn.Conv2d):
                fm_window = self.conv_feature_size*self.conv_feature_size 
                W2_pruned = torch.cat([torch.stack([W2[:, j] for j in range(f*fm_window, (f+1)*fm_window)]) for f in torch.nonzero(idx_surv)])
                W2_pruned = torch.t(W2_pruned)

                layer2_pruned = nn.Linear(W2_pruned.shape[1], W2_pruned.shape[0])
            elif isinstance(layer1, nn.Conv1d):
                fm_window = self.conv_feature_size
                W2_pruned = torch.cat([torch.stack([W2[:, j] for j in range(f * fm_window, (f + 1) * fm_window)]) for f in torch.nonzero(idx_surv)])
                W2_pruned = torch.t(W2_pruned)

                layer2_pruned = nn.Linear(W2_pruned.shape[1], W2_pruned.shape[0])

            else:
                W2_pruned = W2[:, idx_surv]
                layer2_pruned = nn.Linear(W2_pruned.shape[1], W2_pruned.shape[0])

        # Set surviving weights to new layers
        layer1_pruned.weight.data = W1_pruned
        batchnorm_layer.weight.data, batchnorm_layer.bias.data = gamma_pruned, beta_pruned
        layer2_pruned.weight.data = W2_pruned

        # Set surviving biases to new layers
        if B1_flag:
            layer1_pruned.bias.data = B1[idx_surv]
        if B2_flag:
            layer2_pruned.bias.data = B2

        # Set new layers in pruned model
        set_layer_to_idx(pruned_model, copy.deepcopy(self.idx_dict), layer_idx, layer1_pruned)
        set_layer_to_idx(pruned_model, copy.deepcopy(self.idx_dict), batchnorm_idx, bn_pruned)
        set_layer_to_idx(pruned_model, copy.deepcopy(self.idx_dict), next_layer_idx, layer2_pruned)

        return pruned_model

    def get_gamma_threshold(self, model):
        # will return the gamma threshold that resutls in
        # desired sparsity of the model

        bn = torch.zeros(0)
        if self.args['device'] == torch.device('cuda'):
            bn = bn.cuda()

        # get all gammas across all layers
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn = torch.cat((bn, m.weight.data))

        # sort all gamma values and find threshold corresponding to desired sparsity
        y, i = torch.sort(bn)
        thre_index = int(bn.shape[0] * self.args['sparsity'])
        thre = y[thre_index]

        return thre
