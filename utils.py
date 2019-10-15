import copy
import numpy as np
import skimage as sk
import os
import random
import math
from sklearn.metrics import classification_report, confusion_matrix
import csv
import mne

import torch.nn as nn
import torch
import torch.nn.functional as F
import onnx
from onnx_tf.backend import prepare


def weights_init(module):
    if isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight,a=math.sqrt(5))
        torch.nn.init.constant_(module.bias, 0)
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        weight_shape = module.weight.shape
        out_channels, in_channels, kernel_size = weight_shape[0], weight_shape[1], weight_shape[2:]

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)


# set all seeds for reproducability
def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate_cifar10(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/2 and 3/4 epochs"""
    if epoch in [40, 60]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']>0: print("changed learning rate to {}".format(lr))


def adjust_learning_rate_mnist(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if epoch in [int(args['epochs']/3), int(2 * args['epochs'] / 3)]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print("changed learning rate to {}".format(lr))


def adjust_learning_rate_physionet(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if epoch in [int(args['epochs']/3), int(2 * args['epochs'] / 3)]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print("changed learning rate to {}".format(lr))


def adjust_learning_rate_shhs(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if epoch in [int(args['epochs']/3), int(2 * args['epochs'] / 3)]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print("changed learning rate to {}".format(lr))


def get_layers(network, all_layers=[]):
    '''
    gets all layers of a network
    '''
    for layer in network.children():
        if type(layer) == nn.Sequential:
            get_layers(layer, all_layers)
        if list(layer.children()) == []:
            all_layers.append(layer)
    return all_layers


def make_idx_dict(model, ctr, ary, d):
    for m_idx, m_k in enumerate(model._modules.keys()):
        n_ary = copy.deepcopy(ary)
        if len(model._modules[m_k]._modules.keys()):
            n_ary.append(m_k)
            ctr, d = make_idx_dict(model._modules[m_k], ctr, n_ary, d)
        else:
            n_ary.append(m_k)
            ctr = ctr+1
            d[ctr] = n_ary
    return ctr, d


def get_layer_from_idx(model, idx_ds, idx):
    if len(idx_ds[idx]) == 1:
        return model._modules[idx_ds[idx][0]]
    m_idx = idx_ds[idx].pop(0)
    return get_layer_from_idx(model._modules[m_idx],idx_ds,idx)


def set_layer_to_idx(model, idx_ds, idx, layer):
    if len(idx_ds[idx]) == 1:
        model._modules[idx_ds[idx][0]] = layer
    else:
        m_idx = idx_ds[idx].pop(0)
        set_layer_to_idx(model._modules[m_idx], idx_ds, idx, layer)


def _lr_rate_schedule(args, optimizer, epoch):

    if (epoch * 3 == args.epochs) or (epoch * 3 == 2 * args.epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print('reduce lr to {}'.format(lr))


def test(args, model, device, test_loader, type='large'):
    model.eval()
    data_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            if args['noise_removal']:
                data = data.cpu().data.numpy()
                n_batch, n_channel, n_length = data.shape
                if args['dataset'] == 'physionet':
                    sample_freq = 100
                elif args['dataset'] == 'shhs':
                    sample_freq = 125
                info = mne.create_info(['eeg_ch1']*n_batch, sample_freq, ch_types=['eeg']*n_batch)
                raw = mne.io.RawArray(copy.copy(data[:, 0, :].reshape(n_batch, n_length)), info)
                raw.filter(args['l_min'], args['l_max'])
                data = raw._data.reshape(n_batch,n_channel,n_length).astype(np.float32)
                data = torch.from_numpy(data)

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            true_labels.extend(target.data.cpu().numpy().flatten().tolist())
            pred_labels.extend(pred.data.cpu().numpy().flatten().tolist())

    metrics = classification_report(true_labels, pred_labels, target_names=args['classes'], output_dict=True)
    cfm = confusion_matrix(true_labels, pred_labels)
    test_loss /= data_len
    metrics['ben_acc'] = 100. * correct / data_len
    macro_f1 = metrics['macro avg']['f1-score']

    if args['get_hypnogram']:
        file_name = '{}/{}_{}_{}/sparsity_{}_{}_{}.csv'.format(args['log_dir'], args['logging_comment'], type, args['run'],
                                                   args['sparsity'], args['test_corruption'], args['test_corruption_strength'])

        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(['True Label'], ['Predicted Label']))
            writer.writerows(zip(true_labels, pred_labels))


    return macro_f1, metrics, cfm


def print_results(args, train_f1=None, val_f1=None, test_f1=None,
                  adv_train_f1=None, adv_val_f1=None, adv_test_f1=None,
                  noise_train_f1=None, noise_val_f1=None, noise_test_f1=None,
                  vf1_avf1_avg=None, vf1_nvf1_avg=None, epoch='N/A'):

    if train_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TRAIN F1 (benign): {}'.format(epoch, train_f1))

    if val_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VAL F1 (benign): {}'.format(epoch, val_f1))

    if test_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TEST F1 (benign): {}'.format(epoch, test_f1))

    if adv_train_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TRAIN F1 (adversarial): {}'.format(epoch, adv_train_f1))

    if adv_val_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VAL F1 (adversarial): {}'.format(epoch, adv_val_f1))

    if adv_test_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TEST F1 (adversarial): {}'.format(epoch, adv_test_f1))

    if noise_train_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TRAIN F1 (gaussian : strength {}): {}'.format(epoch,
                                                                                                     args['train_corruption_strength'],
                                                                                                     noise_train_f1))
    if noise_val_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VAL F1 (gaussian : strength {}): {}'.format(epoch,
                                                                                                   args['train_corruption_strength'],
                                                                                                   noise_val_f1))
    if noise_test_f1 is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TEST F1 ({}:{}): {}'.format(epoch, args['test_corruption'],
                                                                                   args['test_corruption_strength'],
                                                                                   noise_test_f1))

    if vf1_avf1_avg is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VF1/AVF1 AVG: {}'.format(epoch, vf1_avf1_avg))

    if vf1_nvf1_avg is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VF1/NVF1 AVG: {}'.format(epoch, vf1_nvf1_avg))


# https://github.com/hendrycks/robustness
def gaussian_noise(x, mean, std, severity, dataset):
    if dataset == 'physionet':
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        c = c * std
        return x + np.random.normal(loc=mean, scale=c, size=x.shape)

    elif dataset == 'shhs':
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        c = c * std
        return x + np.random.normal(loc=mean, scale=c, size=x.shape)

    return 0

# def gaussian_noise(x, x_max, x_min, severity=1):
#     c = [0.004, 0.006, .008, .009, .01][severity - 1]
#
#     x_copy = copy.deepcopy(x)
#     x_copy = (x_copy - x_min) / (x_max - x_min)
#     x_copy = np.clip(x_copy + np.random.normal(size=x_copy.shape, scale=c), 0, 1)
#     x_copy = x_copy * (x_max - x_min) + x_min
#
#     return x_copy


def impulse_noise(x, x_max, x_min, severity=1):
    # c = [.01, .02, .03, .05, .07][severity - 1]

    c = [.001, .003, .005, .007, .009][severity - 1]

    x_copy = copy.deepcopy(x)

    x_copy = (x_copy - x_min) / (x_max - x_min)
    x_copy = np.clip(sk.util.random_noise(np.array(x_copy), mode='s&p', amount=c), 0, 1)

    x_copy = x_copy * (x_max - x_min) + x_min

    return x_copy


def shot_noise(x, x_max, x_min, severity=1):
    c = [5000, 2500, 1000, 750, 500][severity - 1]

    x_copy = copy.deepcopy(x)
    x_copy = (x_copy - x_min) / (x_max - x_min)
    x_copy = np.clip(np.random.poisson(x_copy * c) / c, 0, 1)
    x_copy = x_copy * (x_max - x_min) + x_min

    return x_copy


def save_data(dataset, path):
    data = {'eegs': dataset[0], 'labels': dataset[1]}
    np.save(path, data)


def covert_to_tensorflow_and_save(pytorch_model,path,args):
    if args['dataset'] == 'mnist':
        dummy_input = torch.randn(1, 1, 28, 28)
    elif args['dataset'] == 'cifar10':
        dummy_input = torch.randn(1, 1, 32, 32)
    elif args['dataset'] == 'physionet':
        dummy_input = torch.randn(1, 1, 12000)
    elif args['dataset'] == 'shhs':
        dummy_input = torch.randn(1, 1, 15000)

    if args['device'] == torch.device('cuda'):
        dummy_input = dummy_input.cuda()

    torch.onnx.export(pytorch_model, dummy_input, path + '.onnx')
    onnx_model = onnx.load(path + '.onnx')

    tf_model = prepare(onnx_model)
    tf_model.export_graph(path + '.pb')

    # graph_def_file = args['chkpnt_dir'] + "tf_model.pb"
    # input_arrays = ['0']
    # output_arrays = ['LogSoftmax']

    # converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
    # tflite_model = converter.convert()

    return tf_model


def load_model(model_path):
    model = torch.load(model_path)
    model = model.eval()
    return model


def save_model(model, model_path, args):
    model_path = model_path[:-3]  # remove '.pt'
    torch.save(model, model_path + '.pt', pickle_protocol=4)
    # covert_to_tensorflow_and_save(model, model_path, args)
