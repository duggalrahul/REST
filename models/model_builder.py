from __future__ import print_function

import torch
import torch.utils.data as data
import torch.utils.data as utils
import numpy as np
from random import shuffle
import glob


from utils import gaussian_noise, impulse_noise, shot_noise
from models.sorsnet import SorsNet
from models.biswal import BiswalNet
from models.deep_residual import DeepResidual
from data.shhs.shhs import prepare_dataset


# https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
class GenHelper(data.Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


# def train_valid_split(ds, train_split=0.9, random_seed=None):
#     '''
#     This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
#     efficiently.
#     :return:
#     '''
#     if random_seed != None:
#         np.random.seed(random_seed)
#
#     dslen = len(ds)
#     indices = list(range(dslen))
#     train_size = int(dslen * train_split)
#     valid_size = dslen - train_size
#     np.random.shuffle(indices)
#     train_mapping = indices[0:train_size]
#     valid_mapping = indices[train_size:]
#     train = GenHelper(ds, dslen - valid_size, train_mapping)
#     valid = GenHelper(ds, valid_size, valid_mapping)
#
#     return train, valid


class Model_Builder:
    def __init__(self, model_type, dataset, model_path, args, get_hypnogram=False):
        self.args = args
        self.model_type = model_type
        self.dataset = dataset
        self.model_path = model_path
        self.data_dir = args['data_dir']

        self.min_value = np.Inf
        self.max_value = -np.Inf

        self.num_classes = {
            'cifar10': 10,
            'mnist': 10,
            'physionet': 5
        }

        extend = False

        if model_type == 'biswalnet':
            config = [3, 4, 6, 3]
            if dataset == 'physionet':
                self.model = BiswalNet(config, k_size=178)
            elif dataset == 'shhs':
                self.model = BiswalNet(config, k_size=178)

        elif model_type == 'sorsnet':
            extend = True
            if dataset == 'physionet':
                self.model = SorsNet(n_linear=768)
            elif dataset == 'shhs':
                self.model = SorsNet(n_linear=1024)

        elif model_type == 'deep_residual':
            if dataset == 'physionet':
                self.model = DeepResidual(n_linear=5632)
            elif dataset == 'shhs':
                self.model = DeepResidual(n_linear=7168)

        device = args['device']
        self.model = self.model.to(device)

        self.val_loader = None

        if dataset == 'physionet':
            if get_hypnogram:
                self.train_loader, self.val_loader, self.test_loader = None, None, self.get_single_physionet_test_patient(extend=extend)
            else:
                self.train_loader, self.val_loader, self.test_loader = self.get_physionet(channel=self.args['signal_type'], extend=extend)

        elif dataset == 'shhs':
            if get_hypnogram:
                self.train_loader, self.val_loader, self.test_loader = None, None, self.get_single_shhs_test_patient(extend=extend)
            else:
                self.train_loader, self.val_loader, self.test_loader = self.get_shhs(extend=extend)


    def get_single_shhs_test_patient(self,extend):
        shhs_input = self.data_dir + '/shhs/shhs_subset/'
        shhs_output = self.data_dir + '/shhs/shhs_subset/output'

        if extend:
            shhs_output = shhs_output + '_extended'

        data, labels = prepare_dataset(shhs_input, preprocessed_dir=shhs_output, shhs='1',
                                       nb_patients=None,filter=False,channel='EEG', extend=extend)

        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.split_eeg_data(data, labels)

        val_split = int(0.9 * len(data))

        test_data_single_patient = data[val_split+1:val_split+2]
        test_labels_single_patient = labels[val_split+1:val_split+2]

        test_data_single_patient, test_labels_single_patient = self.transform(test_data_single_patient,
                                                                              test_labels_single_patient,
                                                                              shuf=False)

        if self.args['test_corruption'] == 'none':
            test_dataset = utils.TensorDataset(test_data_single_patient, test_labels_single_patient)
        else:
            corruption_type = self.args['test_corruption']
            corruption_strength = self.args['test_corruption_strength']

            train_mean = float(train_data.numpy().mean())
            train_std = float(train_data.numpy().std())
            train_max = float(train_data.numpy().max())
            train_min = float(train_data.numpy().min())

            if corruption_type == 'gaussian_noise':
                corrupted_test_data = gaussian_noise(test_data_single_patient.numpy(), train_mean, train_std, corruption_strength,self.args['dataset'])
            elif corruption_type == 'impulse_noise':
                corrupted_test_data = impulse_noise(test_data_single_patient.numpy(), train_max, train_min, corruption_strength)
            elif corruption_type == 'shot_noise':
                corrupted_test_data = shot_noise(test_data_single_patient.numpy(), train_max, train_min, corruption_strength)
            else:
                print("Noise not available")
                exit(0)

            corrupted_test_data = torch.stack([torch.Tensor(i) for i in corrupted_test_data])
            test_dataset = utils.TensorDataset(corrupted_test_data, test_labels_single_patient)

        test_loader = utils.DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        return test_loader


    def get_shhs(self, extend):
        shhs_input = self.data_dir + '/shhs/shhs_subset/'
        shhs_output = self.data_dir + '/shhs/shhs_subset/output'

        if extend:
            shhs_output = shhs_output + '_extended'

        data, labels = prepare_dataset(shhs_input, preprocessed_dir=shhs_output, shhs='1', nb_patients=None, filter=False,
                                       channel='EEG', extend=extend)

        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.split_eeg_data(data, labels)

        self.min_value = float(np.min(train_data.numpy()))
        self.max_value = float(np.max(train_data.numpy()))

        train_dataset = utils.TensorDataset(train_data, train_labels)
        train_loader = utils.DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)


        if self.args['test_corruption'] == 'none':
            test_dataset = utils.TensorDataset(test_data, test_labels)
            val_dataset = utils.TensorDataset(val_data, val_labels)
        else:
            corruption_type = self.args['test_corruption']
            corruption_strength = self.args['test_corruption_strength']

            train_mean = float(train_data.numpy().mean())
            train_std = float(train_data.numpy().std())
            train_max = float(train_data.numpy().max())
            train_min = float(train_data.numpy().min())

            if corruption_type == 'gaussian_noise':
                corrupted_test_data = gaussian_noise(test_data.numpy(), train_mean, train_std, corruption_strength,self.args['dataset'])
                corrupted_val_data = gaussian_noise(val_data.numpy(), train_mean, train_std, corruption_strength,self.args['dataset'])
            elif corruption_type == 'impulse_noise':
                corrupted_test_data = impulse_noise(test_data.numpy(), train_max, train_min, corruption_strength)
                corrupted_val_data = impulse_noise(val_data.numpy(), train_max, train_min, corruption_strength)
            elif corruption_type == 'shot_noise':
                corrupted_test_data = shot_noise(test_data.numpy(), train_max, train_min, corruption_strength)
                corrupted_val_data = shot_noise(val_data.numpy(), train_max, train_min, corruption_strength)
            else:
                print("Noise not available")
                exit(0)

            corrupted_test_data = torch.stack([torch.Tensor(i) for i in corrupted_test_data])
            test_dataset = utils.TensorDataset(corrupted_test_data, test_labels)

            corrupted_val_data = torch.stack([torch.Tensor(i) for i in corrupted_val_data])
            val_dataset = utils.TensorDataset(corrupted_val_data, val_labels)

        test_loader = utils.DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        val_loader = utils.DataLoader(val_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader, test_loader

    def get_single_physionet_test_patient(self, channel='fz', extend=False):
        self.data_dir = self.data_dir + '/physionet/output_' + channel
        if extend:
            self.data_dir = self.data_dir + '_extended/'
        else:
            self.data_dir = self.data_dir + '/'

        data, labels, self.min_value, self.max_value = self.get_physionet_channel(self.data_dir + '*.npz')
        train_data, train_labels, _, _, _, _ = self.split_eeg_data(data, labels)

        val_split = int(0.9 * len(data))

        test_data_single_patient = data[val_split + 1:val_split + 2]
        test_labels_single_patient = labels[val_split + 1:val_split + 2]

        test_data_single_patient, test_labels_single_patient = self.transform(test_data_single_patient,test_labels_single_patient,shuf=False)

        if self.args['test_corruption'] == 'none':
            test_dataset = utils.TensorDataset(test_data_single_patient, test_labels_single_patient)
        else:
            corruption_type = self.args['test_corruption']
            corruption_strength = self.args['test_corruption_strength']

            train_mean = float(train_data.numpy().mean())
            train_std = float(train_data.numpy().std())
            train_max = float(train_data.numpy().max())
            train_min = float(train_data.numpy().min())

            if corruption_type == 'gaussian_noise':
                corrupted_test_data = gaussian_noise(test_data_single_patient.numpy(), train_mean, train_std, corruption_strength,self.args['dataset'])
            elif corruption_type == 'impulse_noise':
                corrupted_test_data = impulse_noise(test_data_single_patient.numpy(), train_max, train_min, corruption_strength)
            elif corruption_type == 'shot_noise':
                corrupted_test_data = shot_noise(test_data_single_patient.numpy(), train_max, train_min, corruption_strength)
            else:
                print("Noise not available")
                exit(0)

            corrupted_test_data = torch.stack([torch.Tensor(i) for i in corrupted_test_data])
            test_dataset = utils.TensorDataset(corrupted_test_data, test_labels_single_patient)

        test_loader = utils.DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        return test_loader

    def get_physionet_channel(self,path):
        channel_files = glob.glob(path)

        data, labels = [], []
        min_val, max_val = np.inf, -np.inf

        for file in channel_files:
            eeg = np.load(file)

            signal, label = eeg.f.x, eeg.f.y

            min_val = np.min([min_val, np.min(signal)])
            max_val = np.max([max_val, np.max(signal)])

            data.append(signal)
            labels.append(label)

        return data, labels, min_val, max_val

    def get_physionet(self, channel='fz', extend=False):
        self.data_dir = self.data_dir + '/physionet/output_' + channel
        if extend:
            self.data_dir = self.data_dir + '_extended/'
        else:
            self.data_dir = self.data_dir + '/'

        data, labels, self.min_value, self.max_value = self.get_physionet_channel(self.data_dir + '*.npz')
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.split_eeg_data(data, labels)

        train_dataset = utils.TensorDataset(train_data, train_labels)
        train_loader = utils.DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        if self.args['test_corruption'] == 'none':
            test_dataset = utils.TensorDataset(test_data, test_labels)
            val_dataset = utils.TensorDataset(val_data, val_labels)
        else:
            corruption_type = self.args['test_corruption']
            corruption_strength = self.args['test_corruption_strength']

            train_mean = float(train_data.numpy().mean())
            train_std = float(train_data.numpy().std())
            train_max = float(train_data.numpy().max())
            train_min = float(train_data.numpy().min())

            if corruption_type == 'gaussian_noise':
                corrupted_test_data = gaussian_noise(test_data.numpy(), train_mean, train_std, corruption_strength,self.args['dataset'])
                corrupted_val_data = gaussian_noise(val_data.numpy(), train_mean, train_std, corruption_strength,self.args['dataset'])
            elif corruption_type == 'impulse_noise':
                corrupted_test_data = impulse_noise(test_data.numpy(), train_max, train_min, corruption_strength)
                corrupted_val_data = impulse_noise(val_data.numpy(), train_max, train_min, corruption_strength)
            elif corruption_type == 'shot_noise':
                corrupted_test_data = shot_noise(test_data.numpy(), train_max, train_min, corruption_strength)
                corrupted_val_data = shot_noise(val_data.numpy(), train_max, train_min, corruption_strength)
            else:
                print("Noise not available")
                exit(0)

            corrupted_test_data = torch.stack([torch.Tensor(i) for i in corrupted_test_data])
            test_dataset = utils.TensorDataset(corrupted_test_data, test_labels)

            corrupted_val_data = torch.stack([torch.Tensor(i) for i in corrupted_val_data])
            val_dataset = utils.TensorDataset(corrupted_val_data, val_labels)

        test_loader = utils.DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        val_loader = utils.DataLoader(val_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader, test_loader

    def split_eeg_data(self, data, labels, shuffle=True):
        train_split = int(0.8 * len(data))
        val_split = int(0.9 * len(data))

        train_data = data[:train_split]
        train_labels = labels[0:train_split]
        train_data, train_labels = self.transform(train_data, train_labels, shuffle)

        val_data = data[train_split:val_split]
        val_labels = labels[train_split:val_split]
        val_data, val_labels = self.transform(val_data, val_labels, shuffle)

        test_data = data[val_split:]
        test_labels = labels[val_split:]
        test_data, test_labels = self.transform(test_data, test_labels, shuffle)

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    def transform(self, data, labels, shuf):
        data = np.concatenate(data, axis=0)
        data = np.swapaxes(data, 1, 2)

        labels = np.concatenate(labels, axis=0)

        if shuf:
            indices = list(range(len(labels)))
            shuffle(indices)
            data = data[indices, :, :]
            labels = labels[indices]

        data = torch.stack([torch.Tensor(i) for i in data])
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.long)

        return data, labels

    def get_bounds(self):
        return self.min_value, self.max_value

    def get_model(self):
        return self.model

    def refresh_model_builder(self):
        self.__init__(self.model_type, self.dataset, self.args)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

