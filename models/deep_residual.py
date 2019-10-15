import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calc_padding(kernel_size):
    return math.floor((kernel_size - 1) / 2)


class ResFirstBlock(nn.Module):
    def __init__(self, filters=(64, 64), kernel_size=17, dropout_rate=0.5, bias=True):
        super(ResFirstBlock, self).__init__()

        nb_filter1, nb_filter2 = filters
        padding = calc_padding(kernel_size)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=nb_filter1, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=nb_filter1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter1, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filter1)
        self.relu2 = nn.ReLU()
        # self.dr1 = nn.Dropout(p=dropout_rate)

        self.conv3 = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size, padding=padding, bias=bias)

        self.bn3 = nn.BatchNorm1d(num_features=nb_filter2)
        self.relu3 = nn.ReLU()
        # self.dr2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(x)))
        out = self.relu3(self.bn3(self.conv3(out)))

        return x + out


class ResSubSamBlock(nn.Module):
    def __init__(self, filters=(64, 64), kernel_size=17, subsam=2, dropout_rate=0.5, bias=True):
        super(ResSubSamBlock, self).__init__()

        nb_filter1, nb_filter2 = filters

        padding = calc_padding(kernel_size)
        self.conv1 = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter1, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(num_features=nb_filter1)
        self.pool1 = nn.MaxPool1d(kernel_size=subsam)
        self.relu1 = nn.ReLU()
        # self.dr1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size, bias=bias, padding=padding)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filter2)
        self.relu2 = nn.ReLU()
        # self.dr2 = nn.Dropout(p=dropout_rate)

        padding = calc_padding(kernel_size=1)
        self.short = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=1, padding=padding, bias=bias)
        self.short_pool = nn.MaxPool1d(kernel_size=subsam)

    def forward(self, x):
        out = self.relu1(self.pool1(self.bn1(self.conv1(x))))
        out = self.relu2(self.bn2(self.conv2(out)))

        out = out + self.short_pool(self.short(x))

        return out


class ResNoSubBlock(nn.Module):
    def __init__(self, filters=(64, 64), kernel_size=17, dropout_rate=0.5, bias=False):
        super(ResNoSubBlock, self).__init__()

        nb_filter1, nb_filter2 = filters

        padding = calc_padding(kernel_size)
        self.conv1 = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter1, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(num_features=nb_filter1)
        self.relu1 = nn.ReLU()
        # self.dr1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size, bias=bias, padding=padding)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filter2)
        self.relu2 = nn.ReLU()
        # self.dr2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        out = out + x

        return out


class DeepResidual(nn.Module):
    def __init__(self, n_linear):
        super(DeepResidual, self).__init__()

        self.layer1 = ResFirstBlock(filters=[64, 64])

        self.layer2 = ResSubSamBlock(filters=[64, 64])
        self.layer3 = ResNoSubBlock(filters=[64, 64])

        self.layer4 = ResSubSamBlock(filters=[64, 128])
        self.layer5 = ResNoSubBlock(filters=[128, 128])

        self.layer6 = ResSubSamBlock(filters=[128, 128])
        self.layer7 = ResNoSubBlock(filters=[128, 128])

        self.layer8 = ResSubSamBlock(filters=[128, 192])
        self.layer9 = ResNoSubBlock(filters=[192, 192])

        self.layer10 = ResSubSamBlock(filters=[192, 192])
        self.layer11 = ResNoSubBlock(filters=[192, 192])

        self.layer12 = ResSubSamBlock(filters=[192, 256])
        self.layer13 = ResNoSubBlock(filters=[256, 256])

        self.layer14 = ResSubSamBlock(filters=[256, 256])
        self.layer15 = ResNoSubBlock(filters=[256, 256])
        self.layer16 = ResSubSamBlock(filters=[256, 512])

        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(n_linear, 5)

        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.output(x)
        return x


class DeepSleepKDD(nn.Module):
    def __init__(self):
        super(DeepSleepKDD, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=25, kernel_size=100, stride=1)
        self.bn1 = nn.BatchNorm1d(25)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn2 = nn.BatchNorm1d(25)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn3 = nn.BatchNorm1d(25)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn4 = nn.BatchNorm1d(25)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn5 = nn.BatchNorm1d(25)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn6 = nn.BatchNorm1d(25)
        self.relu6 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn7 = nn.BatchNorm1d(25)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn8 = nn.BatchNorm1d(25)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=100, stride=1)
        self.bn9 = nn.BatchNorm1d(25)
        self.relu9 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=100, stride=1)
        self.bn10 = nn.BatchNorm1d(50)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=100, stride=1)
        self.bn11 = nn.BatchNorm1d(50)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=100, stride=1)
        self.bn12 = nn.BatchNorm1d(50)
        self.relu12 = nn.ReLU()

        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=100, stride=1)
        self.bn13 = nn.BatchNorm1d(100)
        self.relu13 = nn.ReLU()

        self.conv14 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=100, stride=1)
        self.bn14 = nn.BatchNorm1d(100)
        self.relu14 = nn.ReLU()

        self.conv15 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=100, stride=1)
        self.bn15 = nn.BatchNorm1d(100)
        self.relu15 = nn.ReLU()

        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool1d(kernel_size=10, stride=10)

        self.conv16 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=45, stride=45)
        self.bn16 = nn.BatchNorm1d(100)
        self.relu16 = nn.ReLU()

        self.pool6 = nn.MaxPool1d(kernel_size=10, stride=10)

        self.conv17 = nn.Conv1d(in_channels=100, out_channels=4, kernel_size=5, stride=1)
        self.bn17 = nn.BatchNorm1d(4)
        self.relu17 = nn.ReLU()

        self.fc1 = nn.Linear(8, 100)
        self.bn18 = nn.BatchNorm1d(100)
        self.relu18 = nn.ReLU()

        self.fc2 = nn.Linear(100, 5)
        self.bn19 = nn.BatchNorm1d(5)
        self.relu19 = nn.ReLU()

        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)  # 0
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)  # 3
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)  # 6
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)  # 9
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)  # 12
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)  # 15
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.pool1(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)

        x = self.pool2(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)

        x = self.pool3(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu14(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu15(x)

        x = self.pool4(x)
        x = self.pool5(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu16(x)

        x = self.pool6(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu17(x)
        x = x.view(int(x.size(0)), -1)

        x = self.fc1(x)
        x = self.bn18(x)
        x = self.relu18(x)

        x = self.fc2(x)
        x = self.bn19(x)
        x = self.relu19(x)

        x = self.output(x)

        return x