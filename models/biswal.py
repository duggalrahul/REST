import torch
import torch.nn as nn


class FilterBlock(nn.Module):
    def __init__(self):
        super(FilterBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=100, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=50, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        out = out + x  # shortcut connection

        return out


class BiswalNet(nn.Module):
    '''
        config = array containing the number of blocks in each super block


        ASSUMPTION = in_channels is always equal to out_channels
    '''
    def __init__(self, config, k_size):
        super(BiswalNet, self).__init__()

        self.filter_block = FilterBlock()

        self.layer1 = self._make_layer(256, 256, config[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.layer2 = self._make_layer(256, 256, config[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.layer3 = self._make_layer(256, 256, config[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.layer4 = self._make_layer(256, 256, config[3])
        self.pool4 = nn.AvgPool1d(kernel_size=k_size)

        self.fc1 = nn.Linear(256, 5)
        self.output = nn.LogSoftmax(dim=1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        for idx in range(num_blocks):
            layers.append(ResBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.filter_block(x)

        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)
        x = self.pool3(x)

        x = self.layer4(x)
        x = self.pool4(x)

        x = x.view(int(x.size(0)), -1)

        x = self.output(self.fc1(x))

        return x

