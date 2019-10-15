import torch.nn as nn

class SorsNet(nn.Module):
    def __init__(self, n_linear):
        super(SorsNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.bn6 = nn.BatchNorm1d(128)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=2, padding=3)
        self.bn7 = nn.BatchNorm1d(256)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.bn8 = nn.BatchNorm1d(256)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.bn9 = nn.BatchNorm1d(256)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.bn10 = nn.BatchNorm1d(256)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn11 = nn.BatchNorm1d(256)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn12 = nn.BatchNorm1d(256)
        self.relu12 = nn.ReLU(inplace=True)

        self.fc13 = nn.Linear(n_linear, 100)
        self.bn13 = nn.BatchNorm1d(100)
        self.relu13 = nn.ReLU(inplace=True)

        self.fc14 = nn.Linear(100, 5)
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
        x = self.conv7(x)  # 18
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.conv8(x)  # 21
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.conv9(x)  # 24
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)  # 27
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.conv11(x)  # 30
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)  # 33
        x = self.bn12(x)
        x = self.relu12(x)
        x = x.view(int(x.size(0)), -1)
        x = self.fc13(x) #36
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.fc14(x) #39
        x = self.output(x)

        return x