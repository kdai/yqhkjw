import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 1)  # conv output size: [125, 25] x 32
        self.conv2 = nn.Conv2d(32, 64, 5, 1)  # conv output size: [121, 21] x 64
        self.pool = nn.MaxPool2d(5, 5)  # pool output size: [25, 5] x 64 = 8000
        self.fc1 = nn.Linear(8000, 2048)
        self.fc2 = nn.Linear(2048, 512)

        self.dp1 = nn.Dropout2d(.5)
        self.dp2 = nn.Dropout(.5)

        self.relu = F.relu

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dp1(self.pool(self.conv2(out)))
        out = self.relu(self.fc1(out))
        out = self.dp2(self.fc2(out))
        return out  # batch x 512


# class FNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dp1 = nn.Dropout2d(.25)
#         self.fc1 = nn.Linear(4608, 512)  # 64 x 8 x 9 = 4608
#         self.dp2 = nn.Dropout(.5)
#         self.fc2 = nn.Linear(512, 128)
#         self.dp3 = nn.Dropout(.5)
#         self.fc3 = nn.Linear(128, 1)
#
#         self.relu = F.relu
#
#     def forward(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.dp1(self.pool(self.conv2(out)))
#         out = torch.flatten(out, start_dim=1)
#         out = self.relu(self.dp2(self.fc1(out)))
#         out = self.relu(self.dp3(self.fc2(out)))
#         out = self.fc3(out)
#
#         return out

class FNet(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        out = self.relu(self.dropout1(self.fc1(x)))
        out = self.relu(self.dropout2(self.fc2(out)))
        out = self.fc3(out)
        # out = self.softmax(out)[:,1].contiguous()
        return out  # [n, 8,2]


class HNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128 * 28, 1024)
        self.dp1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(1024, 256)
        self.dp2 = nn.Dropout(.5)
        self.fc3 = nn.Linear(256, 1)

        self.relu = F.relu
        self.softmax = F.softmax

    def forward(self, xij):
        out = self.relu(self.dp1(self.fc1(xij)))
        out = self.relu(self.dp2(self.fc2(out)))
        out = self.fc3(out)
        out = self.softmax(out)

        return out


class LIMloss(nn.Module):
    def __init__(self):
        super(LIMloss, self).__init__()
        # self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.margin = torch.tensor([1], dtype=torch.float32, requires_grad=False).to(self.devices)
        self.alpha = torch.tensor([0], dtype=torch.float32, requires_grad=False).to(self.devices)
        # self.margin = torch.Tensor([1], self.device)
        # self.alpha = torch.Tensor([0], self.device)

    def forward(self, fxi, fxj, w):
        # fxi = [n, 8]
        # fxj = [n, 8]
        # w = [n, 8]
        # pdb.set_trace()
        loss = torch.mul(w, torch.max(self.alpha, torch.add(torch.sub(self.margin, fxi), fxj))).mean(1).mean()  # [n,8]

        return loss
