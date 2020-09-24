import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# default input parameters: feature dimension(512), hidden_dim(512), sequence_len(5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    def __init__(self, sequence_len, feature_dim, hidden_dim):
        super().__init__()
        self.sequence_len = sequence_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: tensor, (batch, seq_len, feature_dim)
        out = self.lstm(x)
        # output((batch, seq_len, num_directions*hidden_size), (h_n, c_n))
        return out[0]


class FC_Regression(nn.Module):
    def __init__(self, sequence_len, feature_dim, hidden_dim, bidirectional=True):
        super().__init__()
        self.sequence_len = sequence_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional is True else 1
        self.BN1 = nn.BatchNorm1d(512, eps=1e-5, momentum=.1, affine=True).cuda()
        self.BN2 = nn.BatchNorm1d(128, eps=1e-5, momentum=.1, affine=True).cuda()
        self.dropout = nn.Dropout(p=.5, inplace=False)
        self.fc1 = nn.Linear(self.hidden_dim*self.num_directions, 512)  # for bidirections
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            self.fc1,
            self.BN1,
            self.dropout,
            self.relu,
            self.fc2,
            self.BN2,
            self.dropout,
            self.relu,
            self.fc3
        )
    def forward(self, x):
        # x(batch, seq_len, num_directions*hidden_size)
        ret = [self.net(x[:, N, :]) for N in range(self.sequence_len)]
        # ret, (seq_len, batch, 1)
        ret = torch.stack(ret).permute(1, 0, 2).contiguous().cuda()
        return ret  # (batch, sequence_len)


class H(nn.Module):
    def __init__(self, sequence_len, feature_dim):
        super().__init__()
        self.sequence_len = sequence_len
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(self.feature_dim*2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=.5, inplace=False)
        self.BN1 = nn.BatchNorm1d(self.sequence_len, eps=1e-5, momentum=.1, affine=True).cuda()
        self.BN2 = nn.BatchNorm1d(self.sequence_len, eps=1e-5, momentum=.1, affine=True).cuda()
        self.relu = nn.ReLU()

    def forward(self, xi, xj):
        # x: tensor, (batch, seq_len, feature_dim)
        x = torch.cat((xi, xj), dim=2)
        out = self.fc1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out: tensor, (batch, seq_len, 1)
        return out


class LimLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.margin = torch.Tensor([1]).to(self.devices)
        self.alpha = torch.Tensor([0]).to(self.devices)

    def forward(self, fxi, fxj, hij):
        """
        :param fxi, fxj: tensor, (batch, seq_len)
        :param hij: tensor, (batch, seq_len)
        :return:
        """
        w = F.softmax(hij, dim=1)  # h: (batch, seq_len)
        loss = torch.mul(w, torch.max(self.alpha, torch.add(torch.sub(self.margin, fxi), fxj))).mean(1).mean()
        return loss