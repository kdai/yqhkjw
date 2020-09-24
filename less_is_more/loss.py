import torch
import torch.nn as nn
import torch.nn.functional as FF
import pdb
from opts import args
from torch.nn import init
class LIMloss(nn.Module):
    def __init__(self):
        super(LIMloss, self).__init__()
        # self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.margin = torch.tensor([1], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)
        # self.margin = torch.Tensor([1], self.device)
        # self.alpha = torch.Tensor([0], self.device)

    def forward(self, fxi, fxj, w):
        # fxi = [n, 8]
        # fxj = [n, 8]
        # w = [n, 8]
        # pdb.set_trace()
        loss = torch.mul(w, torch.max(self.alpha, torch.add(torch.sub(self.margin, fxi), fxj))).sum(1).mean() #[n,8]

        return loss
class Rankingloss(nn.Module):
    def __init__(self):
        super(Rankingloss,self).__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.margin = torch.tensor([1], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,fxi,fxj):  
        # pdb.set_trace()
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, fxi), fxj)).mean(1).mean()
        return loss