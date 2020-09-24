import torch
import os
import network
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from opts import parser_opts
from torch.utils.data import DataLoader
from dataset import Pairs
from tqdm import tqdm
import pdb

# hyperparameters
arg_opt = parser_opts()

domain = arg_opt.domain
feature_dim = arg_opt.feature_dim
hidden_dim = arg_opt.hidden_dim
seq_len = arg_opt.seq_len
batch_size = arg_opt.batch_size
epoch = arg_opt.epoch
lr = arg_opt.learning_rate
weight_decay = arg_opt.weight_decay
momentum = arg_opt.momentum
step_size = arg_opt.step_size
gamma = arg_opt.gamma
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device_ids=[0, 1]
# networks
rnn = network.LSTM(seq_len, feature_dim, hidden_dim).to(device)
f = network.FC_Regression(seq_len, feature_dim, hidden_dim, True).to(device)
h = network.H(seq_len, feature_dim).to(device)
limloss = network.LimLoss().to(device)

# rnn = nn.DataParallel(rnn, device_ids=device_ids)
# f = nn.DataParallel(f, device_ids=device_ids)
# h = nn.DataParallel(h, device_ids=device_ids)
# limloss = nn.DataParallel(limloss, device_ids=device_ids)

opt = optim.SGD(list(rnn.parameters())+list(f.parameters())+list(h.parameters()),
                lr=lr, momentum=momentum, weight_decay=weight_decay)

scheduder = lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

# dataloader
dataloader_args = dict(shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=12)
train_loader = DataLoader(Pairs(domain, seq_len), **dataloader_args)

def train(domain):
    rnn.train()
    f.train()
    h.train()
    limloss.train()
    if not os.path.exists(os.path.join('./', 'train_results')):
        os.mkdir(os.path.join('./', 'train_results'))
    if not os.path.exists(os.path.join('./', 'model_params')):
        os.mkdir(os.path.join('./', 'model_params'))

    with open(os.path.join('./', 'train_results', '{}_train.txt'.format(domain)), 'w') as file:
        for i in range(epoch):
            print('In epoch {}:'.format(i))
            for batch_idx, (xi, xj) in tqdm(enumerate(train_loader)):
                opt.zero_grad()
                xi = xi.to(device)
                xj = xj.to(device)
                fxi = f(rnn(xi))
                fxj = f(rnn(xj))
                hij = h(xi, xj)
                loss = limloss(fxi, fxj, hij)
                loss.backward()
                opt.step()
            print('Epoch: {}, Loss: {:.6f}\n'.format(i, loss.data))
            file.writelines('Epoch: {}, Loss: {:.6f}\n'.format(i, loss.data))
            scheduder.step()
        torch.save(rnn.state_dict(), os.path.join('./', 'model_params', '{}_rnn.pth'.format(domain)))
        torch.save(f.state_dict(), os.path.join('./', 'model_params', '{}_f.pth'.format(domain)))

if __name__ == '__main__':
    train(domain)
