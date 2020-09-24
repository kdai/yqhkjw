import torch
import os
import network
import json5 as js
from tqdm import tqdm
from dataset import TestSet
from opts import parser_opts

# hyperparameters
arg_opt = parser_opts()

domain = arg_opt.domain

feature_dim = arg_opt.feature_dim
hidden_dim = arg_opt.hidden_dim
seq_len = arg_opt.seq_len
batch_size = arg_opt.batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

rnn = network.LSTM(seq_len, feature_dim, hidden_dim).to(device)
f = network.FC_Regression(seq_len, feature_dim, hidden_dim, True).to(device)

# load model parameters
# rnn.load_state_dict(torch.load('./model_params/{}_rnn.pth'.format(domain)))
# f.load_state_dict(torch.load('./model_params/{}_f.pth'.format(domain)))

rnn.load_state_dict(torch.load('./model_params/{}_rnn.pth'.format(domain)))
f.load_state_dict(torch.load('./model_params/{}_f.pth'.format(domain)))

# test set
test_set = TestSet(domain, seq_len)

def test(domain):
    rnn.eval()
    f.eval()

    if not os.path.exists(os.path.join('./', 'test_results')):
        os.mkdir(os.path.join('./', 'test_results'))

    results = {}
    for video in tqdm(test_set.videos()):
        feature = test_set[video].to(device)  # feature: tensor, (n_seq, seq_len, feature_dim)
        scores = f(rnn(feature))  # scores: tensor, (n_seq, 1)
        scores = scores.cpu().detach().numpy().tolist()
        scores = [i[0] for i in scores]
        results[video] = scores

    with open(os.path.join('./test_results', '{}.json'.format(domain)), 'w') as file:
        file.write(js.dumps(results))

if __name__ == '__main__':
    test(domain)

