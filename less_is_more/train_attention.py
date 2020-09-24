
import torch
import logging
import torch.optim as optim
# import opts
import os
import tools
import pdb
from torch.utils.data import DataLoader
from loss import LIMloss,Rankingloss
from dataset import Pairs
from Recorder import Recorder, Drawer
from tools import clip2frame
import numpy as np
from opts import args
from tqdm import tqdm
from collections import defaultdict
from evaluate import evaluate
import model
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def test():
    summary = defaultdict(float)
    f.eval()
    gt_dict = np.load(args.test_path+'/gt_dict.npy',allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())
    total_features = np.load(args.test_path+'/feature/'+args.domain+'.npy',allow_pickle=True ).tolist()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            feature = total_features[ve]
            scores = []
            for feat in feature:
                segment = feat['segment']
                scores.append(f(torch.Tensor(feat['features']).to(device).view(1,-1))[0].item())
            summary[ve] = scores
    mechine_summary = clip2frame(summary)
    mAP,pre,recall = evaluate(mechine_summary,category_dict,args.topk_mAP)
    return mAP,pre,recall

def train(epoch_idx,mAP):
    f.train()
    h.train()
    logging.info('In epoch {}:\n'.format(epoch_idx+1))
    for batch_idx, (xi, xj) in enumerate(train_loader):
        opt.zero_grad()
        xi = xi.to(device).view(-1,args.feature_dim).contiguous()
        xj = xj.to(device).view(-1,args.feature_dim).contiguous()
        att_fxi,fxi = f(xi)
        att_fxi = att_fxi.view(-1,args.num_per_group).contiguous()
        fxi = fxi.view(-1,args.num_per_group).contiguous()

        att_fxj,fxj = f(xj)
        att_fxj = att_fxj.view(-1,args.num_per_group).contiguous()
        fxj = fxj.view(-1,args.num_per_group).contiguous()

        # torch.topk(fxi,4)
        with torch.no_grad():
            xij = torch.cat((xi, xj), dim=-1).cuda().view(-1,args.feature_dim*2).contiguous()
        w = h(xij).view(-1,args.num_per_group)
        # loss1 = limloss(fxi, fxj, w)
        lossxj = limloss(fxi,att_fxj,w)
        lossi = rankloss(att_fxi,fxi)
        lossj = rankloss(att_fxj,fxj)
        loss = lossxj+lossi+lossj

        print("In epoch {}, [{}/{}]: loss: {:.6f}, current test mAP: {:.4f}".format(epoch_idx+1,batch_idx,len(train_loader),loss.item(),mAP))
        recoder.update('loss', loss.data, epoch_idx*len(train_loader)+batch_idx)
        loss.backward()
        opt.step()
        # torch.save(f.state_dict(), os.path.join('./model_param', 'f_{}_{}.pth'.format(dataset, domain)))
        recoder.save()

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader_dict = dict(shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    # train_loader = DataLoader(Pairs(dataset, domain, short_path, long_path), **loader_dict)
    dataset = Pairs(args.train_path,args.domain,args.num_per_group)
    train_loader = DataLoader(dataset, **loader_dict)
    f = getattr(model,args.FNet)(args.feature_dim).to(device)
    h = getattr(model,args.HNet)(args.feature_dim*2, args.num_per_group).to(device)
    limloss = LIMloss().to(device)
    rankloss = Rankingloss().to(device)
    recoder = Recorder('{}_{}'.format(args.dataset, args.domain))
    tools.mkdir_if_missed('./model_param')
    mAP = 0
    opt = optim.SGD(list(f.parameters())+list(h.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch_idx in range(args.epoch):
        train(epoch_idx,mAP)
        if epoch_idx%args.interval == 0:
            print('========================testing=============================')
            mAP,pre,recall = test()
            # print(mAP)
        dataset.GeneratePairs()
        train_loader = DataLoader(dataset, **loader_dict)
    # Drawer('../fig/{}_{}.pkl'.format(dataset, domain),
        #    '../fig/{}_{}_train_loss.png'.format(dataset, domain))





