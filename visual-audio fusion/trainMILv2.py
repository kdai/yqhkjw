
import torch
import logging
import torch.optim as optim
# import opts
import os
import tools
import pdb
from torch.utils.data import DataLoader
from loss import *
from dataset import *
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
    gt_dict = np.load(args.test_path+'/gt_'+args.dataset+'.npy',allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())
    video_features = np.load(args.test_path+'/feature/'+args.domain+'_1s.npy',allow_pickle=True ).tolist()
    audio_features = np.load(args.test_path+'/feature/'+args.domain+'_audio_edited_nopost.npy',allow_pickle=True ).tolist()
    # pdb.set_trace()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            prefix = ve.split('.')[0]
            vfs = video_features[prefix]
            afs = audio_features[prefix]
            scores = []
            vfeat = []
            afeat = []
            for vf,af in zip(vfs,afs):
                vfeat.append(vf['features'])
                afeat.append(af['features'])
            if(len(afeat)==0):
                continue
            
            vfeat = torch.Tensor(vfeat).to(device).unsqueeze(0)
            afeat = torch.Tensor(afeat).to(device).unsqueeze(0)
            scores = f(vfeat,afeat)
            summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
    mechine_summary = clip2frame(summary)
    mAP,pre,recall = evaluate(mechine_summary,category_dict,args.topk_mAP)
    return mAP,pre,recall

def train(epoch_idx,mAP):
    f.train()
    logging.info('In epoch {}:\n'.format(epoch_idx+1))
    mloss = 0
    for batch_idx, (posv,posa,negv,nega) in tqdm(enumerate(train_loader)):
       
        # pdb.set_trace()
        opt.zero_grad()
        # b,p,dim = axi.shape
        posv = posv.to(device)
        posa = posa.to(device)
        negv = negv.to(device)
        nega = nega.to(device)

        pos_score= f(posv,posa)
        neg_score= f(negv,nega)

        psloss = SLoss(pos_score)
        nsloss= SLoss(neg_score)
        ahloss = rankingloss(pos_score.mean(),neg_score.mean())
        
        loss = ahloss+psloss+nsloss
        # recoder.update('loss', loss.data, epoch_idx*len(train_loader)+batch_idx)
        loss.backward()
        opt.step()
        mloss+=loss.item()

        # loss = attloss+att_visual_i_loss+att_audio_i_loss+att_visual_j_loss+att_audio_j_loss #0.6543
        recoder.update('psloss',psloss.item(),epoch_idx*len(train_loader))
        recoder.update('ahloss',ahloss.item(),epoch_idx*len(train_loader))
        recoder.update('nsloss',nsloss.item(),epoch_idx*len(train_loader))

    recoder.save()
    print("Domain: {}; In epoch {}, [{}/{}]: loss: {:.6f}, max test mAP: {:.4f}, current test mAP: {:.4f}".format(args.domain,epoch_idx+1,batch_idx,len(train_loader),mloss/len(train_loader),maxMap,mAP))

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader_dict = dict(shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    dataset = MILDatasetv2(args.train_path,args.domain,args.num_per_group)
    train_loader = DataLoader(dataset, **loader_dict)
    f = getattr(model,args.FNet)(AM=getattr(model,args.AM)).to(device)
    SLoss = SmoothnessLoss().to(device)
    AHLoss = AdaptiveHingerLoss().to(device)

    rankingloss = Rankingloss().to(device)
    recoder = Recorder('{}_{}'.format(args.dataset, args.domain))
    tools.mkdir_if_missed('./model_param')
    mAP = 0
    maxMap = -1
    opt = optim.SGD(f.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch_idx in range(args.epoch):
        train(epoch_idx,mAP)
        if epoch_idx%args.interval == 0:
            print('========================testing=============================')
            mAP,pre,recall = test()
            if mAP>maxMap:
                torch.save({'fNet':f.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
                maxMap = mAP
            torch.save({'fNet':f.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_final.pth'.format(args.dataset, args.domain)))
            
            # print(mAP)
        dataset.GeneratePairs()
        train_loader = DataLoader(dataset, **loader_dict)

    # Drawer('../fig/{}_{}.pkl'.format(dataset, domain),
        #    '../fig/{}_{}_train_loss.png'.format(dataset, domain))





