
import torch
import logging
import torch.optim as optim
# import opts
import os
import tools
import pdb
from torch.utils.data import DataLoader
from loss import *
from dataset import Pairs
from Recorder import Recorder, Drawer
from tools import clip2frame
import numpy as np
from opts import args
from tqdm import tqdm
from collections import defaultdict
from evaluate import evaluate
import torch.optim.lr_scheduler as lr_scheduler
import model
import random
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
def addGaussianNoise(x,sigma=0.01):
    if random.random() < 0.5:
        return x
    return x+torch.zeros_like(x.data).normal_()*sigma

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
            for vf,af in zip(vfs,afs):
                vseg = vf['segment']
                aseg = af['segment']
                visual = torch.Tensor(vf['features']).to(device).view(-1,args.feature_dim).contiguous()
                audio = torch.Tensor(af['features']).to(device).view(-1,args.audio_dim).contiguous()
                scores.append(f(visual,audio)[0].item())
                
            summary[ve] = scores
    mechine_summary = clip2frame(summary)
    mAP,pre,recall = evaluate(mechine_summary,category_dict,args.topk_mAP)
    return mAP,pre,recall

def train(epoch_idx,mAP):
    f.train()
    h.train()
    logging.info('In epoch {}:\n'.format(epoch_idx+1))
    for batch_idx, (vxi,vxj,axi, axj) in enumerate(train_loader):
        opt.zero_grad()
        b,p,dim = axi.shape
        vxi = vxi.to(device).view(-1,args.feature_dim).contiguous()
        vxj = vxj.to(device).view(-1,args.feature_dim).contiguous()
        axi = axi.to(device).view(-1,dim).contiguous()
        axj = axj.to(device).view(-1,dim).contiguous()
        # vxi = addGaussianNoise(vxi)
        # vxj = addGaussianNoise(vxj)
        # axi = addGaussianNoise(axi)
        # axj = addGaussianNoise(axj)

        visual_attouti,audio_attouti,vouti,aouti,visual_attoi,audio_attoi,voi,aoi = f(vxi,axi)
        visual_attoutj,audio_attoutj,voutj,aoutj,visual_attoj,audio_attoj,voj,aoj = f(vxj,axj)

        # torch.topk(fxi,4)
        # with torch.no_grad():
        b,feature_dim = visual_attoi.shape
        attij = torch.cat((visual_attoi, visual_attoj), dim=-1).cuda().view(-1,feature_dim*2).contiguous()
            # b,feature_dim = voi.shape
            # vij = torch.cat((voi, voj), dim=-1).cuda().view(-1,feature_dim*2).contiguous()
            # b,feature_dim = aoi.shape
            # aij = torch.cat((aoi, aoj), dim=-1).cuda().view(-1,feature_dim*2).contiguous()
        w = h(attij).view(-1,args.num_per_group)
        visaul_attloss = limloss(visual_attouti, visual_attoutj, w)
        audio_attloss = limloss(audio_attouti, audio_attoutj, w)
        vai_loss = mseloss(visual_attouti,audio_attouti)
        vaj_loss = mseloss(visual_attoutj,audio_attoutj)

        vloss = limloss(vouti, voutj, w)
        # print(vouti,voutj)
        aloss = limloss(aouti, aoutj, w)
        
        # att_visual_i_loss = rankingloss(attouti,vouti)
        # att_audio_i_loss = rankingloss(attouti,aouti)

        # att_visual_j_loss = rankingloss(attoutj,voutj)
        # att_audio_j_loss = rankingloss(attoutj,aoutj)

        # visual_att_j_loss = limloss(vouti, attoutj, w)
        # audio_att_j_loss = limloss(aouti, attoutj, w)
        # loss= attloss+vloss+att_visual_i_loss+att_visual_j_loss+visual_att_j_loss
        # loss = visaul_attloss+audio_attloss+vloss+aloss+att_visual_i_loss+att_audio_i_loss+att_visual_j_loss+att_audio_j_loss  #0.6973
        loss = visaul_attloss+audio_attloss+vai_loss+vaj_loss+vloss+aloss
        # loss = attloss+att_visual_i_loss+att_audio_i_loss+att_visual_j_loss+att_audio_j_loss #0.6543
        # recoder.update('attloss',attloss.item(),epoch_idx*len(train_loader))
        # recoder.update('vloss',vloss.item(),epoch_idx*len(train_loader))
        # recoder.update('aloss',aloss.item(),epoch_idx*len(train_loader))
        # recoder.update('att_visual_i_loss',att_visual_i_loss.item(),epoch_idx*len(train_loader))
        # recoder.update('att_audio_i_loss',att_audio_i_loss.item(),epoch_idx*len(train_loader))
        # recoder.update('att_visual_j_loss',att_visual_j_loss.item(),epoch_idx*len(train_loader))
        # recoder.update('att_audio_j_loss',att_audio_j_loss.item(),epoch_idx*len(train_loader))

        # pdb.set_trace()
        # loss = attloss+vloss+aloss
        print("Domain: {}; In epoch {}, [{}/{}]: loss: {:.6f}, max test mAP: {:.4f}, current test mAP: {:.4f}".format(args.domain,epoch_idx+1,batch_idx,len(train_loader),loss.item(),maxMap,mAP))
        # recoder.update('loss', loss.data, epoch_idx*len(train_loader)+batch_idx)
        loss.backward()
        opt.step()
        recoder.save()

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader_dict = dict(shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    # train_loader = DataLoader(Pairs(dataset, domain, short_path, long_path), **loader_dict)
    dataset = Pairs(args.train_path,args.domain,args.num_per_group)
    train_loader = DataLoader(dataset, **loader_dict)
    f = getattr(model,args.FNet)(args.feature_dim,getattr(model,args.AM)).to(device)
    h = getattr(model,args.HNet)(128*2, args.num_per_group).to(device)
    limloss = LIMloss().to(device)
    rankingloss = Rankingloss().to(device)
    mseloss = torch.nn.MSELoss(reduce=True, size_average=True)
    recoder = Recorder('{}_{}'.format(args.dataset, args.domain))
    tools.mkdir_if_missed('./model_param')
    mAP = 0
    maxMap = -1
    opt = optim.SGD(list(f.parameters())+list(h.parameters())+list(limloss.parameters())+list(rankingloss.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(opt,step_size=args.decay_epoch,gamma = args.lr_decay)
    for epoch_idx in range(args.epoch):
        scheduler.step()
        train(epoch_idx,mAP)
        if epoch_idx%args.interval == 0:
            print('========================testing=============================')
            mAP,pre,recall = test()
            if mAP>maxMap:
                torch.save({'fNet':f.state_dict(),'hNet':h.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
                maxMap = mAP
            torch.save({'fNet':f.state_dict(),'hNet':h.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_final.pth'.format(args.dataset, args.domain)))
            
            # print(mAP)
        dataset.GeneratePairs()
        train_loader = DataLoader(dataset, **loader_dict)

    # Drawer('../fig/{}_{}.pkl'.format(dataset, domain),
        #    '../fig/{}_{}_train_loss.png'.format(dataset, domain))





