
import torch
import logging
import torch.optim as optim
# import opts
import os
import tools
import pdb
from torch.utils.data import DataLoader
from loss import *
import dataset 
from Recorder import Recorder, Drawer
from tools import *
import numpy as np
from opts import args
from tqdm import tqdm
from collections import defaultdict
from evaluate import *
import model
import loss
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def test():
    domains = ['dog','parkour','gymnastics','surfing','skating','skiing']
    result = defaultdict(float)
    TOTAL_MAP = 0
    for dmidx in range(len(domains)):
        summary = defaultdict(float)
        f.eval()
        gt_dict = np.load(args.test_path+'/gt_'+args.dataset+'.npy',allow_pickle=True).tolist()
        category_dict = gt_dict[domains[dmidx]]
        videos = list(category_dict.keys())
        
        video_features = np.load(args.test_path+'/feature/'+domains[dmidx]+'_1s.npy',allow_pickle=True ).tolist()
        audio_features = np.load(args.test_path+'/feature/'+domains[dmidx]+'_audio_edited_nopost.npy',allow_pickle=True ).tolist()

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
                scores,logits = f(vfeat,afeat,dmidx)
                # pdb.set_trace()
                summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
        mechine_summary = clip2frame(summary)
        mAP,pre,recall = evaluate(mechine_summary,category_dict,args.topk_mAP)
        result[domains[dmidx]] = mAP
        TOTAL_MAP+=mAP
    avg_mAP = TOTAL_MAP/len(domains)
    return result,avg_mAP

def TVsumOrCoSumtest():
    if args.dataset == 'tvsum':
        domains = ['BK','BT','DS','FM','GA','MS','PK','PR','VT','VU']
    if args.dataset == 'cosum':
        domains = ['statue_of_liberty','eiffel_tower','NFL','kids_playing_in_leaves','MLB','excavator_river_cross','notre_dame_cathedral'
                        'surf','bike_polo','base_jump']
    result = defaultdict(float)
    TOTAL_MAP = 0
    for dmidx in range(len(domains)):
        summary = defaultdict(float)
        f.eval()
        gt_dict = np.load(args.test_path+'/gt_'+args.dataset+'.npy',allow_pickle=True).tolist()
        category_dict = gt_dict[domains[dmidx]]
        videos = list(category_dict.keys())

        video_features = np.load(args.test_path+'/feature/'+domains[dmidx]+'_1s.npy',allow_pickle=True).tolist()
        audio_features = np.load(args.test_path+'/feature/'+domains[dmidx]+'_audio_edited_nopost.npy',allow_pickle=True ).tolist()
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
                scores,logits = f(vfeat,afeat,dmidx)
                # pdb.set_trace()
                summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
        mechine_summary = clip2segment(summary,category_dict)
        mAP,pre,recall = TVsumOrCoSumEvaluate(mechine_summary,category_dict,args.topk_mAP)
        result[domains[dmidx]] = mAP
        TOTAL_MAP+=mAP
    avg_mAP = TOTAL_MAP/len(domains)
    return result,avg_mAP



def train(epoch_idx,mAP):
    # return
    f.train()
    logging.info('In epoch {}:\n'.format(epoch_idx+1))
    for batch_idx, (posv,posa,negv,nega,pos_label,neg_label) in enumerate(train_loader):
        opt.zero_grad()
        # b,p,dim = axi.shape
        posv = posv.to(device)
        posa = posa.to(device)
        negv = negv.to(device)
        nega = nega.to(device)
        b1,ds,_,_ = posv.shape
        vfeat = torch.cat((posv,negv),0)
        afeat = torch.cat((posa,nega),0)

        pos_label = pos_label.to(device)
        neg_label = neg_label.to(device)
        label = torch.cat((pos_label,neg_label),0).long().squeeze(-1)
        # pdb.set_trace()
        ins_scores,bag_predicts= f(vfeat,afeat)
        # print(ins_scores)
        loss=0
        ahs = []
        ces = []
        for i in range(ds):
            bag_predict = bag_predicts[i]
            ins_score = ins_scores[i]
            celoss = CELoss(bag_predict,label[:,i])
            ahloss = AHLoss(ins_score[:b1,:],ins_score[b1:,:])
            # ahs.append(ahloss)
            # ces.append(celoss)
            if loss==0:
                loss=celoss+ahloss
            else:
                loss=loss+celoss+ahloss
        # ahs = torch.stack(ahs)
        # ces = torch.stack(ces)
        # loss = torch.mean(ahs)+torch.mean(celoss)
        print("In epoch {}, [{}/{}]: loss: {:.6f}, max avg_mAP: {:.4f}, current test mAP: {:.4f}".format(epoch_idx+1,batch_idx,len(train_loader),loss.item(),maxavgMap,avg_mAP))

        print("current mAP: {};".format(mAP))
        print("max mAP: {};".format(maxMAP))

        # recoder.update('loss', loss.data, epoch_idx*len(train_loader)+batch_idx)
        loss.backward()
        opt.step()
        # loss = attloss+att_visual_i_loss+att_audio_i_loss+att_visual_j_loss+att_audio_j_loss #0.6543
        recoder.update('celoss',celoss.item(),epoch_idx)
        recoder.update('ahloss',ahloss.item(),epoch_idx)
        
        recoder.save()

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader_dict = dict(shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    dataset = getattr(dataset,args.DS)(args.train_path,args.domain,args.num_per_group)
    train_loader = DataLoader(dataset, **loader_dict)
    f = getattr(model,args.FNet)(visual_feat_dim=512,audio_feat_dim=128,AM=getattr(model,args.AM)).to(device)
    CELoss = torch.nn.CrossEntropyLoss()
    AHLoss = getattr(loss,args.AHLoss)().to(device)

    rankingloss = Rankingloss().to(device)
    recoder = Recorder('{}_{}'.format(args.dataset, args.domain))
    tools.mkdir_if_missed('./model_param')
    mAP = 0
    avg_mAP = 0
    maxavgMap = -1
    maxMAP = -1
    opt = optim.SGD(f.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # opt = optim.Adam(f.parameters(), lr=args.lr)

    for epoch_idx in range(args.epoch):
        train(epoch_idx,mAP)
        if epoch_idx%args.interval == 0:
            print('========================testing=============================')
            if args.dataset!='youtube':
                mAP,avg_mAP= TVsumOrCoSumtest()
            else:
                mAP,avg_mAP = test()
            if avg_mAP>maxavgMap:
                torch.save({'fNet':f.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
                maxavgMap = avg_mAP
                maxMAP = mAP
            torch.save({'fNet':f.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_final.pth'.format(args.dataset, args.domain)))
            
            # print(mAP)
        # dataset.GeneratePairs()
        # train_loader = DataLoader(dataset, **loader_dict)

    # Drawer('../fig/{}_{}.pkl'.format(dataset, domain),
        #    '../fig/{}_{}_train_loss.png'.format(dataset, domain))





