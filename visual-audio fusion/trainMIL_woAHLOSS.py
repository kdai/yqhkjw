
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

    summary = defaultdict(float)
    f.eval()
    gt_dict = np.load(args.test_path+'/gt_'+args.dataset +
                      '.npy', allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())

    video_features = np.load(
        args.test_path+'/feature/'+args.domain+'_1s.npy', allow_pickle=True).tolist()
    audio_features = np.load(args.test_path+'/feature/'+args.domain +
                             '_audio_edited_nopost.npy', allow_pickle=True).tolist()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            prefix = ve.split('.')[0]
            vfs = video_features[prefix]
            afs = audio_features[prefix]
            scores = []
            vfeat = []
            afeat = []

            for vf, af in zip(vfs, afs):
                vfeat.append(vf['features'])
                afeat.append(af['features'])
            if(len(afeat) == 0):
                continue

            vfeat = torch.Tensor(vfeat).to(device).unsqueeze(0)
            afeat = torch.Tensor(afeat).to(device).unsqueeze(0)
            scores, logits = f(vfeat, afeat)
            # pdb.set_trace()
            summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
    mechine_summary = clip2frame(summary)
    mAP, pre, recall = evaluate(mechine_summary, category_dict, 1)
    return mAP, pre, recall


def TVsumOrCoSumtest():

    summary = defaultdict(float)
    f.eval()
    gt_dict = np.load(args.test_path+'/gt_'+args.dataset +
                      '.npy', allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())

    video_features = np.load(
        args.test_path+'/feature/'+args.domain+'_1s.npy', allow_pickle=True).tolist()
    audio_features = np.load(args.test_path+'/feature/'+args.domain +
                             '_audio_edited_nopost.npy', allow_pickle=True).tolist()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            prefix = ve.split('.')[0]
            vfs = video_features[prefix]
            afs = audio_features[prefix]
            scores = []
            vfeat = []
            afeat = []
            for vf, af in zip(vfs, afs):
                vfeat.append(vf['features'])
                afeat.append(af['features'])
            if(len(afeat) == 0):
                continue
            vfeat = torch.Tensor(vfeat).to(device).unsqueeze(0)
            afeat = torch.Tensor(afeat).to(device).unsqueeze(0)
            scores, logits = f(vfeat, afeat)
            # pdb.set_trace()
            summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
    mechine_summary = clip2segment(summary, category_dict)
    mAP_5, pre, recall = TVsumOrCoSumEvaluate(
        mechine_summary, category_dict, 5)
    mAP_15, _, __ = TVsumOrCoSumEvaluate(mechine_summary, category_dict, 15)
    return mAP_5, mAP_15


def train(epoch_idx, mAP):
    f.train()
    logging.info('In epoch {}:\n'.format(epoch_idx+1))
    for batch_idx, (posv, posa, negv, nega, pos_label, neg_label) in enumerate(train_loader):
        opt.zero_grad()
        # b,p,dim = axi.shape
        posv = posv.to(device)
        posa = posa.to(device)
        negv = negv.to(device)
        nega = nega.to(device)
        b1, _, _ = posv.shape
        vfeat = torch.cat((posv, negv), 0)
        afeat = torch.cat((posa, nega), 0)

        pos_label = pos_label.to(device)
        neg_label = neg_label.to(device)
        label = torch.cat((pos_label, neg_label), 0).long().view(-1)
        # pdb.set_trace()
        ins_scores, bag_predict = f(vfeat, afeat)
        # print(ins_scores)
        celoss = CELoss(bag_predict, label)

        # ahloss = AHLoss(ins_scores[:b1,:],ins_scores[b1:,:])
        # pdb.set_trace()
        # loss = celoss+ahloss
        loss = celoss

        if args.dataset != "youtube":
            print("Domain: {}; In epoch {}, [{}/{}]: loss: {:.6f}, max mAP_5: {:.4f}, mAP_15: {:.4f}, current mAP_5: {:.4f}, mAP_15: {:.4f}".format(
                args.domain, epoch_idx+1, batch_idx, len(train_loader), loss.item(), max_mAP_5, max_mAP_15, mAP_5, mAP_15))
        else:
            print("Domain: {}; In epoch {}, [{}/{}]: loss: {:.6f}, max test mAP: {:.4f}, current test mAP: {:.4f}".format(
                args.domain, epoch_idx+1, batch_idx, len(train_loader), loss.item(), maxMap, mAP))
        # recoder.update('loss', loss.data, epoch_idx*len(train_loader)+batch_idx)
        loss.backward()
        opt.step()
        # loss = attloss+att_visual_i_loss+att_audio_i_loss+att_visual_j_loss+att_audio_j_loss #0.6543
        # recoder.update('celoss',celoss.item(),epoch_idx)
        # recoder.update('ahloss',ahloss.item(),epoch_idx)

        # recoder.save()


if __name__ == '__main__':
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader_dict = dict(shuffle=True, batch_size=args.batch_size,
                       pin_memory=True, num_workers=4)
    dataset = getattr(dataset, args.DS)(
        args.train_path, args.domain, args.num_per_group)
    train_loader = DataLoader(dataset, **loader_dict)
    f = getattr(model, args.FNet)(AM=getattr(model, args.AM)).to(device)
    CELoss = torch.nn.CrossEntropyLoss()
    # AHLoss = getattr(loss,args.AHLoss)().to(device)

    rankingloss = Rankingloss().to(device)
    recoder = Recorder('{}_{}'.format(args.dataset, args.domain))
    tools.mkdir_if_missed('./model_param')
    mAP = 0
    mAP_5 = 0.0
    mAP_15 = 0.0
    maxMap = -1
    max_mAP_5 = -1.0
    max_mAP_15 = -1.0
    opt = optim.SGD(f.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch_idx in range(args.epoch):
        train(epoch_idx, mAP)
        if epoch_idx % args.interval == 0:
            print('========================testing=============================')
            if args.dataset != 'youtube':
                mAP_5, mAP_15 = TVsumOrCoSumtest()
            else:
                mAP, pre, recall = test()
            if mAP > maxMap:
                torch.save({'fNet': f.state_dict(), 'args': args}, os.path.join(
                    './model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
                maxMap = mAP
            if mAP_5 > max_mAP_5:
                max_mAP_5 = mAP_5
                torch.save({'fNet': f.state_dict(), 'args': args}, os.path.join(
                    './model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
            if mAP_15 > max_mAP_15:
                max_mAP_15 = mAP_15
                torch.save({'fNet': f.state_dict(), 'args': args}, os.path.join(
                    './model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
            torch.save({'fNet': f.state_dict(), 'args': args}, os.path.join(
                './model_param', '{}_{}_final.pth'.format(args.dataset, args.domain)))

            # print(mAP)
        dataset.GeneratePairs()
        train_loader = DataLoader(dataset, **loader_dict)

    # Drawer('../fig/{}_{}.pkl'.format(dataset, domain),
        #    '../fig/{}_{}_train_loss.png'.format(dataset, domain))
