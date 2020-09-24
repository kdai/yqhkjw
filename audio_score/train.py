import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
import pdb
import torch.optim as optim
from opts import args
from collections import defaultdict
import os
import model
import tools
from tqdm import tqdm
from dataset import Pairs
from evaluate import evaluate
from recorder import Recorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(epoch_idx):
    f.train()
    h.train()
    for batch_idx, (xi, xj) in enumerate(train_loader):
        opt.zero_grad()
        xi = xi.to(device)
        xj = xj.to(device)
        # fxi = f(xi)
        # fxj = f(xj)
        fxi = torch.stack([f(torch.unsqueeze(xi[:, N, :, :], dim=1)) for N in range(args.num_per_group)]).permute(1, 0,
                                                                                                                  2)
        fxj = torch.stack([f(torch.unsqueeze(xj[:, N, :, :], dim=1)) for N in range(args.num_per_group)]).permute(1, 0,
                                                                                                                  2)
        with torch.no_grad():
            tmp_i = torch.flatten(xi, start_dim=2)  # batch x num_per_group x (20 x 23)
            tmp_j = torch.flatten(xj, start_dim=2)
            xij = torch.cat((tmp_i, tmp_j), dim=2)
        w = h(xij)
        loss = limloss(fxi, fxj, w)
        logging.info("In epoch {}, [{}/{}]: loss: {:.6f}".format(epoch_idx, batch_idx, len(train_loader), loss.item()))
        # logging.info("\ncurrent mAP: {:.4f}".format(mAP))
        # recoder.update("loss", loss.item(), epoch_idx)
        opt.step()


def test():
    summary = defaultdict(float)
    f.eval()
    gt_dict = np.load(args.test_path + '/gt_dict.npy', allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())
    total_features = np.load(os.path.join(args.test_path, "feature", "{}_audio.npy".format(args.domain)),
                             allow_pickle=True).tolist()
    with torch.no_grad():
        for v in tqdm(videos):
            features = total_features[v]
            scores = []
            for feature in features:
                x = feature["features"]
                logging.info("size of input: {}".format(x.shape))
                if x.shape != (20, 23):
                    if np.size(x) < 20 * 23:
                        x = np.pad(x, ((0, 20 - x.shape[0]), (0, 23 - x.shape[1])), mode="constant")
                    else:
                        x = x[0:20, 0:23]
                x = torch.unsqueeze(torch.unsqueeze(torch.Tensor(x), dim=0), dim=0).to(device)
                scores.append(f(x).item())
            summary[v] = scores
    machine_summary = tools.clip2frame(summary)
    return evaluate(machine_summary, category_dict, args.topk_mAP)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loader_dict = dict(shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    # train_path = os.path.join("/home/xuanteng/Highlight/proDataset/TrainingSet")
    dataset = Pairs(args.train_path, args.domain, args.num_per_group)
    train_loader = DataLoader(dataset, **loader_dict)
    # recoder = Recorder("{}_{}".format(args.dataset, args.domain))
    f = model.FNet().to(device)
    h = model.HNet().to(device)
    limloss = model.LIMloss().to(device)

    opt = optim.SGD(list(f.parameters()) + list(h.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    mAPs = []
    for epoch_idx in range(args.epoch):
        # logging.info("in epoch {}:\n".format(epoch_idx))
        train(epoch_idx)
        if epoch_idx % args.interval == 0:
            print("===============testing===============")
            mAP, pre, rec = test()
            logging.info("current mAP: {:.4f}".format(mAP))
            mAPs.append(mAP)
            dataset.GeneratePairs()
            train_loader = DataLoader(dataset, **loader_dict)
    with open('./tmp.txt', 'w') as fp:
        for item in mAPs:
            fp.writelines(str(item) + '\n')
