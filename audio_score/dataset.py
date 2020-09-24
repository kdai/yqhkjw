import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from opts import args
from collections import defaultdict
import random
from tqdm import tqdm
import pdb

class Pairs(Dataset):
    def __init__(self, train_path, domain, num_per_group):
        super(Pairs, self).__init__()
        self.num_per_group = num_per_group
        self.duration = np.load(os.path.join(train_path, "{}_duration.npy".format(domain)), allow_pickle=True).tolist()
        self.audio = np.load(os.path.join(train_path, "{}_audio.npy".format(domain)), allow_pickle=True).tolist()
        keys = list(self.duration.keys())
        self.short_videos = []
        self.long_videos = []
        for key in tqdm(keys):
            if self.duration[key] > args.short_lower and self.duration[key] < args.short_upper:
                self.short_videos.append(key)
            if self.duration[key] > args.long_lower and self.duration[key] < args.long_upper:
                self.long_videos.append(key)
        self.short_dict = defaultdict(list)
        self.long_dict = defaultdict(list)
        short_idx = 0
        for short_video in tqdm(self.short_videos):
            features = self.audio[short_video]
            l = len(features)
            for i in range(l):
                self.short_dict[short_idx] = [short_video, i]  # short_video -> ith segment
                short_idx += 1
        long_idx = 0
        for long_video in tqdm(self.long_videos):
            features = self.audio[long_video]
            l = len(features)
            for i in range(l):
                self.long_dict[long_idx] = [long_video, i]  # long_video -> ith segment
                long_idx += 1
        self.GeneratePairs()

    def GeneratePairs(self):
        # pdb.set_trace()
        groups = []
        pairs = []
        short_keys = list(self.short_dict.keys())
        long_keys = list(self.long_dict.keys())

        if len(long_keys) > len(short_keys):

            re = int(len(long_keys) / len(short_keys))
            remainder = len(long_keys) % len(short_keys)
            new_short_keys = []
            for i in range(re):
                new_short_keys += short_keys
            new_short_keys += random.sample(short_keys, remainder)
            short_keys = new_short_keys
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            pairs = list(zip(short_keys, long_keys))
        else:
            re = int(len(short_keys) / len(long_keys))
            remainder = len(short_keys) % len(long_keys)
            new_lone_keys = []
            for i in range(re):
                new_lone_keys += long_keys
            new_lone_keys += random.sample(long_keys, remainder)
            long_keys = new_lone_keys
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            pairs = list(zip(short_keys, long_keys))
        # pdb.set_trace()
        total_pairs = len(pairs)
        remainder = total_pairs % self.num_per_group
        left = random.sample(pairs, self.num_per_group - remainder)
        pairs += left
        self.groups = np.array(pairs).reshape(-1, self.num_per_group, 2).tolist()
        return

    def __getitem__(self, idx):
        XI = []
        XJ = []
        group = self.groups[idx]
        for pair in group:
            (s_key, l_key) = pair
            [video, idx] = self.short_dict[s_key]
            xi = self.audio[video][idx]["features"]
            # xi = torch.Tensor(xi)  # 20 x 23
            [video, idx] = self.long_dict[l_key]
            xj = self.audio[video][idx]["features"]
            # xj = torch.Tensor(xj)
            XI.append(xi)
            XJ.append(xj)
        XI = torch.Tensor(XI)
        XJ = torch.Tensor(XJ)
        return XI, XJ  # XI: num_per_group x 20 x 23

    def __len__(self):
        return len(self.groups)

if __name__ == "__main__":
    train_path = os.path.join("/home/share/Highlight/proDataset/TrainingSet")
    training_set = Pairs(train_path, "skating", 8)
    train_loader = DataLoader(training_set, shuffle=True, batch_size=16, pin_memory=True, num_workers=8)
    for batch_idx, item in enumerate(train_loader):
        print(batch_idx)
        print(len(item))
        print(item[0].shape, item[1].shape)