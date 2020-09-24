import torch
from torch.utils.data import Dataset
# import ujson as js
import random
import pdb
import os
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from opts import args
import pdb
import random
from tqdm import tqdm 

class Pairs(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.features = np.load(train_path+'/'+domain+'.npy',allow_pickle=True).tolist()
        keys = list(self.duration.keys())
        self.short_videos = []
        self.long_videos = []
        for key in tqdm(keys):
            # if self.duration[key]>args.short_lower and self.duration[key]<args.short_upper: 
            #     self.short_videos.append(key)
            # if self.duration[key]>args.long_lower and self.duration[key]<args.long_upper:
            #     self.long_videos.append(key)
            if self.duration[key]<args.short_upper: 
                self.short_videos.append(key)
            if self.duration[key]>args.long_lower:
                self.long_videos.append(key)
        
        self.short_dict = defaultdict(list)
        short_idx = 0
        for short_video in tqdm(self.short_videos):
            features = self.features[short_video]
            feat_len = len(features)
            for i in range(feat_len):
                self.short_dict[short_idx] = [short_video,i]
                short_idx+=1

        self.long_dict = defaultdict(list)
        long_idx = 0
        for long_video in tqdm(self.long_videos):
            # features = np.load(long_video,allow_pickle=True ).tolist()
            features = self.features[long_video]
            feat_len = len(features)
            for i in range(feat_len):
                self.long_dict[long_idx] = [long_video,i]
                long_idx+=1

        self.GeneratePairs() # n x num_per_group x 2
    def GeneratePairsv2(self):
        # pdb.set_trace()
        groups = []
        pairs = []
        short_keys = list(self.short_dict.keys())
        long_keys = list(self.long_dict.keys())

        if len(long_keys)>len(short_keys):
            while len(long_keys) != 0:
                if len(short_keys)==0:
                    short_keys = list(self.short_dict.keys())
                s_key = random.sample(short_keys, 1)[0]
                short_keys.remove(s_key)
                l_key = random.sample(long_keys, 1)[0]
                long_keys.remove(l_key)
                # print(len(long_keys))
                pairs.append((s_key, l_key))
        else:
            while len(short_keys) != 0:
                s_key = random.sample(short_keys, 1)[0]
                short_keys.remove(s_key)
                # print(len(short_keys))

                if len(long_keys)==0:
                    long_keys = list(self.long_dict.keys())
                l_key = random.sample(long_keys, 1)[0]
                long_keys.remove(l_key)
                pairs.append((s_key, l_key))
        
        pairs = set(pairs)
        total_pairs = len(pairs)
        left = total_pairs % self.num_per_group
        while len(pairs) > left:
            group = []
            while len(group) < self.num_per_group:
                group.append(pairs.pop())
            groups.append(group)
        self.groups = groups
        return groups
    def GeneratePairs(self):
        # pdb.set_trace()
        groups = []
        pairs = []
        short_keys = list(self.short_dict.keys())
        long_keys = list(self.long_dict.keys())
        
        if len(long_keys)>len(short_keys):
            
            re = int(len(long_keys)/len(short_keys))
            remainder = len(long_keys)%len(short_keys)
            new_short_keys = []
            for i in range(re):
                new_short_keys+=short_keys
            new_short_keys += random.sample(short_keys, remainder)
            short_keys = new_short_keys
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            pairs = list(zip(short_keys,long_keys))
        else:
            re = int(len(short_keys)/len(long_keys))
            remainder = len(short_keys)%len(long_keys)
            new_lone_keys = []
            for i in range(re):
                new_lone_keys+=long_keys
            new_lone_keys += random.sample(long_keys, remainder)
            long_keys = new_lone_keys
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            pairs = list(zip(short_keys,long_keys))
        # pdb.set_trace()
        total_pairs = len(pairs)
        remainder = total_pairs % self.num_per_group
        left=random.sample(pairs, self.num_per_group-remainder)
        pairs+=left
        self.groups = np.array(pairs).reshape(-1,8,2).tolist()
        return

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        
        XI = torch.Tensor()
        XJ = torch.Tensor()
        group = self.groups[idx] # num_per_group x 2
        for pair in group:
            (s_key, l_key) = pair
            [video,idx] = self.short_dict[s_key]
            xi =  self.features[video][idx]['features']
            [video,idx] = self.long_dict[l_key]
            xj = self.features[video][idx]['features']
            xi = torch.Tensor(xi).view(1, -1)
            xj = torch.Tensor(xj).view(1, -1)
            XI = torch.cat((XI, xi), dim=0)
            XJ = torch.cat((XJ, xj), dim=0)

        return XI, XJ

if __name__ == "__main__":
    feature = np.load('/home/share/Highlight/proDataset/DomainSpecific/feature/dog/msjK8nHZHZ0.mp4.npy').tolist()
    pdb.set_trace()
    test_set = Test('/home/share/Highlight/proDataset/DomainSpecific','dog')
    # train_loader = DataLoader(test_set, shuffle=True, batch_size=16, pin_memory=True, num_workers=8,collate_fn = test_set.collate_fn)
    train_loader = DataLoader(test_set, shuffle=True, batch_size=16, pin_memory=True, num_workers=8)

    for batch_idx, feature,labels,names in enumerate(train_loader):
        print(names)
    # pdb.set_trace()
    # len(t)
    # just 4 testing dataset
