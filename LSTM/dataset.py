import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import ujson as js
from random import randint
import os
import pdb


class Feature_Set(Dataset):
    def __init__(self, domain, seq_len, is_short):
        super().__init__()
        self.domain = domain
        self.is_short = is_short
        self.seq_len = seq_len
        self.feature_list = {}
        self.features = os.path.join('/home/share/Highlight/code/instagram_dataset/features/{}.json'.format(domain))
        self.results = os.path.join('/home/share/Highlight/code/instagram_dataset/results/{}.json'.format(domain))
        with open(self.features, 'r') as file1:
            with open(self.results, 'r') as file2:
                features = js.load(file1)
                results = js.load(file2)
                for item in tqdm(features):
                    if len(item['clips']) < self.seq_len + 5:  # ignore those video with less than seq_len + 5 clips
                        continue
                    video = item['video']
                    if (results[video] == 'short' and self.is_short is True) or \
                       (results[video] == 'long' and self.is_short is False):
                        tmp = []
                        for clip in item['clips']:
                            tmp.append(clip['features'])
                        self.feature_list[video] = tmp
        self.len = len(self.feature_list)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        :param idx: the name of video those features should be returned
        :return: feature vector
        """
        return self.feature_list[idx]

    def videos(self):
        return self.feature_list.keys()

class Pairs(Dataset):
    def __init__(self, domain, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.short_set = Feature_Set(domain, self.seq_len, True)
        self.long_set = Feature_Set(domain, self.seq_len, False)
        self.pair = []
        for s_video in self.short_set.videos():
            for l_video in self.long_set.videos():
                s_idx = self.get_random(len(self.short_set[s_video]))
                l_idx = self.get_random(len(self.long_set[l_video]))
                s_tmp = []
                l_tmp = []
                for i in range(s_idx, s_idx + self.seq_len):
                    s_tmp.append(self.short_set[s_video][i])  # 5 x 1 x 512
                for j in range(l_idx, l_idx + seq_len):
                    l_tmp.append(self.long_set[l_video][j])  # 5 x 1 x512
                self.pair.append([s_tmp, l_tmp])
        self.pair_len = len(self.pair)

    def get_random(self, video_clips):
        return randint(1, video_clips - self.seq_len - 2)

    def __len__(self):
        return self.pair_len

    def __getitem__(self, idx):
        """
        :param idx: index, number
        :return: a pair, torch.Tensor, 2 x seq_len x 1 x 512,  default seq_len = 5
        """
        data = self.pair[idx]
        xi = torch.Tensor(data[0])  # n_clips x 512
        xj = torch.Tensor(data[1])  # n_clips x 512

        return (xi, xj) # (n_clips x 512, n_clips x 512)

    def get_pairs(self):
        return self.pair

class TestSet(Dataset):
    def __init__(self, domain, sequence_len):
        self.domain = domain
        self.seq_len = sequence_len
        self.src_file = os.path.join('/home/share/Highlight/code/instagram_dataset/features/{}_t.json'.format(self.domain))
        self.feature_list = {}
        with open(self.src_file, 'r') as file:
            data = js.load(file)
            for item in tqdm(data):
                if len(item['clips']) < self.seq_len + 5:
                    continue
                video = item['video']
                tmp = []
                for i in range(0, len(item['clips'])-self.seq_len, self.seq_len):
                    segs = []
                    for j in range(self.seq_len):
                        segs.append(item['clips'][i+j]['features'])
                    tmp.append(segs)
                self.feature_list[video] = tmp
        self.len = len(self.feature_list)

    def __len__(self):
        return self.len

    def videos(self):
        return self.feature_list.keys()

    def __getitem__(self, idx):
        return torch.Tensor(self.feature_list[idx])

class Alldomain_Set(Dataset):
    def __init__(self, domain_list, seq_len):
        self.pairs = []
        self.seq_len = seq_len
        for domain in domain_list:
            tmp_pair = Pairs(domain, self.seq_len).get_pairs()
            for pair in tmp_pair:
                self.pairs.append(pair)
        self.len = len(self.pairs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.pairs[idx]
        xi = torch.Tensor(data[0])
        xj = torch.Tensor(data[1])
        return (xi, xj)

if __name__ == '__main__':
    domain_list = ['dog', 'skating', 'parkour']
    test = Alldomain_Set(domain_list, 4)
    pdb.set_trace()
    print('finish!')


