import torch
from model import F
import os
from opts import args
import pdb
import numpy as np
from collections import defaultdict
from tools import clip2frame
from tqdm import tqdm
from evaluate import evaluate
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
f = F(feature_dim = 512)
# f.load_state_dict(torch.load(os.path.join('./save', 'skating_f.pth')))
f.to(device)
def test():
    summary = defaultdict(float)
    f.eval()
    # test_feature_path = args.test_path+'/feature/'+args.domain
    gt_dict = np.load(args.test_path+'/gt_dict.npy',allow_pickle=True).tolist()
    # pdb.set_trace()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())
    total_features = np.load(args.test_path+'/feature/'+args.domain+'.npy',allow_pickle=True ).tolist()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            # feature = np.load(test_feature_path+'/'+ve+'.npy').tolist()
            feature = total_features[ve]
            scores = []
            for feat in feature:
                segment = feat['segment']
                scores.append(f(torch.Tensor(feat['features']).to(device).view(1,-1)).item())
            summary[ve] = scores
    # gt_domain_dict = 
    mechine_summary = clip2frame(summary)
    ret = evaluate(mechine_summary,category_dict,args.topk_mAP)
    print(ret)
    print('aaaa')
if __name__ == '__main__':
    test()

