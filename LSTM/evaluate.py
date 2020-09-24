import numpy as np
import os
import ujson as js
from normalize import generateGT_from_youtube, generateGT_from_tvsum, generateGT_from_summe, clip2frame
from tqdm import tqdm
from opts import parser_opts
import pdb

arg_opt = parser_opts()
domain = arg_opt.domain
proportion = arg_opt.proportion

def summary(domain, dataset):
    GT = None
    if dataset == 'youtube':
        GT = generateGT_from_youtube(domain)  # GT: dict, (video_name-> GT) GT: binary list, (n_frames)
    if dataset == 'tvsum':
        GT = generateGT_from_tvsum(domain)
    if dataset == 'summe':
        GT = generateGT_from_summe()
    with open(os.path.join('./test_results', '{}.json'.format(domain)), 'r') as file:
        machine_summary = js.load(file)  # machine_summary: dict, (video_name-> clips_scores)
        machine_summary = clip2frame(machine_summary) # dict, (video_name-> np.array)

        human_annotation = []
        predicted_annotation = []

        for video in GT.keys():
            h = GT[video]
            x = machine_summary[video]
            human_annotation.append(h)
            predicted_annotation.append(x)

        print(total_evaluate(predicted_annotation, human_annotation, 1))
        print(total_evaluate(predicted_annotation, human_annotation))



def evaluate_summary(machine_summary, user_summary, topk=5):
    """Compare machine summary with user summary.
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    machine_summary： [1，n]
    user_summary：    [n_user,n_frames]
    """
    # machine_summary = machine_summary.astype(np.int16)
    # user_summary = user_summary.astype(np.int16)
    n_users, n_frames = user_summary.shape

    # binarization
    # machine_summary[machine_summary > 0] = 1
    # user_summary[user_summary > 0] = 1


    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])
    sortIdx = np.argsort(-machine_summary)
    machine_summary = machine_summary[sortIdx]
    idx = int(n_frames*proportion)
    thresh = machine_summary[idx]
    n_user_summary = []
    for i in range(n_users):
        temp = user_summary[i][sortIdx]
        n_user_summary.append(temp)
    user_summary = np.array(n_user_summary)
    APs = []
    avg_precision = []
    avg_recall = []
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        n_good = gt_summary.sum()
        ap = 0.0
        intersect_size = 0.0
        old_recall = 0.0
        old_precision = 1.0
        for j in range(n_frames):
            if gt_summary[j] == 1 and machine_summary[j] >= thresh:
                intersect_size+=1
            recall = intersect_size/n_good
            precision = intersect_size / (j+1)
            ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision
        APs.append(ap)
        avg_precision.append(old_precision)
        avg_recall.append(old_recall)
    aps = np.array(APs)
    avg_precision = np.array(avg_precision)
    avg_recall = np.array(avg_recall)
    aps.sort()
    avg_precision.sort()
    avg_recall.sort()
    topk_mAP = aps[-topk:].mean()
    topk_pre = avg_precision[-topk:].mean()
    topk_rec = avg_recall[-topk:].mean()
    return topk_mAP, topk_pre, topk_rec
def total_evaluate(machine_summary, user_summary, topk=5):
    '''
    machine_summary [n_sample,n]
    user_summary [n_sample,n_user,n_frames]
    '''
    n_sample = len(user_summary)
    mAP = 0.0
    precision = 0.0
    recall = 0.0
    for i in tqdm(range(n_sample)):
        (tmp1, tmp2, tmp3) = evaluate_summary(machine_summary[i], user_summary[i], topk)
        mAP += tmp1
        precision += tmp2
        recall += tmp3
    return mAP/n_sample, precision/n_sample, recall/n_sample


if __name__ == '__main__':
    print('enter dataset: youtube | tvsum | summe?')
    dataset = input()
    summary(domain, dataset)