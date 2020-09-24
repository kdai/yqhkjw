import os
import numpy as np
import cv2
from hdf5storage import loadmat
from tqdm import tqdm
from opts import args
import pdb
from collections import defaultdict

domain = args.domain


#  alias syncds="rsync -av -e ssh fating@172.18.167.17:/data/fating/HighlightDataset/proDataset/ /home/share/Highlight/proDataset/"
def generateGT_from_youtube(path):
    count = 0
    category = []
    for root, dirs, files in os.walk(path + '/video'):
        for dr in dirs:
            category.append(dr)
    ctg_dict = defaultdict(defaultdict)
    for ctg in category:
        video_dir = os.path.join(path + '/video', ctg)
        ret = defaultdict(list)
        # with open('/home/share/Highlight/code/instagram_dataset/video_list/{}_youtube'.format(domain), 'r') as file:
        for root, dirs, files in os.walk(video_dir):
            for video in files:
                count += 1
                print(str(count) + ' processing: ' + video)
                video = video.split('.')[0]
                with open(os.path.join(path + '/label', '{}.json'.format(video))) as label_file:
                    data = js.load(label_file)
                    flag = data[-1]
                    cap = cv2.VideoCapture(os.path.join(video_dir, str(video) + '.mp4'))
                    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    frames = np.zeros(int(n_frames) + 1, dtype=np.int16)
                    # pdb.set_trace()
                    for idx, pair in enumerate(data[0]):
                        if flag[idx] == 1:
                            frames[int(pair[0]): int(pair[1])] = 1
                    ret[str(video) + '.mp4'] = frames[np.newaxis]
        ctg_dict[ctg] = ret
    np.save(path + '/gt_youtube.npy', ctg_dict)
    # return


# def generateGT_from_tvsum(domain):
#     src = os.path.join('/home/share/Highlight/code/less_is_more/ydata-tvsum50.mat')
#     mat = loadmat(src)['tvsum50'][0]  # ndarray (50, )
#     ret = {}
#     for i in mat:
#         cate = i[1][0][0]
#         if cate == domain:
#             video_name = i[0][0][0]+'.mp4'
#             ret[video_name] = i[5].transpose()  # user_annotations, (user_nums x n_frames)
#     return ret
#
# def generateGT_from_summe():
#     src_dir = os.path.join('/home/share/Highlight/code/dataset/SumMe/GT')
#     ret = {}
#     for file in os.listdir(src_dir):
#         new_name = file.replace(' ', '_')
#         os.rename(os.path.join(src_dir, file), os.path.join(src_dir, new_name))
#         data = loadmat(os.path.join(src_dir, new_name))
#         # print(data.keys())
#         key = new_name.split('.')[0]+'.mp4'
#         tmp = data['user_score'].transpose() # (user_nums x n_frames)
#         tmp[tmp > 0] = 1
#         ret[key] = tmp
#     return ret


if __name__ == '__main__':
    GT = generateGT_from_youtube('/home/share/Highlight/proDataset/DomainSpecific')
    # create_label
