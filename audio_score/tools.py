import os
from shutil import copy
from tqdm import tqdm
import pdb
# import ujson as js
import numpy as np
import cv2
from collections import defaultdict
from opts import args
def mkdir_if_missed(path):
    if not os.path.exists(path):
        os.mkdir(path)


def copy_if_exist(src_file, dest_path):
    if os.path.exists(src_file):
        copy(src_file, dest_path)



def create_mp4_time_dict(path,savepath):
    mp4_time = defaultdict(float)
    for root,dirs,files in os.walk(path):
        for fe in files:
            cap = cv2.VideoCapture(os.path.join(root,fe))
            # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
            if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
                # get方法参数按顺序对应下表（从0开始编号)
                rate = cap.get(5)   # 帧速率
                FrameNumber = cap.get(7)  # 视频文件的帧数
                duration = FrameNumber/rate
                mp4_time[fe]=duration
                print('save:'+fe)
            else:
                print('=========error: '+fe+'==========')
    np.save(savepath,mp4_time)
def removeDim(path):
    gt_dict = np.load(path+'/gt_dict.npy').tolist()
    domains = list(gt_dict.keys())
    for domain in domains:
        gt_domain = gt_dict[domain]
        videos = list(gt_domain.keys())
        for ve in videos:
            print(path+'/feature/'+domain+'/'+ve+'.npy')
            feature = np.load(path+'/feature/'+domain+'/'+ve+'.npy').tolist()[0]
            np.save(path+'/feature/'+domain+'/'+ve+'.npy',feature)
            # pdb.set_trace()
            # print('aaa')
def clip2frame(clip_scores):
    ret = {}
    for video in tqdm(clip_scores.keys()):
        tmp = np.zeros(16*(len(clip_scores[video])+1), dtype=np.float32)
        for idx, score in enumerate(clip_scores[video]):
            tmp[args.frames_per_clip*idx: args.frames_per_clip*(idx+1)] = score
        ret[video] = tmp
    return ret
class DealWithFeatureJson:
    def __init__(self, whole_json_path, sl_file, dest_path):
        self.whole_json_path = whole_json_path
        self.dest_path = dest_path
        with open(sl_file) as f:
            self.sl_dict = js.load(f)  # video -> long or short
        mkdir_if_missed(os.path.join(dest_path, "long_videos"))
        mkdir_if_missed(os.path.join(dest_path, "short_videos"))

    def SpiltJson(self):
        with open(self.whole_json_path, 'r') as f:
            whole_features = js.load(f)
        # pdb.set_trace()
        for item in tqdm(whole_features):
            video_name = item["video"]
            if video_name in self.sl_dict.keys():
                if self.sl_dict[video_name] == "short":
                    with open(os.path.join(self.dest_path, "short_videos", "{}.json".format(video_name)), 'w') as ff:
                        ff.write(js.dumps(item))
                if self.sl_dict[video_name] == "long":
                    with open(os.path.join(self.dest_path, "long_videos", "{}.json".format(video_name)), "w") as ff:
                        ff.write(js.dumps(item))

if __name__ == "__main__":
    removeDim('/home/share/Highlight/proDataset/DomainSpecific')