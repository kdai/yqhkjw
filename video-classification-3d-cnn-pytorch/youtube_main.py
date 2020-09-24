import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import cv2
from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from collections import defaultdict
import pdb
def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
# 字节bytes转化kb\m\g
def recursive_get_list(path):
    result = []
    for root,dirs,files in os.walk(path):
        for dr in dirs:
            subre = recursive_get_list(os.path.join(root,dr))
            result+=subre
        for fe in files:
            if 'json' in fe:
               pass
            else:
                result.append(os.path.join(root,fe)) 
    return result


def formatSize(path):
    try:
        bytes = os.path.getsize(path)
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("传入的字节格式不对")
        return "Error"
    # if kb >= 1024:
    #     M = kb / 1024
    #     if M >= 1024:
    #         G = M / 1024
    #         return  G
    #     else:
    #         return M
    # else:
    return kb
if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400
    # mkdir(opt.savedir)
    model = generate_model(opt)
    ctg = opt.video_root.split('/')[-1]
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    # with open(opt.input, 'r') as f:
    #     for row in f:
    #         input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp_'+ctg):
        subprocess.call('rm -rf tmp_'+ctg, shell=True)
    if os.path.exists(opt.savename):
        outputs = np.load(opt.savename).tolist()
    else:
        outputs = defaultdict(list)
    exists_keys = list(outputs.keys())
    if os.path.exists(opt.duration_path):
        mp4_duration = np.load(opt.duration_path).tolist()
    else:
        mp4_duration = defaultdict(float)
    for root,dirs ,files in os.walk(opt.video_root):
        input_files = files
    input_files = recursive_get_list(opt.video_root)
    for video_path in input_files:
        input_file = video_path.split('/')[-1]
        if input_file in exists_keys:
            print('pass: '+input_file)
            continue 
        video_path = os.path.join(opt.video_root, input_file)
        size = formatSize(video_path)
        if size<10:
            print('too small: '+input_file)
            continue
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)  
            if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
                # get方法参数按顺序对应下表（从0开始编号)
                rate = cap.get(5)   # 帧速率
                FrameNumber = cap.get(7)  # 视频文件的帧数
                duration = FrameNumber/rate
                mp4_duration[input_file]=duration
            print(video_path)
            subprocess.call('mkdir tmp_'+ctg, shell=True)
            subprocess.call('ffmpeg -i {} tmp_'.format(video_path)+ctg+'/image_%05d.jpg',shell=True)
            # pdb.set_trace()
            video,clips = classify_video('tmp_'+ctg, input_file, model, opt)
            # print(result['video'])
            # diry = result['video'].split('/')[0]
            # mkdir(opt.savedir+'/'+diry)
            # np.save(opt.savedir+'/'+result['video'],result['clips'])
            # outputs.append(result)
            outputs[video] = clips
            np.save(opt.savename,outputs)
            np.save(opt.duration_path,mp4_duration)
            subprocess.call('rm -rf tmp_'+ctg, shell=True)
        else:
            print('{} does not exist'.format(input_file))

    if os.path.exists('tmp_'+ctg):
        subprocess.call('rm -rf tmp_'+ctg, shell=True)
    # pdb.set_trace()
    np.save(opt.savename,outputs)
    # with open(opt.output, 'w') as f:
    #     json.dump(outputs, f)
