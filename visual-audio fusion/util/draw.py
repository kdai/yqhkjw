import numpy as np
import pdb
import random
import matplotlib.pyplot as plt 
# from scipy import interpolate
import torch.nn.functional as F
import torch
from collections import defaultdict
import matplotlib as mlp
import cv2
import shutil
from matplotlib.pyplot import MultipleLocator
mlp.rcParams['axes.spines.right'] = False
mlp.rcParams['axes.spines.top'] = False
##折线图 (PL,MT,LP,Softmax,Ours/3) test EMS
def task1():
    fontsize = 14
   

    markersize=14
    linewidth=2
    plt.figure(figsize=(10,5))
    x = [20,40,60,80,100]
    youtube = [61.415,62.69,64.36,63.02,62.85]
    tvsum = [72.98,72.72,73.24,71.84,70.55]
    cosum = [88.15,89.49,92.90,91.45,88.12]

    plt.plot(x, youtube, marker='d',label='YouTube', markersize=9,linewidth=linewidth, color='dodgerblue')
    plt.plot(x, tvsum, marker='o',label='TVSum', markersize=9,linewidth=linewidth, color='darkviolet')
    plt.plot(x, cosum, marker='>',label='CoSum', markersize=9,linewidth=linewidth, color='goldenrod')
    # plt.plot(x, test_sm, marker='v',label='Softmax', markersize=9, color='green')
    # plt.plot(x, test_ours_sm_rank_isw_ew, marker='*', label='Ours',markersize=13, color='firebrick')
    # plt.plot(x,test_pl,'.-',label='PL',color="gold",marker='^',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_lp,'.-',label='LP',color="black",marker='>',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_mt,'.-',label='MT',color="firebrick",marker='s',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_sm,'.-',label='Softmax',color="dodgerblue",marker='d',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours',color="limegreen",marker='*',linewidth=linewidth,markersize=markersize) 
 
    
    # x = ["Top-1","Top-2","Top-3","Top-4","Top-5","Top-6","Top-7","Top-8"]
    plt.grid(ls='--')
    plt.xticks(x, x,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # plt.yticks([0.2,0.4,0.5,0.6,0.8], [0.2,0.4,0.5,0.6,0.8],fontsize=fontsize)
    # plt.ylabel(r"mAP/Top-5 mAP (%)",fontsize=12)
    # plt.Xlabel(r"mAP/Top-5 mAP (%)",fontsize=12)

    #显示图示  
    plt.legend()  
    plt.savefig("bagsizeEvaluation.pdf")  
    print('save')
task1()