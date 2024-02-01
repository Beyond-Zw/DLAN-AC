import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import math
from collections import OrderedDict
import copy
import time
import random
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def dismap(x, name='dismap'):
    # import pdb;pdb.set_trace()
    x = x.data.cpu().numpy()
    x = x.mean(1)
    for j in range(x.shape[0]):
        plt.cla()
        y = x[j]
        # import pdb;pdb.set_trace()
        df = pd.DataFrame(y)
        sns.heatmap(df)
        plt.savefig('results/dismap/{}_{}.png'.format(name,str(j)))
        plt.close()
    return True

def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))



def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    # return score_smoothing(anomaly_score_list)
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def draw_score_curve(aa, bb, cc, cur_gt, name='results/curves_pt', vid = ''):
    
    T = range(len(aa))
    xnew = np.linspace(0,len(aa),10*len(aa))
    aa_new = 1-np.array(aa)
    aa_new = moving_average(aa_new,5)
    bb_new = 1-np.array(bb)
    bb_new = moving_average(bb_new,5)
    cc_new = 1-np.array(cc)
    cc_new = moving_average(cc_new,5)
    # cur_gt = make_interp_spline(T, cur_gt)(xnew)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cur_ans = cc_new
    # print(vid)
    # import pdb;pdb.set_trace()
    ax1.plot(cur_gt, color='r')
    ax2.plot(cur_ans, color='g')
    plt.title(vid)
    plt.show()
    plt.savefig(name+'/'+vid+'_all.png')
    # print('Save: ',root +'/'+vid+'.png')
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cur_ans = aa_new
    # print(vid)
    # import pdb;pdb.set_trace()
    ax1.plot(cur_gt, color='r')
    ax2.plot(cur_ans, color='g')
    plt.title(vid)
    plt.show()
    plt.savefig(name+'/'+vid+'_fra.png')
    # print('Save: ',root +'/'+vid+'.png')
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cur_ans = bb_new
    # print(vid)
    # import pdb;pdb.set_trace()
    ax1.plot(cur_gt, color='r')
    ax2.plot(cur_ans, color='g')
    plt.title(vid)
    plt.show()
    plt.savefig(name+'/'+vid+'_fea.png')
    # print('Save: ',root +'/'+vid+'.png')
    plt.close()
    # import pdb;pdb.set_trace()
    return True

def depict(videos_list, psnr_list, feature_distance_list, labels_list, root='results/AUCs'):
    video_num = 0
    label_length = 0
    import pdb
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        start = label_length
        end = label_length + len(psnr_list[video_name])
        # anomaly_score_total_list = score_sum(anomaly_score_list(psnr_list[video_name]), 
        #                                  anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)
        anomaly_score_ae_list = np.asarray(anomaly_score_list(psnr_list[video_name]))
        anomaly_score_mem_list = np.asarray(anomaly_score_list_inv(feature_distance_list[video_name]))
        if (1-labels_list[start:end]).max() <1 or (1-labels_list[start:end]).min()==1:
            accuracy_ae = accuracy_me = 0
        else:
            accuracy_ae = AUC(anomaly_score_ae_list, np.expand_dims(1-labels_list[start:end], 0))
            accuracy_me = AUC(anomaly_score_mem_list, np.expand_dims(1-labels_list[start:end], 0))
        assert len(labels_list[start:end])==len(anomaly_score_ae_list)
        # pdb.set_trace()
        label_length = end
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #print vid, tf_idf
        ax1.plot(1-labels_list[start:end], color='r')
        ax2.plot(anomaly_score_ae_list, color='g')
        ax3.plot(anomaly_score_mem_list, color='b')
        plt.title(video_name+' {:.4f} {:.4f}'.format(accuracy_ae, accuracy_me), y=3.4)
        plt.show()
        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(root+'/'+video_name+'.png')
        # print('Save: ',root +'/'+vid+'.png')
        plt.close()
    # pdb.set_trace()
    return True



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)

def score_smoothing_SV_AE(score,sigma=2):
    # print('score:',score)
    # r = score.shape[0] //39
    # if r%2==0:
    #     r+=1
    score = np.array(score)
    r = 5
    if r > score.shape[0] // 2:
        r = score.shape[0] // 2 - 1
    if r % 2 == 0:
        r += 1
    gaussian_temp=np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i]=np.exp(-(i-r+1)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score=score
    for i in range(r,score.shape[0]-r):
        new_score[i]=np.dot(score[i-r:i+r-1],gaussian_temp)
    # print('new_score:',new_score)
    return new_score.tolist()

def score_smoothing_ce_SV_AE(score,sigma=2):
    score = score.tolist()
    s1 = score[0]
    ls = []
    for j in range(1000):
        ls.append(s1)

    score = ls + score

    a = score
    r = 2
    if r % 2 == 0:
        r += 1
    gaussian_temp=np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i]=np.exp(-(i-r+1)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score = score
    for i in range(r+1,len(score)):
        new_score[i]=np.dot(score[i-r-1:i+r-2],gaussian_temp)
    new_score = np.array(new_score)
    b = new_score[::]
    print('a:',a)
    print('b',b)
    return new_score[1000::]
    # return new_score

def score_smoothing(score,sigma=2):
    # print('score:',score)
    # r = score.shape[0] //39
    # if r%2==0:
    #     r+=1
    r = 5
    if r > score.shape[0] // 2:
        r = score.shape[0] // 2 - 1
    if r % 2 == 0:
        r += 1
    gaussian_temp=np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i]=np.exp(-(i-r+1)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score=score
    for i in range(r,score.shape[0]-r):
        new_score[i]=np.dot(score[i-r:i+r-1],gaussian_temp)
    # print('new_score:',new_score)
    return new_score

def score_smoothing_ce(score,sigma=2):
    score = score.tolist()
    s1 = score[0]
    ls = []
    for j in range(1000):
        ls.append(s1)

    score = ls + score

    a = score
    r = 2
    if r % 2 == 0:
        r += 1
    gaussian_temp=np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i]=np.exp(-(i-r+1)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score = score
    for i in range(r+1,len(score)):
        new_score[i]=np.dot(score[i-r-1:i+r-2],gaussian_temp)
    new_score = np.array(new_score)
    b = new_score[::]
    print('a:',a)
    print('b',b)
    return new_score[1000::]
    # return new_score

def manual_seed(seed=2022):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Logger(object):
    def __init__(self, fileN="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()