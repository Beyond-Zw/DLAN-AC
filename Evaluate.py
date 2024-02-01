import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from sklearn.metrics import roc_auc_score
import random
import glob
import pickle
import argparse
from utils import *

parser = argparse.ArgumentParser(description="DLAN-AC")
parser.add_argument('--gpus', nargs='+', type=str, default='1', help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--lr_step_size', type=int, default=20, help='learning rate step size for parameters')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=list, default=[512], help='channel dimension of the features')
parser.add_argument('--pdim', type=list, default=[512], help='channel dimension of the prototypes')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log1', help='directory of log')
parser.add_argument('--log_name', type=str, default='log1', help='directory of log')
parser.add_argument('--AC_clustering', type=bool, default=False, help='if AC_clustering')

args = parser.parse_args()

manual_seed(2022)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

def auc_cal(epoch_num):

    # args.model_dir = f'exp/{args.dataset_type}/{args.log_name}/model_' + str(epoch_num) + '.pth'
    args.model_dir = "none"
    test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                 transforms.ToTensor(), ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(args.model_dir)['state_dict']
    model.cuda()
    labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')


    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args.dataset_type =='shanghai':
            labels_list = np.append(labels_list, labels[4+label_length:videos[video_name]['length']+label_length])
        else:

            labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])

        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    # inference
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_batch))
        for k,(imgs) in enumerate(test_batch):

            if k == label_length-4*(video_num+1):
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

            imgs = Variable(imgs).cuda()

            outputs = model.forward(args, imgs[:, 0:3 * 4], False)

            mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()

            psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
            pbar.update(1)

    # result_dict = {'psnr': psnr_list}
    # pickle_path = f'./PSNR/{epoch_num}.pkl'
    # with open(pickle_path, 'wb') as writer:
    #     pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)
    # with open(pickle_path, 'rb') as reader:
    #     results = pickle.load(reader)
    # psnr_list = results['psnr']

    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]

        anomaly_score_total_list += anomaly_score_list(psnr_list[video_name])

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

    print('The result of ', args.dataset_type)
    print(f'epoch{epoch_num}-AUC: ', round(accuracy*100,3), '%')
    auc = round(accuracy*100,3)
    # if auc >= 80.0:
    #     result_dict = {'psnr': psnr_list}
    #     pickle_path = f'./PSNR/model_{epoch_num}_{args.log_name}_{auc}_.pkl'
    #     with open(pickle_path, 'wb') as writer:
    #         pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

    return auc



if __name__ == "__main__":
    auc_cal(0)

