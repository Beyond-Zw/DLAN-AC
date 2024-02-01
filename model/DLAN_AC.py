import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
from sklearn import preprocessing, manifold,datasets
import cv2 as cv
import time
from math import *
from .prototypes import *
from minisom import MiniSom

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


class NetVlad_DRCS(nn.Module):
    """NetVLAD layer implementation with dynamic redundant clustering strategy (DRCS)"""

    def __init__(self, dim, num_clusters, vlad_weights):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVlad_DRCS, self).__init__()
        vlad_weights = vlad_weights.astype(np.float32)
        x, y, d = vlad_weights.shape
        vlad_weights = vlad_weights.reshape(x * y, d)
        self.centers_num_K = num_clusters  # K
        self.centers_num_G = vlad_weights.shape[0] - num_clusters

        self.dim = dim
        self.normalize_input = True

        vlad_weights = torch.from_numpy(vlad_weights)
        self.conv = nn.Conv2d(dim, self.centers_num_K + self.centers_num_G, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(vlad_weights)

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.centers_num_K + self.centers_num_G, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.centers_num_K + self.centers_num_G, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.centers_num_K + self.centers_num_G):                          # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)
        vlad = vlad[:, :self.centers_num_K, :]
        vlad = F.normalize(vlad, p=2, dim=2)                                                    # intra-normalization
        vlad = vlad.view(x.size(0), -1)                                                         # flatten
        vlad = F.normalize(vlad, p=2, dim=1)                                                    # L2 normalize

        return vlad

class Encoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128, n_channel, 64)

    def forward(self, x, skip1, skip2, skip3, AC_clustering=False):

        if AC_clustering:
            tensorConv = x
        else:
            tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5, feature_dim=512, proto_keys_dim=512,
                 AC_clustering=False, side_length_som=5, sigma_som=0.5, som_lr=0.5, neighborhood_function_som='', *arg):
        super(convAE, self).__init__()

        def Outhead(feature_dim):
            return torch.nn.Sequential(
                # torch.nn.ReLU(inplace=False),
                torch.nn.Tanh(),
                torch.nn.Linear(feature_dim, feature_dim)
            )

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        if not AC_clustering:
            self.prototypes_size = arg[0]

            self.vlad_weights = arg[1]

            self.prototypes = Prototypes(self.prototypes_size, feature_dim, proto_keys_dim)
            self.netvlad_drcs = NetVlad_DRCS(feature_dim, self.prototypes_size, self.vlad_weights)

        self.AC = MiniSom(side_length_som, side_length_som, 512, neighborhood_function=neighborhood_function_som, sigma=sigma_som,
                           learning_rate=som_lr, random_seed=2022)
        self.ohead = Outhead(feature_dim)

    def forward(self, args, x, train=True):

        fea, skip1, skip2, skip3 = self.encoder(x)

        if args.AC_clustering:
            B, H, W, D = fea.permute(0, 2, 3, 1).shape
            N = H * W
            fea_ = fea.permute(0, 2, 3, 1).contiguous().view(B, N, D)[0].cpu().detach().numpy()
            fea_ = preprocessing.scale(fea_, axis=0)

            self.AC.train(fea_, 5000, random_order=False, verbose=False)  # 迭代次数不是指将所有样本迭代多少次，而是每次迭代输入一个样本，迭代次数为1024则是指刚好将样本全部迭代一次

            AC_weights = self.AC.get_weights()
            res = []
            for i in range(N):
                row, col = self.AC.winner(fea_[i])
                res.append([row * 4 + col])
            classify = {}
            for i, win in enumerate(res):
                if not classify.get(win[0]):
                    classify.setdefault(win[0], [i])
                else:
                    classify[win[0]].append(i)
            # ************************show AC clustering img*****************
            img_template = np.zeros([32, 32, 3], np.uint8)
            cluster_color_table = {'0': [128, 224, 160], '1': [64, 96, 224], '2': [160, 128, 224], '3': [96, 192, 160], '4': [64, 128, 128], '5': [224, 32, 96], '6': [224, 192, 96], '7': [224, 32, 224], '8': [224, 64, 192], '9': [160, 64, 192], '10': [160, 96, 192], '11': [160, 64, 128], '12': [224, 64, 192], '13': [224, 64, 96], '14': [192, 32, 64], '15': [224, 128, 224], '16': [64, 64, 128], '17': [192, 128, 224], '18': [160, 224, 64], '19': [128, 224, 160], '20': [32, 96, 64], '21': [160, 224, 96], '22': [224, 128, 96], '23': [96, 96, 32], '24': [128, 160, 96], '25': [64, 32, 64], '26': [64, 128, 160], '27': [64, 224, 32], '28': [32, 224, 64], '29': [128, 160, 192], '30': [192, 64, 64], '31': [32, 192, 192], '32': [224, 192, 32], '33': [32, 32, 96], '34': [96, 96, 160], '35': [224, 192, 224], '36': [224, 160, 192], '37': [160, 224, 64], '38': [192, 32, 128], '39': [128, 32, 192], '40': [128, 64, 32], '41': [32, 96, 128], '42': [160, 128, 128], '43': [32, 160, 224], '44': [32, 32, 192], '45': [32, 224, 96], '46': [224, 192, 128], '47': [224, 128, 32], '48': [32, 96, 64], '49': [96, 64, 128], '50': [96, 32, 128], '51': [160, 128, 224], '52': [32, 32, 32], '53': [160, 64, 64], '54': [64, 128, 64], '55': [160, 192, 32], '56': [128, 224, 128], '57': [160, 32, 192], '58': [192, 64, 128], '59': [96, 64, 192], '60': [160, 192, 32], '61': [160, 192, 160], '62': [64, 128, 160], '63': [160, 96, 64], '64': [96, 32, 32], '65': [192, 32, 128], '66': [160, 128, 224], '67': [32, 192, 160], '68': [32, 192, 32], '69': [192, 64, 64], '70': [128, 128, 160], '71': [192, 128, 32], '72': [192, 64, 128], '73': [160, 128, 160], '74': [192, 160, 32], '75': [224, 160, 32], '76': [32, 32, 96], '77': [160, 96, 224], '78': [160, 64, 224], '79': [32, 96, 64], '80': [160, 192, 96], '81': [192, 192, 64], '82': [128, 128, 160], '83': [96, 128, 192], '84': [192, 128, 128], '85': [192, 64, 64], '86': [160, 160, 192], '87': [192, 32, 192], '88': [96, 96, 32], '89': [160, 96, 224], '90': [32, 192, 192], '91': [96, 32, 128], '92': [128, 224, 32], '93': [224, 128, 192], '94': [32, 32, 224], '95': [160, 224, 224], '96': [128, 224, 32], '97': [224, 160, 32], '98': [128, 64, 32], '99': [64, 96, 32], '100': [64, 224, 128], '101': [64, 160, 160], '102': [64, 224, 96], '103': [192, 128, 32], '104': [128, 160, 128], '105': [192, 160, 32], '106': [192, 192, 128], '107': [32, 32, 32], '108': [160, 96, 96], '109': [64, 64, 192], '110': [32, 32, 96], '111': [192, 160, 64], '112': [192, 96, 160], '113': [32, 192, 128], '114': [224, 224, 64], '115': [32, 32, 32], '116': [96, 160, 192], '117': [128, 64, 160], '118': [128, 160, 64], '119': [160, 192, 160], '120': [64, 160, 32], '121': [64, 192, 160], '122': [160, 192, 96], '123': [160, 64, 96], '124': [160, 160, 64], '125': [32, 32, 128], '126': [160, 192, 128], '127': [64, 128, 96], '128': [32, 192, 160], '129': [64, 224, 160], '130': [32, 32, 64], '131': [160, 64, 128], '132': [32, 192, 160], '133': [192, 160, 32], '134': [192, 160, 192], '135': [96, 224, 192], '136': [224, 192, 32], '137': [96, 64, 224], '138': [160, 64, 64], '139': [64, 32, 128], '140': [224, 32, 192], '141': [192, 96, 64], '142': [128, 64, 128], '143': [96, 160, 224], '144': [32, 32, 64], '145': [192, 224, 224], '146': [224, 128, 96], '147': [128, 192, 32], '148': [128, 64, 128], '149': [128, 160, 32], '150': [128, 32, 224], '151': [160, 192, 224], '152': [64, 160, 160], '153': [128, 96, 192], '154': [96, 64, 192], '155': [32, 224, 32], '156': [160, 160, 192], '157': [64, 192, 224], '158': [64, 32, 128], '159': [64, 96, 160], '160': [96, 192, 32], '161': [32, 192, 128], '162': [96, 128, 224], '163': [32, 128, 64], '164': [224, 32, 96], '165': [64, 224, 32], '166': [64, 64, 96], '167': [128, 224, 192], '168': [96, 160, 128]}
            s = 0
            cluster_color_key = list(cluster_color_table.keys())
            for cluster_item in classify.values():
                for fea_point in range(len(cluster_item)):
                    row, col = cluster_item[fea_point] // 32, cluster_item[fea_point] % 32
                    # print(row, col)
                    img_template[row, col, 0] = cluster_color_table[cluster_color_key[s]][0]
                    img_template[row, col, 1] = cluster_color_table[cluster_color_key[s]][1]
                    img_template[row, col, 2] = cluster_color_table[cluster_color_key[s]][2]
                s += 1
            # som_results_dir = os.path.join('./exp', args.dataset_type, 'som_results/img')
            ac_results_dir = os.path.join('./exp', args.dataset_type, f'som_results/img_{args.AC_log_name}')
            if not os.path.exists(ac_results_dir):
                os.makedirs(ac_results_dir)
            img_x = ((x.permute(0, 2, 3, 1)[0].cpu().detach().numpy()[..., 0:3] + 1) / 2)*255
            img_template = cv.resize(img_template,(256,256),interpolation=cv.INTER_NEAREST)
            # cv.imshow('input_img', img_x)
            # cv.waitKey(1)
            cv.imshow("ac_cluster", img_template)
            # cv.imwrite(f"{som_results_dir}/som_fae_cluster_{j}.bmp", img_template)
            # cv.imwrite(f"{som_results_dir}/som_fae_cluster_{j}_gt.bmp", img_x)
            cv.waitKey(1)
            # *********************************************
            AC_cluster_number = len(classify)

        else:

            proto_keys = self.netvlad_drcs(fea)
            batch_size, dims = proto_keys.size()
            proto_keys = proto_keys.view(batch_size, self.prototypes_size, 512)  # todo:先分解成（bs, 512, 10）,再进行维度转换
            proto_keys = self.ohead(proto_keys)

        if train:
            if args.AC_clustering:
                output = self.decoder(fea, skip1, skip2, skip3, args.AC_clustering)
                return output, AC_cluster_number, AC_weights
            else:
                fuse_fea, cp_loss, sp_loss = self.prototypes(fea, proto_keys, train)
                output = self.decoder(fuse_fea, skip1, skip2, skip3)

                return output, cp_loss, sp_loss
        # test
        else:
            fuse_fea = self.prototypes(fea, proto_keys, train)
            output = self.decoder(fuse_fea, skip1, skip2, skip3)
            # print('test')

            return output












