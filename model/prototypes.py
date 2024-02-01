import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F


class Prototypes(nn.Module):
    def __init__(self, prototypes_size, feature_dim, proto_keys_dim):
        super(Prototypes, self).__init__()
        # Constants
        self.prototypes_size = prototypes_size
        self.feature_dim = feature_dim
        self.proto_keys_dim = proto_keys_dim

    def get_score(self, proto_keys, query):
        bs, n, d = query.size()  # n=w*h
        bs, m, d = proto_keys.size()
        # import pdb;pdb.set_trace()
        score = torch.bmm(query, proto_keys.permute(0, 2, 1))  # b X h X w X m
        score = score.view(bs, n, m)  # b X n X m

        score_query = F.softmax(score, dim=1)
        score_proto = F.softmax(score, dim=2)

        return score_query, score_proto

    def query_loss(self, query, proto_keys, train):

        batch_size, h, w, dims = query.size()  # b X n X d, n=w*h
        query = query.view(batch_size, h * w, dims)
        bs, k, dims = proto_keys.size()
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        if train:
            # Normal constrain
            loss_mse = torch.nn.MSELoss()

            proto_keys = F.normalize(proto_keys, dim=-1)
            # _, softmax_score_proto = self.get_score(keys, query)
            softmax_score_proto, _ = self.get_score(proto_keys, query)

            weighted_proto = softmax_score_proto.unsqueeze(-1) * proto_keys.unsqueeze(1)
            weighted_proto = weighted_proto.sum(2)
            weighted_proto = F.normalize(weighted_proto, dim=-1)

            # maintain the distinction among attribute vectors
            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)

            # 1st 2nd closest proto_keys
            pos = torch.gather(proto_keys, 1, gathering_indices[:, :, :1].repeat((1, 1, dims)))
            neg = torch.gather(proto_keys, 1, gathering_indices[:, :, 1:2].repeat((1, 1, dims)))

            cp_loss = loss_mse(query, pos)
            sp_loss = loss(query, pos, neg)

            return weighted_proto, cp_loss, sp_loss


        else:
            # Normal constrain
            proto_keys = F.normalize(proto_keys, dim=-1)
            softmax_score_proto, _ = self.get_score(proto_keys, query)

            weighted_proto = softmax_score_proto.unsqueeze(-1) * proto_keys.unsqueeze(1)
            weighted_proto = weighted_proto.sum(2)
            weighted_proto = F.normalize(weighted_proto, dim=-1)

            return weighted_proto

    def fuse(self, query, weighted_proto):
        batch_size, h, w, dims = query.size()  # b X h X w X d
        batch_size, n, dims = weighted_proto.size()

        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        weighted_proto_reshape = weighted_proto.contiguous().view(batch_size * n, dims)

        updated_fea = torch.cat((query_reshape, weighted_proto_reshape), dim=1)
        updated_fea = updated_fea.view(batch_size, 32, 32, 2 * dims)
        updated_fea = updated_fea.permute(0, 3, 1, 2)

        return updated_fea

    def forward(self, query, proto_keys, train=True):
        if train:
            # batch_size, dims,h,w = query.size() # b X d X h X w
            query = F.normalize(query, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d

            weighted_proto, cp_loss, sp_loss = self.query_loss(query, proto_keys, train)
            fuse_fea = self.fuse(query, weighted_proto)

            return fuse_fea, cp_loss, sp_loss
        else:
            query = F.normalize(query, dim=1)
            query = query.permute(0, 2, 3, 1)  # b X h X w X d

            weighted_proto = self.query_loss(query, proto_keys, train)
            fuse_fea = self.fuse(query, weighted_proto)

            return fuse_fea




    
    
