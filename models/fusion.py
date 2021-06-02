import torch
import torch.nn as nn
import math
from libs import TensorList
from models.classification_heads import one_hot


# TODO: add the parameter thing
class fuse_prob(nn.Module):
    def __init__(self, dc_factor=4, weight_learn = False):
        super().__init__()
        self.dc_factor = dc_factor
        self.softmax = nn.Softmax(dim=3)
        if weight_learn:
            self.weight_dense = nn.Parameter(torch.Tensor(torch.ones((dc_factor,1))*(1/dc_factor)))
        else:
            self.weight_dense = torch.ones((dc_factor, 1))*(1/dc_factor)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, query_logits):
        wt_norm = self.sm(self.weight_dense)

        query_logits = query_logits.view(query_logits.shape[0], int(query_logits.shape[1]/self.dc_factor)
                                         , self.dc_factor, -1)
        #log(mean(softmax(dense_scores)))
        fused_score = torch.log(torch.sum(self.softmax(query_logits)*wt_norm, dim=2))
        return fused_score


class fuse_score(nn.Module):
    def __init__(self, dc_factor=4, weight_learn=False):
        super().__init__()
        self.dc_factor = dc_factor
        if weight_learn:
            self.weight_dense = nn.Parameter(torch.Tensor((1/dc_factor)*torch.ones((dc_factor, 1))))
        else:
            self.weight_dense = torch.ones((dc_factor, 1))*(1/dc_factor)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, query_logits):
        wt_norm = self.sm(self.weight_dense)
        query_logits = query_logits.view(query_logits.shape[0], int(query_logits.shape[1]/self.dc_factor),
                                         self.dc_factor, -1)
        fused_score = torch.sum(query_logits*wt_norm, dim=2)
        return fused_score

