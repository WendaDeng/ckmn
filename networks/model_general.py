import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import general_res
from . import general_i3d

import numpy as np

import os
import sys


class General_Model(nn.Module):
    def __init__(self, opt):
        super(General_Model, self).__init__()

        if opt.general_base_model == 'resnet50':
            weight_dir = '../weights/resnet-50-kinetics.pth'
            model = general_res.resnet50(opt, weight_dir)
        elif opt.general_base_model == 'resnet101':
            weight_dir = '../weights/resnet-101-kinetics.pth'
            model = general_res.resnet101(opt, weight_dir)
        elif opt.general_base_model == 'resnext101':
            weight_dir = '../weights/resnext-101-kinetics.pth'
            model = general_res.resnext101(opt, weight_dir)
        elif opt.general_base_model == 'I3D':
            weight_dir = '../weights/rgb_imagenet.pt'
            model = general_i3d.I3D(opt, weight_dir)
        else:
            print('Error, no such model')

        self.model = model

    def forward(self, action_frame):
        N, T, C, D, aH, aW = action_frame.size()
        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.permute(0, 2, 1, 3, 4, 5)
        action_frame = action_frame.contiguous().view(N, C, -1, aH, aW)

        general_feature = self.model(action_frame)
       
        return general_feature
