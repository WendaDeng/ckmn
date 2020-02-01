import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

from . import object_detector_network
from . import scene_detector_network
from . import action_detector_network


class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        self.object_detector = object_detector_network.Object_Detector(opt)
	
        self.final_classifier = nn.Linear(opt.object_classes, opt.event_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        for l in self.children():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, 0, 0.01)
                #nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)


    def forward(self, sceobj_frame, action_frame):

        N, T, D, C, H, W = sceobj_frame.size()

        ## scene and object frame input size N T D C H W
        # N T D C H W -> NTD C H W
        sceobj_frame = sceobj_frame.view(-1, C, H, W)
        # NTD C H W -> NTD F
        object_feature = self.object_detector(sceobj_frame)
        del sceobj_frame
        # NTD F -> N T D F
        object_feature = object_feature.view(N, T, D, -1)
        # N T D F -> N T F
        object_feature, _ = torch.max(object_feature, dim=2)

        ## max pooling
        object_feature, _ = torch.max(object_feature, dim=1)

        ## concat & classification
        classification = self.relu(object_feature)
        classification = self.dropout(classification)
        classification = self.final_classifier(classification)

        return classification
