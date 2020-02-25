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

        self.action_detector = action_detector_network.Action_Detector(opt)
	
        self.final_classifier = nn.Linear(opt.action_classes, opt.event_classes)
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

        N, T, D, C, aH, aW = action_frame.size()

        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.view(-1, C, D, aH, aW)
        # NT C D H W ->  NT F
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F -> N T F
        action_feature = action_feature.view(N, T, -1)
        
        ## max pooling N T F -> N F
        action_feature, _ = torch.max(action_feature, dim=1)

        ## concat & classification
        classification = self.relu(action_feature)
        classification = self.dropout(classification)
        classification = self.final_classifier(classification)

        return classification
