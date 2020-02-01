import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

from . import object_detector_network
from . import scene_detector_network
from . import action_detector_network
from .non_local_embedded_gaussian import NONLocalBlock1D

class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        #### parameters
        self.opt = opt
        self.concept_number = opt.scene_classes + opt.object_classes + opt.action_classes
        self.T = opt.segment_number
        self.scene_classes = opt.scene_classes
        self.object_classes = opt.object_classes
        self.action_classes = opt.action_classes
        if opt.general_base_model == 'I3D':
            self.general_feature_dim = 1024
        else:
            self.general_feature_dim = 2048
        
        #### concept detectors
        self.scene_detector = scene_detector_network.Scene_Detector(opt)
        self.object_detector = object_detector_network.Object_Detector(opt)
        self.action_detector = action_detector_network.Action_Detector(opt)
	
        #### nonlocal
        self.scene_nonlocal = NONLocalBlock1D(self.scene_classes)
        self.object_nonlocal = NONLocalBlock1D(self.object_classes)
        self.action_nonlocal = NONLocalBlock1D(self.action_classes)

        self.scene_nonlocal_bn1 = nn.BatchNorm1d(self.T)
        self.object_nonlocal_bn1 = nn.BatchNorm1d(self.T)
        self.action_nonlocal_bn1 = nn.BatchNorm1d(self.T)

        self.scene_avgpool = nn.AdaptiveAvgPool1d(1)
        self.object_avgpool = nn.AdaptiveAvgPool1d(1)
        self.action_avgpool = nn.AdaptiveAvgPool1d(1)

        #### feature reduce dimentation & fusion
        self.reduce_dim = 1024
        self.concat_bn1 = nn.BatchNorm1d(self.concept_number)
        self.concat_reduce_dim = nn.Linear(self.concept_number, self.reduce_dim)
        
        #### classification
        self.final_bn1 = nn.BatchNorm1d(self.reduce_dim)
        self.final_classifier = nn.Linear(self.reduce_dim, opt.event_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        for l in self.children():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)


    def forward(self, sceobj_frame):

        action_frame = sceobj_frame.permute(0, 1, 3, 2, 4, 5)
        N, T, D, C, H, W = sceobj_frame.size()
        _, _, _, _, aH, aW = action_frame.size()

        ## scene and object frame input size N T D C H W
        # N T D C H W -> NTD C H W
        sceobj_frame = sceobj_frame.view(-1, C, H, W)
        # NTD C H W -> NTD F
        scene_feature = self.scene_detector(sceobj_frame)
        object_feature = self.object_detector(sceobj_frame)
        del sceobj_frame
        # NTD F -> N T D F
        scene_feature = scene_feature.view(N, T, D, -1)
        object_feature = object_feature.view(N, T, D, -1)
        # N T D F -> N T F
        scene_feature, _ = torch.max(scene_feature, dim=2)
        object_feature, _ = torch.max(object_feature, dim=2)

        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.view(-1, C, D, aH, aW)
        # NT C D H W ->  NT F
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F -> N T F
        action_feature = action_feature.view(N, T, -1)

        #### self attention
        ## relu
        scene_feature = F.relu(scene_feature)
        object_feature = F.relu(object_feature)
        action_feature = F.relu(action_feature)

        ## bn
        scene_feature = self.scene_nonlocal_bn1(scene_feature)
        object_feature = self.object_nonlocal_bn1(object_feature)
        action_feature = self.action_nonlocal_bn1(action_feature)

        ## permute
        scene_feature = scene_feature.permute(0, 2, 1)
        object_feature = object_feature.permute(0, 2, 1)
        action_feature = action_feature.permute(0, 2, 1)
        
        ## self attention
        scene_feature = self.scene_nonlocal(scene_feature)
        object_feature = self.object_nonlocal(object_feature)
        action_feature = self.action_nonlocal(action_feature)

        ## max pooling
        #scene_feature = scene_feature.permute(0, 2, 1)
        #object_feature = object_feature.permute(0, 2, 1)
        #action_feature = action_feature.permute(0, 2, 1)

        #scene_feature, _ = torch.max(scene_feature, dim=1)
        #object_feature, _ = torch.max(object_feature, dim=1)
        #action_feature, _ = torch.max(action_feature, dim=1)
        
        ## adaptive average pooling
        scene_feature = self.scene_avgpool(scene_feature).squeeze(2)
        object_feature = self.object_avgpool(object_feature).squeeze(2)
        action_feature = self.action_avgpool(action_feature).squeeze(2)

        ## concat & classification
        classification = torch.cat((torch.cat((scene_feature, object_feature), 1), action_feature), 1)
        del scene_feature, object_feature, action_feature

        classification = self.relu(classification)
        classification = self.concat_bn1(classification)
        classification = self.dropout(classification)
        classification = self.concat_reduce_dim(classification)

        classification = self.relu(classification)
        classification = self.final_bn1(classification)
        classification = self.dropout(classification)
        classification = self.final_classifier(classification)

        return classification
