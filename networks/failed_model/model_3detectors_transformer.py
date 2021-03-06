import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import ipdb
import math
import numpy as np

from . import object_detector_network
from . import scene_detector_network
from . import action_detector_network
from . import transformer

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

        #### transformer
        self.optimized_concept_dim = 256
        self.action_optimized_concept_dim = 512
        n_head = 1
        d_k = 64
        d_v = 64
        self.scene_encoding = transformer.EncoderLayer(self.scene_classes, self.optimized_concept_dim, n_head, d_k, d_v)
        self.object_encoding = transformer.EncoderLayer(self.object_classes, self.optimized_concept_dim, n_head, d_k, d_v)
        self.action_encoding = transformer.EncoderLayer(self.action_classes, self.action_optimized_concept_dim, n_head, d_k, d_v)

        #### feature reduce dimentation & fusion
        self.reduce_temp_dimentation = 512
        self.reduce_dropout = nn.Dropout(0.5)
        self.concat_reduce_dim = nn.Linear(self.optimized_concept_dim * 2 + self.action_optimized_concept_dim, self.reduce_temp_dimentation)
        # self.concat_reduce_dim = nn.Linear(self.concept_number, self.reduce_temp_dimentation)

        #### classification
        self.final_classifier = nn.Linear(self.reduce_temp_dimentation, opt.event_classes)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)

        for l in self.children():
            if isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(l.weight, mean=0, std=0.01)
                nn.init.constant_(l.bias, 0)


    def forward(self, sceobj_frame, action_frame):
        
        #### concept detector
        N, T, D, C, H, W = sceobj_frame.size()
        _, _, _, _, aH, aW = action_frame.size()
        ## scene and object frame input size N T D C H W
        # N T D C H W -> NTD C H W
        sceobj_frame = sceobj_frame.contiguous().view(-1, C, H, W)
        # NTD C H W -> NTD F
        scene_feature = self.scene_detector(sceobj_frame)
        object_feature = self.object_detector(sceobj_frame)
        del sceobj_frame
        # NTD F -> N T D F
        scene_feature = scene_feature.contiguous().view(N, T, D, -1)
        object_feature = object_feature.contiguous().view(N, T, D, -1)
        # N T D F -> N T F
        # segment level scene object concept feature
        scene_feature, _ = torch.max(scene_feature, dim=2)
        object_feature, _ = torch.max(object_feature, dim=2)
        
        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.contiguous().view(-1, C, D, aH, aW)
        # NT C D H W ->  NT F
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F -> N T F
        # segment level action concept feature
        action_feature = action_feature.contiguous().view(N, T, -1)

        #### encoding feature
        scene_feature_encod = self.scene_encoding(scene_feature)
        object_feature_encod = self.object_encoding(object_feature)
        action_feature_encod = self.action_encoding(action_feature)
        
        ## max pooling
        scene_feature_encod, _ = torch.max(scene_feature_encod, dim=1)
        object_feature_encod, _ = torch.max(object_feature_encod, dim=1)
        action_feature_encod, _ = torch.max(action_feature_encod, dim=1)

        concat_feature = torch.cat((torch.cat((scene_feature_encod, object_feature_encod), 1), action_feature_encod), 1) 

        #### reduce dimentation
        concat_feature = self.relu(concat_feature)
        concat_feature = self.reduce_dropout(concat_feature)
        concat_feature = self.concat_reduce_dim(concat_feature)

        #### classification
        final_classification = self.relu(concat_feature)
        final_classification = self.dropout(final_classification)
        final_classification = self.final_classifier(final_classification)

        return final_classification
