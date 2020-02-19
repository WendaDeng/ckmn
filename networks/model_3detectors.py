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

        self.concept_number = opt.scene_classes + opt.object_classes + opt.action_classes
        self.latent_dimension = 512
        self.num_class = opt.event_classes

        self.scene_detector = scene_detector_network.Scene_Detector(opt)
        self.object_detector = object_detector_network.Object_Detector(opt)
        self.action_detector = action_detector_network.Action_Detector(opt)
	
        self.concat_reduce_dim = nn.Linear(self.concept_number, self.latent_dimension)
        self._add_classification_layer(self.latent_dimension)
        # self.final_classifier = nn.Linear(self.latent_dimension, opt.event_classes)
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


    def _add_classification_layer(self, input_dim):
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
        else:
            self.final_classifier = nn.Linear(input_dim, self.num_class)


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
        
        ## max pooling
        scene_feature, _ = torch.max(scene_feature, dim=1)
        object_feature, _ = torch.max(object_feature, dim=1)
        action_feature, _ = torch.max(action_feature, dim=1)

        ## concat & classification
        classification = torch.cat((torch.cat((scene_feature, object_feature), 1), action_feature), 1)
        del scene_feature, object_feature, action_feature

        classification = self.relu(classification)
        classification = self.dropout(classification)
        classification = self.concat_reduce_dim(classification)

        classification = self.relu(classification)
        classification = self.dropout(classification)

        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(classification)

            # Noun
            base_out_noun = self.fc_noun(classification)

            output = (base_out_verb, base_out_noun)
        else:
            output = self.final_classifier(classification)

        return output
