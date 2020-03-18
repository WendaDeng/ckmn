import os
import torch
import torch.nn as nn
import pandas as pd

from . import object_detector_network
from . import scene_detector_network
from . import action_detector_network


class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        self.concept_number = opt.object_feature_dim + opt.action_classes
        self.latent_dimension = 1024
        self.num_class = opt.event_classes

        self.action_detector = action_detector_network.Action_Detector(opt)

        self.concat_reduce_dim = nn.Linear(self.concept_number, self.latent_dimension)
        self._add_object_ft_layer(opt.object_feature_dim)
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

    def _add_object_ft_layer(self, input_dim):
        self.obj_fc1 = nn.Linear(input_dim, input_dim)
        self.obj_fc2 = nn.Linear(input_dim, input_dim)


    def _add_classification_layer(self, input_dim):
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
        else:
            self.final_classifier = nn.Linear(input_dim, self.num_class)


    def forward(self, action_frame, object_feature):
        '''
        :param action_frame:
        :param object_feature: tensor of shape [N T F]
        :return:
        '''

        # N: batch size;    T: segment number;  D: sample duration
        N, T, D, C, aH, aW = action_frame.size()

        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.view(-1, C, D, aH, aW)
        # NT C D aH aW ->  NT F
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F -> N T F
        action_feature = action_feature.view(N, T, -1)

        object_feature = self.relu(object_feature)
        object_feature = self.dropout(object_feature)
        object_feature = self.obj_fc1(object_feature)
        object_feature = self.relu(object_feature)
        object_feature = self.dropout(object_feature)
        object_feature = self.obj_fc2(object_feature)
        
        ## max pooling N T F -> N F
        object_feature, _ = torch.max(object_feature, dim=1)
        action_feature, _ = torch.max(action_feature, dim=1)

        ## concat & classification  size: (N self.concept_number)
        classification = torch.cat((object_feature, action_feature), 1)
        del object_feature, action_feature

        classification = self.relu(classification)
        classification = self.dropout(classification)
        classification = self.concat_reduce_dim(classification)

        classification = self.relu(classification)
        classification = self.dropout(classification)
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            base_out_verb = self.fc_verb(classification)
            base_out_noun = self.fc_noun(classification)

            output = (base_out_verb, base_out_noun)
        else:
            output = self.final_classifier(classification)

        return output
