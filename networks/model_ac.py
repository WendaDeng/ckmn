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

        self.latent_dimension = 1024
        self.num_class = opt.event_classes
        self.action_detector = action_detector_network.Action_Detector(opt)
        self.concat_reduce_dim = nn.Linear(opt.action_classes, self.latent_dimension)
        self._add_classification_layer(opt.action_classes)
        # self.final_classifier = nn.Linear(opt.action_classes, opt.event_classes)
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


    def forward(self, action_frame):

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

        ## classification
        classification = self.relu(action_feature)
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
