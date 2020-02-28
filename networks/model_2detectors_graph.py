import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

from . import object_detector_network
from . import scene_detector_network
from . import action_detector_network
from ipdb import set_trace


class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        self.concept_number = opt.scene_classes + opt.object_classes + opt.action_classes
        self.latent_dimension = 512
        self.num_class = opt.event_classes
        self.objects_per_segment = opt.objects_per_segment

        # self.scene_detector = scene_detector_network.Scene_Detector(opt)
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


    def postprocess_detection_results(self, detections, batch_size, segment_num, frame_num):
        # detections: List[Dict] len: batch_size * segment_num * frame_num
        detection_split = [detections[i * frame_num: (i+1) * frame_num] for i in range(batch_size * segment_num)]

        boxes, boxes_features = [], []

        # len of detection_split: batch_size * segment_num
        for preds in detection_split:
            scores_per_segment = []
            boxes_per_segment = []
            boxes_features_per_segment = []

            # len of preds: frame_num; pred is a dict
            for pred in preds:
                boxes_per_segment.append(pred['boxes'])
                boxes_features_per_segment.append(pred['boxes_feature'])
                scores_per_segment.append(pred['scores'])

            boxes_per_segment = torch.cat(boxes_per_segment)
            boxes_features_per_segment = torch.cat(boxes_features_per_segment)
            scores_per_segment = torch.cat(scores_per_segment)

            # select top k objects from a segment
            vals, inds = torch.topk(scores_per_segment, self.objects_per_segment)
            boxes_per_segment = boxes_per_segment[inds]
            boxes_features_per_segment = boxes_features_per_segment[inds]

            boxes.append(boxes_per_segment)
            boxes_features.append(boxes_features_per_segment)

        set_trace()
        # dim: [batch_size * segment_num, frame_num * self.objects_per_frame, 1024]
        boxes_features = torch.stack(boxes_features)

        return  boxes, boxes_features



    def forward(self, obj_frame):
        
        action_frame = obj_frame.permute(0, 1, 3, 2, 4, 5)
        # N: batch size;    T: segment number;  D: sample duration
        N, T, D, C, H, W = obj_frame.size()
        _, _, _, _, aH, aW = action_frame.size()

        # N T D C H W -> NTD C H W
        obj_frame = obj_frame.view(-1, C, H, W)
        # NTD C H W -> NTD (List[Dict])
        detection_results = self.object_detector(obj_frame)
        # features: NTD -> NT 4D F(1024)
        object_boxes, object_features = self.postprocess_detection_results(detection_results, N, T, D)

        # NT 4D F(1024) -> N T 4D F(1024)
        seg_nums, obj_nums, feat_dim = object_features.shape
        object_features =  object_features.view(N, T, -1, feat_dim)
        # N T 4D F(1024) -> N T F(1024)
        object_feature, _ = torch.max(object_features, dim=2)
        # object_feature = torch.mean(object_features, dim=2)


        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.view(-1, C, D, aH, aW)
        # NT C D H W ->  NT F
        action_feature = self.action_detector(action_frame)
        # NT F -> N T F
        action_feature = action_feature.view(N, T, -1)
        # todo: use object proposals to extract feature

        set_trace()
        ## max pooling N T F -> N F
        object_feature, _ = torch.max(object_feature, dim=1)
        action_feature, _ = torch.max(action_feature, dim=1)

        del obj_frame, action_frame
        ## concat & classification
        classification = torch.cat((object_feature, action_feature), 1)
        del object_feature, action_feature

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
