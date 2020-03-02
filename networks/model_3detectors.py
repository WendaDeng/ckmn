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

        self.concept_number = opt.scene_classes + opt.object_classes + opt.action_classes
        self.latent_dimension = 512
        self.num_class = opt.event_classes
        self.use_class_cnt = opt.use_class_cnt

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

        self.cls_cnt = self.get_norm_class_cnt(os.path.join(opt.data_root_path,
                            os.path.join(opt.annotation_path, 'EPIC_full_train_action_labels.pkl')))


    def _add_classification_layer(self, input_dim):
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
        else:
            self.final_classifier = nn.Linear(input_dim, self.num_class)


    def get_norm_class_cnt(self, annotation_file_path):
        labels = pd.read_pickle(annotation_file_path)
        verb_class_cnt, noun_class_cnt = {}, {}
        verb_sum, noun_sum = 0, 0

        for i, row in labels.iterrows():
            metadata = row.to_dict()
            if metadata['verb_class'] not in verb_class_cnt:
                verb_class_cnt[metadata['verb_class']] = 1
            else:
                verb_class_cnt[metadata['verb_class']] += 1
            verb_sum += 1

            if metadata['noun_class'] not in noun_class_cnt:
                noun_class_cnt[metadata['noun_class']] = 1
            else:
                noun_class_cnt[metadata['noun_class']] += 1
            noun_sum += 1

        norm_verb_cnt, norm_noun_cnt = {}, {}
        for k, v in verb_class_cnt.items():
            norm_verb_cnt[k] = v / verb_sum
        for k, v in noun_class_cnt.items():
            norm_noun_cnt[k] = v / noun_sum

        verb_class = list(range(self.num_class[0]))
        noun_class = list(range(self.num_class[1]))
        for i in verb_class:
            if i not in norm_verb_cnt.keys():   # only has 119 verb class, total is 125
                norm_verb_cnt[i] = 1e-5
        for i in noun_class:
            if i not in norm_noun_cnt.keys():   # only has 321 noun class, total is 352
                norm_noun_cnt[i] = 1e-6

        sorted_norm_verb_cnt = [v for k, v in sorted(norm_verb_cnt.items(), key=lambda item: item[0])]
        verb_cls_cnt = torch.tensor(sorted_norm_verb_cnt, dtype=torch.float)

        sorted_norm_noun_cnt = [v for k, v in sorted(norm_noun_cnt.items(), key=lambda item: item[0])]
        noun_cls_cnt = torch.tensor(sorted_norm_noun_cnt, dtype=torch.float)

        return [verb_cls_cnt, noun_cls_cnt]


    def forward(self, sceobj_frame):
        
        action_frame = sceobj_frame.permute(0, 1, 3, 2, 4, 5)
        # N: batch size;    T: segment number;  D: sample duration
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
        # NT C D aH aW ->  NT F
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F -> N T F
        action_feature = action_feature.view(N, T, -1)
        
        ## max pooling
        # N T F -> N F
        scene_feature, _ = torch.max(scene_feature, dim=1)
        object_feature, _ = torch.max(object_feature, dim=1)
        action_feature, _ = torch.max(action_feature, dim=1)

        ## concat & classification  size: (N self.concept_number)
        classification = torch.cat((torch.cat((scene_feature, object_feature), 1), action_feature), 1)
        del scene_feature, object_feature, action_feature

        classification = self.relu(classification)
        classification = self.dropout(classification)
        classification = self.concat_reduce_dim(classification)

        classification = self.relu(classification)
        classification = self.dropout(classification)

        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            base_out_verb = self.fc_verb(classification)
            base_out_noun = self.fc_noun(classification)

            if self.use_class_cnt:
                base_out_verb = base_out_verb * self.cls_cnt[0].to(base_out_verb.device)
                base_out_noun = base_out_noun * self.cls_cnt[1].to(base_out_noun.device)

            output = (base_out_verb, base_out_noun)
        else:
            output = self.final_classifier(classification)

        return output
