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

        #### co-attention model
        #self.scene_coattention_bn1 = nn.BatchNorm1d(self.T)
        #self.object_coattention_bn1 = nn.BatchNorm1d(self.T)
        #self.action_coattention_bn1 = nn.BatchNorm1d(self.T)
        
        self.k = 256
        self.Wb_sc_ob = Parameter(torch.Tensor(self.object_classes, self.scene_classes))
        self.Wv_sc_ob = Parameter(torch.Tensor(self.k, self.scene_classes))
        self.Ws_sc_ob = Parameter(torch.Tensor(self.k, self.object_classes))
        self.Whv_sc_ob = Parameter(torch.Tensor(1, self.k))
        self.Whs_sc_ob = Parameter(torch.Tensor(1, self.k))

        self.Wb_sc_ac = Parameter(torch.Tensor(self.action_classes, self.scene_classes))
        self.Wv_sc_ac = Parameter(torch.Tensor(self.k, self.scene_classes))
        self.Ws_sc_ac = Parameter(torch.Tensor(self.k, self.action_classes))
        self.Whv_sc_ac = Parameter(torch.Tensor(1, self.k))
        self.Whs_sc_ac = Parameter(torch.Tensor(1, self.k))

        self.Wb_ob_ac = Parameter(torch.Tensor(self.action_classes, self.object_classes))
        self.Wv_ob_ac = Parameter(torch.Tensor(self.k, self.object_classes))
        self.Ws_ob_ac = Parameter(torch.Tensor(self.k, self.action_classes))
        self.Whv_ob_ac = Parameter(torch.Tensor(1, self.k))
        self.Whs_ob_ac = Parameter(torch.Tensor(1, self.k))

        #### feature reduce dimentation & fusion
        self.reduce_temp_dimentation = 512
        self.reduce_dropout = nn.Dropout(0.5)
        # self.concat_bn1 = nn.BatchNorm1d(self.concept_number*2)
        self.concat_reduce_dim = nn.Linear(self.concept_number*2, self.reduce_temp_dimentation)

        #### classification
        # self.final_bn1 = nn.BatchNorm1d(self.reduce_temp_dimentation)
        self.final_classifier = nn.Linear(self.reduce_temp_dimentation, opt.event_classes)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)

        self._reset_parameters()
        for l in self.children():
            if isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, mean=0, std=0.01)
                nn.init.constant_(l.bias, 0)

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.Wb_sc_ob.size(1))
        self.Wb_sc_ob.data.uniform_(-stdv, stdv)
        self.Wb_sc_ac.data.uniform_(-stdv, stdv)
        self.Wb_ob_ac.data.uniform_(-stdv, stdv)
        self.Wv_sc_ob.data.uniform_(-stdv, stdv)
        self.Wv_sc_ac.data.uniform_(-stdv, stdv)
        self.Wv_ob_ac.data.uniform_(-stdv, stdv)
        self.Ws_sc_ob.data.uniform_(-stdv, stdv)
        self.Ws_sc_ac.data.uniform_(-stdv, stdv)
        self.Ws_ob_ac.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Whv_sc_ob.size(1))
        self.Whv_sc_ob.data.uniform_(-stdv, stdv)
        self.Whv_sc_ac.data.uniform_(-stdv, stdv)
        self.Whv_ob_ac.data.uniform_(-stdv, stdv)
        self.Whs_sc_ob.data.uniform_(-stdv, stdv)
        self.Whs_sc_ac.data.uniform_(-stdv, stdv)
        self.Whs_ob_ac.data.uniform_(-stdv, stdv)

    def forward(self, sceobj_frame):
        
        #### concept detector
        action_frame =  sceobj_frame.permute(0, 1, 3, 2, 4, 5)
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

        #### co-attention feature
        ## relu
        #scene_feature = F.relu(scene_feature)
        #object_feature = F.relu(object_feature)
        #action_feature = F.relu(action_feature)

        ## bn
        #scene_feature = self.scene_coattention_bn1(scene_feature)
        #object_feature = self.object_coattention_bn1(object_feature)
        #action_feature = self.action_coattention_bn1(action_feature)

        ## scene and object
        bsz = N
        ds = self.scene_classes 
        do = self.object_classes
        V = scene_feature # bsz x T x ds
        S = object_feature # bsz x T x do
        V_comp = V.view(bsz * T, ds)  # compact image representation, (bsz * T) x ds
        S_comp = S.view(bsz * T, do)  # compact text representation, (bsz * T) x do

        x = F.linear(V_comp, self.Wb_sc_ob, None)  # (bsz * T) x do
        x = x.view(bsz, T, do).transpose(1, 2)  # bsz x do x T
        C = torch.tanh(torch.matmul(S, x))  # bsz x T x T

        img_embed = F.linear(V_comp, self.Wv_sc_ob, None).view(bsz, T, self.k).transpose(1, 2).contiguous()  # bsz x k x T
        text_embed = F.linear(S_comp, self.Ws_sc_ob, None).view(bsz, T, self.k).transpose(1, 2).contiguous()  # bsz x k x T

        H_v = torch.tanh(img_embed + torch.bmm(text_embed, C))  # bsz x k x T
        H_v = H_v.transpose(1, 2).contiguous().view(-1, self.k)  # (bsz * T) x k
        H_s = torch.tanh(text_embed + torch.bmm(img_embed, C.transpose(1, 2)))  # bsz x k x T
        H_s = H_s.transpose(1, 2).contiguous().view(-1, self.k)  # (bsz * T) x k

        a_v = F.linear(H_v, self.Whv_sc_ob, None)  # (bsz * T) x 1
        a_v = F.softmax(a_v.view(bsz, 1, T), dim=2)  # bsz x 1 x T
        a_s = F.linear(H_s, self.Whs_sc_ob, None)  # (bsz * T) x 1
        a_s = F.softmax(a_s.view(bsz, 1, T), dim=2)  # bsz x 1 x T

        scene_feature_scob = torch.bmm(a_v, V).squeeze()
        object_feature_scob = torch.bmm(a_s, S).squeeze()
        
        ## scene and action
        bsz = N
        ds = self.scene_classes
        da = self.action_classes
        V = scene_feature
        S = action_feature
        V_comp = V.view(bsz * T, ds)  # compact image representation, (bsz * T) x d
        S_comp = S.view(bsz * T, da)  # compact text representation, (bsz * T) x d

        x = F.linear(V_comp, self.Wb_sc_ac, None)  # (bsz * T) x d
        x = x.view(bsz, T, da).transpose(1, 2)  # bsz x d x T
        C = torch.tanh(torch.matmul(S, x))  # bsz x T x T

        img_embed = F.linear(V_comp, self.Wv_sc_ac, None).view(bsz, T, self.k).transpose(1, 2).contiguous()  # bsz x k x T
        text_embed = F.linear(S_comp, self.Ws_sc_ac, None).view(bsz, T, self.k).transpose(1, 2).contiguous()  # bsz x k x T

        H_v = torch.tanh(img_embed + torch.bmm(text_embed, C))  # bsz x k x T
        H_v = H_v.transpose(1, 2).contiguous().view(-1, self.k)  # (bsz * T) x k
        H_s = torch.tanh(text_embed + torch.bmm(img_embed, C.transpose(1, 2)))  # bsz x k x T
        H_s = H_s.transpose(1, 2).contiguous().view(-1, self.k)  # (bsz * T) x k

        a_v = F.linear(H_v, self.Whv_sc_ac, None)  # (bsz * T) x 1
        a_v = F.softmax(a_v.view(bsz, 1, T), dim=2)  # bsz x 1 x T
        a_s = F.linear(H_s, self.Whs_sc_ac, None)  # (bsz * T) x 1
        a_s = F.softmax(a_s.view(bsz, 1, T), dim=2)  # bsz x 1 x T

        scene_feature_scac = torch.bmm(a_v, V).squeeze()
        action_feature_scac = torch.bmm(a_s, S).squeeze()
    
        ## object and action
        bsz = N
        do = self.object_classes
        da = self.action_classes
        V = object_feature
        S = action_feature
        V_comp = V.view(bsz * T, do)  # compact image representation, (bsz * T) x d
        S_comp = S.view(bsz * T, da)  # compact text representation, (bsz * T) x d

        x = F.linear(V_comp, self.Wb_ob_ac, None)  # (bsz * T) x d
        x = x.view(bsz, T, da).transpose(1, 2)  # bsz x d x T
        C = torch.tanh(torch.matmul(S, x))  # bsz x T x T

        img_embed = F.linear(V_comp, self.Wv_ob_ac, None).view(bsz, T, self.k).transpose(1, 2).contiguous()  # bsz x k x T
        text_embed = F.linear(S_comp, self.Ws_ob_ac, None).view(bsz, T, self.k).transpose(1, 2).contiguous()  # bsz x k x T

        H_v = torch.tanh(img_embed + torch.bmm(text_embed, C))  # bsz x k x T
        H_v = H_v.transpose(1, 2).contiguous().view(-1, self.k)  # (bsz * T) x k
        H_s = torch.tanh(text_embed + torch.bmm(img_embed, C.transpose(1, 2)))  # bsz x k x T
        H_s = H_s.transpose(1, 2).contiguous().view(-1, self.k)  # (bsz * T) x k

        a_v = F.linear(H_v, self.Whv_ob_ac, None)  # (bsz * T) x 1
        a_v = F.softmax(a_v.view(bsz, 1, T), dim=2)  # bsz x 1 x T
        a_s = F.linear(H_s, self.Whs_ob_ac, None)  # (bsz * T) x 1
        a_s = F.softmax(a_s.view(bsz, 1, T), dim=2)  # bsz x 1 x T

        object_feature_obac = torch.bmm(a_v, V).squeeze()
        action_feature_obac = torch.bmm(a_s, S).squeeze()

        ## concat co-attention feature
        scene_feature_co = torch.cat((scene_feature_scac, scene_feature_scob), 1)
        object_feature_co = torch.cat((object_feature_scob, object_feature_obac), 1)
        action_feature_co = torch.cat((action_feature_scac, action_feature_obac), 1)
        concat_feature = torch.cat((torch.cat((scene_feature_co, object_feature_co), 1), action_feature_co), 1) 

        #### reduce dimentation
        concat_feature = self.relu(concat_feature)
        #concat_feature = self.concat_bn1(concat_feature)
        concat_feature = self.reduce_dropout(concat_feature)
        concat_feature = self.concat_reduce_dim(concat_feature)

        #### classification
        final_classification = self.relu(concat_feature)
        #final_classification = self.final_bn1(final_classification)
        final_classification = self.dropout(final_classification)
        final_classification = self.final_classifier(final_classification)

        return final_classification
