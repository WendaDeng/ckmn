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

        #### non local
        self.scene_nonlocal = NONLocalBlock1D(self.T)
        self.object_nonlocal = NONLocalBlock1D(self.T)
        self.action_nonlocal = NONLocalBlock1D(self.T)
        
        #### co-attention model
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

        #### bilinear triliner pooling
        self.bilinear_num = 4
        self.bilinear_dim = 512 
        self.temp_bilinear_dim = self.bilinear_num * self.bilinear_dim
        self.bilinear_dropout = 0.1
        self.scene_bilinear_scob = nn.Linear(self.scene_classes, self.temp_bilinear_dim)
        self.object_bilinear_scob = nn.Linear(self.object_classes, self.temp_bilinear_dim)
        self.scene_bilinear_scac = nn.Linear(self.scene_classes, self.temp_bilinear_dim)
        self.action_bilinear_scac = nn.Linear(self.action_classes, self.temp_bilinear_dim)
        self.object_bilinear_obac = nn.Linear(self.object_classes, self.temp_bilinear_dim)
        self.action_bilinear_obac = nn.Linear(self.action_classes, self.temp_bilinear_dim)
        
        self.trilinear_num = 8
        self.trilinear_dim = 512 
        self.temp_trilinear_dim = self.trilinear_num * self.trilinear_dim
        self.trilinear_dropout = 0.1
        self.scob_trilinear = nn.Linear(self.bilinear_dim, self.temp_trilinear_dim)
        self.scac_trilinear = nn.Linear(self.bilinear_dim, self.temp_trilinear_dim)
        self.obac_trilinear = nn.Linear(self.bilinear_dim, self.temp_trilinear_dim)

        #### classification
        self.final_classifier = nn.Linear(self.trilinear_dim, opt.event_classes)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)

        self._reset_parameters()
        for l in self.children():
            if isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                #nn.init.normal_(l.weight, mean=0, std=0.01)
                #nn.init.uniform_(l.weight, a=0, b=1)
                #nn.init.xavier_normal_(l.weight, gain=1.0)
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
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

        #### self attention
        scene_feature = self.scene_nonlocal(scene_feature)
        object_feature = self.object_nonlocal(object_feature)
        action_feature = self.action_nonlocal(action_feature)

        #### co-attention feature
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

        #### bilinear trilinear pooling
        ## scene & object 
        scene_feature_scob = self.scene_bilinear_scob(scene_feature_scob)
        object_feature_scob = self.object_bilinear_scob(object_feature_scob)
        scob_bilinear = torch.mul(scene_feature_scob, object_feature_scob)
        scob_bilinear = F.dropout(scob_bilinear, self.bilinear_dropout)
        scob_bilinear = scob_bilinear.view(-1, 1, self.bilinear_dim, self.bilinear_num)
        scob_bilinear = torch.squeeze(torch.sum(scob_bilinear, 3))
        scob_bilinear = torch.sqrt(F.relu(scob_bilinear)) - torch.sqrt(F.relu(-scob_bilinear))
        #scob_bilinear = F.relu(scob_bilinear)
        scob_bilinear = F.normalize(scob_bilinear)

        ## scene & action 
        scene_feature_scac = self.scene_bilinear_scac(scene_feature_scac)
        action_feature_scac = self.action_bilinear_scac(action_feature_scac)
        scac_bilinear = torch.mul(scene_feature_scac, action_feature_scac)
        scac_bilinear = F.dropout(scac_bilinear, self.bilinear_dropout)
        scac_bilinear = scac_bilinear.view(-1, 1, self.bilinear_dim, self.bilinear_num)
        scac_bilinear = torch.squeeze(torch.sum(scac_bilinear, 3))
        scac_bilinear = torch.sqrt(F.relu(scac_bilinear)) - torch.sqrt(F.relu(-scac_bilinear))
        #scac_bilinear = F.relu(scac_bilinear)
        scac_bilinear = F.normalize(scac_bilinear)

        ## object & action
        object_feature_obac = self.object_bilinear_obac(object_feature_obac)
        action_feature_obac = self.action_bilinear_obac(action_feature_obac)
        obac_bilinear = torch.mul(object_feature_obac, action_feature_obac)
        obac_bilinear = F.dropout(obac_bilinear, self.bilinear_dropout)
        obac_bilinear = obac_bilinear.view(-1, 1, self.bilinear_dim, self.bilinear_num)
        obac_bilinear = torch.squeeze(torch.sum(obac_bilinear, 3))
        obac_bilinear = torch.sqrt(F.relu(obac_bilinear)) - torch.sqrt(F.relu(-obac_bilinear))
        #obac_bilinear = F.relu(obac_bilinear)
        obac_bilinear = F.normalize(obac_bilinear)

        ## scene & object & action
        scob_bilinear = self.scob_trilinear(scob_bilinear)
        scac_bilinear = self.scac_trilinear(scac_bilinear)
        obac_bilinear = self.obac_trilinear(obac_bilinear)
        scobac_trilinear = torch.mul((torch.mul(scob_bilinear, scac_bilinear)), obac_bilinear)
        scobac_trilinear = F.dropout(scobac_trilinear, self.trilinear_dropout)
        scobac_trilinear = scobac_trilinear.view(-1, 1, self.trilinear_dim, self.trilinear_num)
        scobac_trilinear = torch.squeeze(torch.sum(scobac_trilinear, 3))
        scobac_trilinear = torch.sqrt(F.relu(scobac_trilinear)) - torch.sqrt(F.relu(-scobac_trilinear))
        #scobac_trilinear = F.relu(scobac_trilinear)
        scobac_trilinear = F.normalize(scobac_trilinear)

        #### classification
        final_classification = self.relu(scobac_trilinear)
        final_classification = self.dropout(final_classification)
        final_classification = self.final_classifier(final_classification)

        return final_classification
