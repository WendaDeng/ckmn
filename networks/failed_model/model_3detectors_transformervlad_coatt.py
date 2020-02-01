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
            
        #### transformer vlad
        self.alpha = 100.0
        self.scene_num_clusters = 196
        self.scene_soft_weight = nn.Conv1d(self.T, self.scene_num_clusters, kernel_size=(1), bias=True)
        self.scene_centroids = nn.Parameter(torch.rand(self.scene_num_clusters, self.T))
        
        self.object_num_clusters = 196
        self.object_soft_weight = nn.Conv1d(self.T, self.object_num_clusters, kernel_size=(1), bias=True)
        self.object_centroids = nn.Parameter(torch.rand(self.object_num_clusters, self.T))

        self.action_num_clusters = 196
        self.action_soft_weight = nn.Conv1d(self.T, self.action_num_clusters, kernel_size=(1), bias=True)
        self.action_centroids = nn.Parameter(torch.rand(self.action_num_clusters, self.T))

        #### co-attention model
        self.k = 128
        self.Wb_sc_ob = Parameter(torch.Tensor(self.object_num_clusters, self.scene_num_clusters))
        self.Wv_sc_ob = Parameter(torch.Tensor(self.k, self.scene_num_clusters))
        self.Ws_sc_ob = Parameter(torch.Tensor(self.k, self.object_num_clusters))
        self.Whv_sc_ob = Parameter(torch.Tensor(1, self.k))
        self.Whs_sc_ob = Parameter(torch.Tensor(1, self.k))

        self.Wb_sc_ac = Parameter(torch.Tensor(self.action_num_clusters, self.scene_num_clusters))
        self.Wv_sc_ac = Parameter(torch.Tensor(self.k, self.scene_num_clusters))
        self.Ws_sc_ac = Parameter(torch.Tensor(self.k, self.action_num_clusters))
        self.Whv_sc_ac = Parameter(torch.Tensor(1, self.k))
        self.Whs_sc_ac = Parameter(torch.Tensor(1, self.k))

        self.Wb_ob_ac = Parameter(torch.Tensor(self.action_num_clusters, self.object_num_clusters))
        self.Wv_ob_ac = Parameter(torch.Tensor(self.k, self.object_num_clusters))
        self.Ws_ob_ac = Parameter(torch.Tensor(self.k, self.action_num_clusters))
        self.Whv_ob_ac = Parameter(torch.Tensor(1, self.k))
        self.Whs_ob_ac = Parameter(torch.Tensor(1, self.k))

        #### feature reduce dimentation & fusion
        self.common_dimentation = 256
        self.reduce_temp_dimentation = 512
        self.reduce_dropout = nn.Dropout(0.5)
        self.scene_reduce_dim = nn.Linear(self.scene_num_clusters*2, self.common_dimentation)
        self.object_reduce_dim = nn.Linear(self.object_num_clusters*2, self.common_dimentation)
        self.action_reduce_dim = nn.Linear(self.action_num_clusters*2, self.common_dimentation)
        self.concat_reduce_dim = nn.Linear(self.common_dimentation*3, self.reduce_temp_dimentation)

        #### classification
        self.final_classifier = nn.Linear(self.reduce_temp_dimentation, opt.event_classes)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)

        #### init params
        self.scene_soft_weight.weight = nn.Parameter((2.0*self.alpha*self.scene_centroids).unsqueeze(-1))
        self.scene_soft_weight.bias = nn.Parameter(-self.alpha*self.scene_centroids.norm(dim=-1))

        self.object_soft_weight.weight = nn.Parameter((2.0*self.alpha*self.object_centroids).unsqueeze(-1))
        self.object_soft_weight.bias = nn.Parameter(-self.alpha*self.object_centroids.norm(dim=-1))

        self.action_soft_weight.weight = nn.Parameter((2.0*self.alpha*self.action_centroids).unsqueeze(-1))
        self.action_soft_weight.bias = nn.Parameter(-self.alpha*self.action_centroids.norm(dim=-1))

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

        #### transformer vlad
        ## N T F -> N F T
        scene_feature_t = scene_feature#.permute(0, 2, 1)
        object_feature_t = object_feature#.permute(0, 2, 1)
        action_feature_t = action_feature#.permute(0, 2, 1)
        
        ## scene vlad
        # transformer
        scene_atten = torch.bmm(scene_feature_t, scene_feature_t.transpose(1, 2))
        scene_atten = scene_atten / np.power(self.T, 0.5)
        scene_atten = F.softmax(scene_atten, dim=2)
        scene_atten = F.dropout(scene_atten, p=0.5)
        scene_feature_t = torch.bmm(scene_atten, scene_feature_t)
        # vlad
        scene_soft_weight = self.scene_soft_weight(scene_feature_t)
        scene_soft_weight = F.softmax(scene_soft_weight, dim=1)
        scene_residual = scene_feature_t.expand(self.scene_num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                self.scene_centroids.expand(scene_feature_t.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        scene_residual *= scene_soft_weight.unsqueeze(2)
        scene_vlad = scene_residual.sum(dim=-1)
        scene_vlad = F.normalize(scene_vlad, p=2, dim=1)
        
        ## obejct vlad
        # transformer
        object_atten = torch.bmm(object_feature_t, object_feature_t.transpose(1, 2))
        object_atten = object_atten / np.power(self.T, 0.5)
        object_atten = F.softmax(object_atten, dim=2)
        object_atten = F.dropout(object_atten, p=0.5)
        object_feature_t = torch.bmm(object_atten, object_feature_t)
        # vlad
        object_soft_weight = self.object_soft_weight(object_feature_t)
        object_soft_weight = F.softmax(object_soft_weight, dim=1)
        object_residual = object_feature_t.expand(self.object_num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                self.object_centroids.expand(object_feature_t.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        object_residual *= object_soft_weight.unsqueeze(2)
        object_vlad = object_residual.sum(dim=-1)
        object_vlad = F.normalize(object_vlad, p=2, dim=1)
        
        ## action vlad
        # transformer
        action_atten = torch.bmm(action_feature_t, action_feature_t.transpose(1, 2))
        action_atten = action_atten / np.power(self.T, 0.5)
        action_atten = F.softmax(action_atten, dim=2)
        action_atten = F.dropout(action_atten, p=0.5)
        action_feature_t = torch.bmm(action_atten, action_feature_t)
        # vlad
        action_soft_weight = self.action_soft_weight(action_feature_t)
        action_soft_weight = F.softmax(action_soft_weight, dim=1)
        action_residual = action_feature_t.expand(self.action_num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                self.action_centroids.expand(action_feature_t.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        action_residual *= action_soft_weight.unsqueeze(2)
        action_vlad = action_residual.sum(dim=-1)
        action_vlad = F.normalize(action_vlad, p=2, dim=1)

        #### co-attention feature
        scene_vlad = scene_vlad.permute(0, 2, 1).contiguous()
        object_vlad = object_vlad.permute(0, 2, 1).contiguous()
        action_vlad = action_vlad.permute(0, 2, 1).contiguous()
        
        ## scene and object
        # import ipdb; ipdb.set_trace()
        bsz = N
        ds = self.scene_num_clusters 
        do = self.object_num_clusters
        V = scene_vlad # bsz x T x ds
        S = object_vlad # bsz x T x do
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

        scene_vlad_scob = torch.bmm(a_v, V).squeeze()
        object_vlad_scob = torch.bmm(a_s, S).squeeze()
        
        ## scene and action
        bsz = N
        ds = self.scene_num_clusters
        da = self.action_num_clusters
        V = scene_vlad
        S = action_vlad
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

        scene_vlad_scac = torch.bmm(a_v, V).squeeze()
        action_vlad_scac = torch.bmm(a_s, S).squeeze()
    
        ## object and action
        bsz = N
        do = self.object_num_clusters
        da = self.action_num_clusters
        V = object_vlad
        S = action_vlad
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

        object_vlad_obac = torch.bmm(a_v, V).squeeze()
        action_vlad_obac = torch.bmm(a_s, S).squeeze()
        
        #print(scene_feature_scac.size(), scene_feature_scob.size(), object_feature_scob.size(), object_feature_obac.size(), action_feature_scac.size(), action_feature_obac.size())

        ## concat co-attention feature
        scene_feature_co = torch.cat((scene_vlad_scac, scene_vlad_scob), 1)
        object_feature_co = torch.cat((object_vlad_scob, object_vlad_obac), 1)
        action_feature_co = torch.cat((action_vlad_scac, action_vlad_obac), 1)
       
        #### feature reduce dimentation 
        scene_feature_co = self.relu(scene_feature_co)
        scene_feature_co = self.reduce_dropout(scene_feature_co)
        scene_feature_co = self.scene_reduce_dim(scene_feature_co)
        
        object_feature_co = self.relu(object_feature_co)
        object_feature_co = self.reduce_dropout(object_feature_co)
        object_feature_co = self.object_reduce_dim(object_feature_co)

        action_feature_co = self.relu(action_feature_co)
        action_feature_co = self.reduce_dropout(action_feature_co)
        action_feature_co = self.action_reduce_dim(action_feature_co)

        concat_feature = torch.cat((torch.cat((scene_feature_co, object_feature_co), 1), action_feature_co), 1) 

        #### reduce dimentation
        concat_feature = self.relu(concat_feature)
        concat_feature = self.reduce_dropout(concat_feature)
        concat_feature = self.concat_reduce_dim(concat_feature)

        #### classification
        final_classification = self.relu(concat_feature)
        final_classification = self.dropout(final_classification)
        final_classification = self.final_classifier(final_classification)

        return final_classification
