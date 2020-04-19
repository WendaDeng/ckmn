import torch
import torch.nn as nn

from . import object_detector_network
from . import action_detector_network
from . import graph_layers


class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        self.concept_number = opt.object_classes + opt.action_classes
        self.latent_dimension = 1024
        self.num_class = opt.event_classes
        self.graph_fusion = opt.graph_fusion

        self.object_detector = object_detector_network.Object_Detector(opt)
        self.action_detector = action_detector_network.Action_Detector(opt)

        self.object_fc = nn.Linear(opt.object_classes, 512)
        self.action_fc = nn.Linear(opt.action_classes, 512)
        self.concat_reduce_dim = nn.Linear(self.latent_dimension, self.latent_dimension)
        self._add_graph_layers(opt.object_classes, opt.object_classes)
        self._add_classification_layer(512, self.latent_dimension)
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
                nn.init.normal_(l.weight, 0, 0.001)
                #nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)


    def _add_graph_layers(self, input_dim, output_dim):
        # Graph Convolution
        self.gc1 = graph_layers.GraphConvolution(input_dim, 32)  # nn.Linear(128, 32)
        self.gc2 = graph_layers.GraphConvolution(32, output_dim)
        self.gc3 = graph_layers.GraphConvolution(input_dim, 32)  # nn.Linear(128, 32)
        self.gc4 = graph_layers.GraphConvolution(32, output_dim)  # nn.Linear(128, 32)


    def _add_classification_layer(self, input_dim, hidden_dim):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            self.fc_verb = nn.Linear(hidden_dim, self.num_class[0])
            self.fc_noun = nn.Linear(hidden_dim, self.num_class[1])
        else:
            self.final_classifier = nn.Linear(hidden_dim, self.num_class)


    def forward(self, sceobj_frame):
        
        action_frame = sceobj_frame.permute(0, 1, 3, 2, 4, 5)
        # N: batch size;    T: segment number;  D: sample duration
        N, T, D, C, H, W = sceobj_frame.size()
        _, _, _, _, aH, aW = action_frame.size()

        ## scene and object frame input size N T D C H W
        # N T D C H W -> NTD C H W
        sceobj_frame = sceobj_frame.view(-1, C, H, W)
        # NTD C H W -> NTD F1
        object_feature = self.object_detector(sceobj_frame)
        del sceobj_frame
        # NTD F1 -> NTD F
        object_feature = self.object_fc(object_feature)
        object_feature = self.relu(object_feature)
        # NTD F -> NT D F
        object_feature = object_feature.view(N*T, D, -1)

        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.view(-1, C, D, aH, aW)
        # NT C D aH aW ->  NT F2
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F2 -> NT F
        action_feature = self.action_fc(action_feature)
        action_feature = self.relu(action_feature)

        # insert action feature into object feature
        for i in range(N * T):
            for j in range(D):
                if j == int(D / 2):
                    object_feature[i,j] = action_feature[i]

        obj_graph_feature = []
        for i in range(N * T):
            x = object_feature[i].cpu().data
            x, adj = graph_layers.graph_generator(x)
            x = torch.from_numpy(x).to(object_feature.device)
            adj = torch.from_numpy(adj).to(object_feature.device)
            # Graph Convolution
            x1 = self.relu(self.gc1(x, adj))
            x1 = self.gc2(x1, adj)

            # Learnable Graph branch
            x2 = x.view(-1, x.shape[-1])
            x2 = x2.matmul(x2.t())
            # x_norm = torch.norm(x2, p=2, dim=1).view(-1, 1)
            # x_norm = x_norm.matmul(x_norm.t())
            adj2 = torch.exp(x2 - x2.max(dim=1, keepdim=True)[0])  # 1 + x2 / x_norm
            d_inv_sqrt2 = torch.diag(torch.pow(torch.sum(adj2, dim=1), -0.5))
            adj_hat2 = d_inv_sqrt2.matmul(adj2).matmul(d_inv_sqrt2)
            adj_hat2 = adj_hat2.view(1, adj_hat2.shape[0], adj_hat2.shape[1])

            y2 = self.relu(self.gc3(x, adj_hat2))

            y22 = y2.view(-1, y2.shape[-1])
            y22 = y22.matmul(y22.t())
            y2_norm = torch.norm(y22, p=2, dim=1).view(-1, 1)
            y2_norm = y2_norm.matmul(y2_norm.t())
            adj3 = 1 + y22 / y2_norm
            d_inv_sqrt3 = torch.diag(torch.pow(torch.sum(adj3, dim=1), -0.5))
            adj_hat3 = d_inv_sqrt3.matmul(adj3).matmul(d_inv_sqrt3)
            adj_hat3 = adj_hat3.view(1, adj_hat3.shape[0], adj_hat3.shape[1])

            # y2 = self.gc4(y2, adj_hat2)
            y2 = self.gc4(y2, adj_hat3)

            obj_graph_feature.append((x1 + y2) / 2.0)

        obj_graph_feature = torch.stack(obj_graph_feature, 0)
        # NT D F -> N T D F
        obj_graph_feature = obj_graph_feature.view(N, T, D, -1)
        object_feature = object_feature.view(N, T, D, -1)
        # N T D F -> N T F
        obj_graph_feature = torch.mean(obj_graph_feature, dim=2)
        object_feature = torch.mean(object_feature, dim=2)

        if self.graph_fusion == 'mean':
            object_feature = (object_feature + obj_graph_feature) / 2
        elif self.graph_fusion == 'max':
            object_feature = torch.max(object_feature, obj_graph_feature)
        elif self.graph_fusion == 'concat':
            # N T F -> N T 2F -> N T F
            object_feature = torch.cat((object_feature, obj_graph_feature), 1)
            object_feature = self.concat_reduce_dim(object_feature)
            object_feature = self.relu(object_feature)

        classification = self.relu(object_feature)
        classification = self.dropout(classification)
        classification = self.fc1(classification)
        del object_feature, action_feature

        classification = self.relu(classification)
        classification = self.dropout(classification)

        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            base_out_verb = self.fc_verb(classification)
            base_out_verb = torch.mean(base_out_verb, dim=1)

            base_out_noun = self.fc_noun(classification)
            base_out_noun = torch.mean(base_out_noun, dim=1)

            output = (base_out_verb, base_out_noun)
        else:
            output = self.final_classifier(classification)
            output = torch.mean(output, dim=1)

        return output
