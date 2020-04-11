import torch
import torch.nn as nn
from . import action_detector_network
from . import graph_layers


class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        self.latent_dimension = 1024
        self.num_class = opt.event_classes
        self.action_detector = action_detector_network.Action_Detector(opt)
        self._add_graph_layers(opt.action_classes, opt.action_classes)
        self._add_classification_layers(opt.action_classes, self.latent_dimension)
        self.dropout = nn.Dropout(opt.drop_rate)
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


    def _add_graph_layers(self, input_dim, output_dim):
        # Graph Convolution
        self.gc1 = graph_layers.GraphConvolution(input_dim, 32)  # nn.Linear(128, 32)
        self.gc2 = graph_layers.GraphConvolution(32, output_dim)
        self.gc3 = graph_layers.GraphConvolution(input_dim, 32)  # nn.Linear(128, 32)
        self.gc4 = graph_layers.GraphConvolution(32, output_dim)  # nn.Linear(128, 32)


    def _add_classification_layers(self, input_dim, hidden_dim):
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            self.fc_verb = nn.Linear(hidden_dim, self.num_class[0])
            self.fc_noun = nn.Linear(hidden_dim, self.num_class[1])
        else:
            self.final_classifier = nn.Linear(2 * hidden_dim, self.num_class)


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

        graph_feature = []
        for i in range(N):
            x = action_feature[i]
            x, adj = graph_layers.graph_generator(x)
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
            adj_hat2 = adj_hat2.view(x.shape[0], adj_hat2.shape[0], adj_hat2.shape[1])

            y2 = self.relu(self.gc3(x, adj_hat2))

            # y22 = y2.view(-1, y2.shape[-1])
            # y22 = y22.matmul(y22.t())
            # y2_norm = torch.norm(y22, p=2, dim=1).view(-1, 1)
            # y2_norm = y2_norm.matmul(y2_norm.t())
            # adj3 = 1 + y22 / y2_norm
            # d_inv_sqrt3 = torch.diag(torch.pow(torch.sum(adj3, dim=1), -0.5))
            # adj_hat3 = d_inv_sqrt3.matmul(adj3).matmul(d_inv_sqrt3)
            # adj_hat3 = adj_hat3.view(x.shape[0], adj_hat3.shape[0], adj_hat3.shape[1])

            y2 = self.gc4(y2, adj_hat2)
            # y2 = self.gc4(y2, adj3_hat)

            graph_feature.append((x1 + y2) / 2.0)

        graph_feature = torch.stack(graph_feature, 0)
        ## max pooling N T F -> N F
        action_feature, _ = torch.max(action_feature, dim=1)
        classification = torch.cat((action_feature, graph_feature), 1)

        ## classification
        classification = self.relu(self.fc1(classification))
        classification = self.dropout(classification)

        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.relu(self.fc_verb(classification))
            base_out_verb = self.dropout(base_out_verb)

            # Noun
            base_out_noun = self.relu(self.fc_noun(classification))
            base_out_noun = self.dropout(base_out_noun)

            output = (base_out_verb, base_out_noun)
        else:
            output = self.relu(self.final_classifier(classification))
            output = self.dropout(output)

        return output
