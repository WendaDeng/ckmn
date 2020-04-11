import numpy as np
from random import choice
from math import sqrt
from torch import FloatTensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # To support batch operations
        support = input.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def compress_feature(input_feat, output_size, global_size):
    # sample vertex
    if input_feat.shape[0] > output_size:
        step = input_feat.shape[0] // global_size
        start_point = np.random.randint(step)
        sample_index = np.linspace(start_point, input_feat.shape[0], global_size + 1, endpoint=False,
                                   dtype=int).tolist()
        local_size = output_size - global_size
        local_center = choice(sample_index)
        for i in range(local_center - local_size // 2, local_center + local_size // 2 + 1):
            if i < 0 or i >= input_feat.shape[0] or i in sample_index:
                continue
            sample_index.append(i)
    else:
        sample_index = np.arange(input_feat.shape[0], dtype=int).tolist()
    output_dimension = len(sample_index)

    # establish the adjacent matrix A^tilde
    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            if i == j:
                adj[i][j] = 1.0
            else:
                adj[i][j] = 1.0 / abs(sample_index[i] - sample_index[j])
    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]

    return output_feat.astype(np.float32), adj_hat.astype(np.float32)


def graph_generator(raw_feat, output_size=32000, global_size=16000):  # raw_feat.shape: (l,4096)
    # L2-normalization
    feat = raw_feat / np.linalg.norm(raw_feat, ord=2, axis=-1).reshape(-1, 1)
    # Compress into 32 segments
    return compress_feature(feat, output_size, global_size)