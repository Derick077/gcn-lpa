import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, concat=True):
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, adj)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)

        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )

# class HomoAggregateLayer(nn.Module):
#
#     def __init__(self, in_layer_shape, out_shape, type_fusion, type_att_size):
#         super(HomoAggregateLayer, self).__init__()
#
#         self.type_fusion = type_fusion
#
#         self.W = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#         self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
#         nn.init.xavier_uniform_(self.bias.data, gain=1.414)
#
#         if type_fusion == 'att':
#             self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
#             nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
#             self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
#             nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
#             self.w_att = nn.Parameter(torch.FloatTensor(2 * type_att_size, 1))
#             nn.init.xavier_uniform_(self.w_att.data, gain=1.414)
#
#     def forward(self, x, adj):
#         device = x.device
#         attention = 0
#         ft = torch.mm(x, self.W.T)
#
#         # 将 scipy 稀疏矩阵转换为 torch 稀疏张量
#         if isinstance(adj, torch.Tensor):
#             sp_adj = adj
#         else:
#             adj = adj.tocoo()
#             indices = torch.from_numpy(
#                 np.vstack((adj.row, adj.col)).astype(np.int64)).to(device)
#             values = torch.from_numpy(adj.data.astype(np.float32)).to(device)
#             sp_adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape)).to(device)
#
#         ft = torch.spmm(sp_adj, ft)
#
#         if self.type_fusion == 'mean':  # 均值
#             agg_ft = ft.mean(0, keepdim=True)
#             attention = []
#         elif self.type_fusion == 'att':  # 注意力机制
#             att_query = torch.mm(ft, self.w_query.to(device))  # (N, type_att_size)
#             att_keys = torch.mm(ft, self.w_keys.to(device))  # (N, type_att_size)
#             att_input = torch.cat([att_keys, att_query], dim=1)  # (N, 2 * type_att_size)
#             att_input = F.dropout(att_input, 0.5, training=self.training)
#             e = F.elu(torch.matmul(att_input, self.w_att.to(device)))  # (N, 1)
#             attention = F.softmax(e.view(-1), dim=0)
#             agg_ft = ft.mul(attention.unsqueeze(-1)).sum(0, keepdim=True)
#
#         output = agg_ft + self.bias.to(device)
#
#         return output, attention
