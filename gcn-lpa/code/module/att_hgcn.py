import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .layer import GATLayer
from torch_geometric.utils import to_scipy_sparse_matrix


class BasicGCN(nn.Module):
    def __init__(self, input_layer_shape, hidden_layer_shape, output_layer_shape, type_fusion, type_att_size):
        super(BasicGCN, self).__init__()
        self.layer1 = GATLayer(input_layer_shape, hidden_layer_shape)
        self.layer2 = GATLayer(hidden_layer_shape, output_layer_shape)

        self.embd2class = nn.ParameterDict()
        self.bias = nn.ParameterDict()

        self.embd2class = nn.Parameter(torch.FloatTensor(output_layer_shape, output_layer_shape))
        nn.init.xavier_uniform_(self.embd2class.data, gain=1.414)
        self.bias = nn.Parameter(torch.FloatTensor(1, output_layer_shape))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)

    def forward(self, ft, adj):
        ft, _ = self.layer1(ft, adj)
        ft = F.relu(ft)
        ft_mid = ft
        ft = F.dropout(ft, training=self.training)
        ft, attention = self.layer2(ft, adj)

        logits = []
        embd = []
        embd = ft_mid
        logits = torch.mm(ft, self.embd2class) + self.bias

        return logits, embd, attention


