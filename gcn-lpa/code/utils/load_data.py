import numpy as np
import scipy.sparse as sp
import torch
import pickle
from torch_geometric.utils import from_scipy_sparse_matrix


def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# def load_graph_data(dataset, show_details=False):
#
#     load_path = "../data/" + dataset + "/" + dataset
#     ft = np.load(load_path + "_feat.npy", allow_pickle=True)
#     label = np.load(load_path + "_label.npy", allow_pickle=True)
#     adj = np.load(load_path + "_adj.npy", allow_pickle=True)
#
#     ft = torch.FloatTensor(ft)
#     label = torch.LongTensor(label)
#
#     adj_sp = sp.coo_matrix(adj)
#     adj = sp_coo_2_sp_tensor(adj_sp)
#
#     return label, ft, adj


def load_graph_data(dataset, show_details=False):
    load_path = "../data/" + dataset + "/" + dataset
    ft = np.load(load_path + "_feat.npy", allow_pickle=True)
    label = np.load(load_path + "_label.npy", allow_pickle=True)
    adj = np.load(load_path + "_adj.npy", allow_pickle=True)

    ft = torch.FloatTensor(ft)
    label = torch.LongTensor(label)

    adj = torch.from_numpy(adj).to(dtype=torch.float)
    # adj_sp = sp.coo_matrix(adj)
    # edge_index, edge_weight = from_scipy_sparse_matrix(adj_sp)

    # if edge_weight is not None:
    #     edge_weight = edge_weight.float()

    return label, ft, adj
