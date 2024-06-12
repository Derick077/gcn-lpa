import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import load_data, set_params, load_graph_data
from utils.evaluate import evaluate
# from utils.cluster import kmeans
from module.att_lpa import *
from module.att_hgcn import BasicGCN
import warnings
import pickle as pkl
import os
import random
import time
from utils.kmeans_gpu import *
from utils.cluster import *

warnings.filterwarnings('ignore')

dataset = "cora"
args = set_params(dataset)
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")


# random seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    epochs = args.epochs
    label, ft, adj = load_graph_data(args.dataset)

    # num_cluster = int(ft.shape[0] * args.compress_ratio)  # 压缩初始伪标签的范围
    num_cluster = 7
    num_class = np.unique(label).shape[0]   # 真实标签的种类数
    init_pseudo_label = 0

    print('number of classes: ', num_cluster, '\n')
    layer_shape = []
    input_layer_shape = [ft.shape[1]]
    hidden_layer_shape = [256]
    output_layer_shape = [num_cluster]

    layer_shape.extend(input_layer_shape)
    layer_shape.extend(hidden_layer_shape)
    layer_shape.extend(output_layer_shape)

    model = BasicGCN(layer_shape[0], layer_shape[1], layer_shape[2], args.type_fusion, args.type_att_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    # if args.cuda and torch.cuda.is_available():
    #     model.cuda()
    #     ft = ft.cuda()
    #     adj = adj.cuda()
    #     # edge_index = edge_index.cuda()
    #     # if edge_weight is not None:
    #     #     edge_weight = edge_weight.cuda()
    #     label = label.cuda()

    best = 1e9
    loss_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embd, _ = model(ft, adj)

        if epoch == 0:
            init_pseudo_label = init_lpa(adj, ft, num_cluster)

            pseudo_label = init_pseudo_label
        elif epoch < args.warm_epochs:
            pseudo_label = init_pseudo_label
        else:
            pseudo_label = lpaf(adj, init_pseudo_label, num_cluster)
            init_pseudo_label = pseudo_label

        label_predict = torch.argmax(pseudo_label, dim=1)
        eva(label.detach().cpu().numpy(), label_predict.detach().cpu().numpy(), True)
        loss_train = F.cross_entropy(logits, label_predict.long().detach())
        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train.item())

        if loss_train < best:
            best = loss_train

        print(
            'epoch: {:3d}'.format(epoch),
            'train loss: {:.4f}'.format(loss_train.item()),
        )

    # evaluate
    logits, embd, _ = model(ft, adj)

    # kmeans(embd.detach().cpu(), label.detach().cpu(), 7)
    # acc, nmi, ari, f1, predict_labels = clustering(torch.FloatTensor(embd), torch.LongTensor(label), num_cluster)
    # print(acc)


if __name__ == '__main__':
    train()
