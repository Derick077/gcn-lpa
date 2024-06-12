import torch.nn.functional as F
import copy
import torch
from torch_geometric.utils import to_scipy_sparse_matrix


def gen_rand_label(ft_dict, num_cluster):
    rand_label = torch.randint(num_cluster, (ft_dict.shape[0],))
    # torch.randint()函数用于生成指定范围内的整数随机数。
    # 它接受三个参数：low（最小值）、high（最大值）和形状（shape）。返回的张量中的元素值将在**[low, high)**范围内。
    # shape[0]是可以得到行数
    rand_label = F.one_hot(rand_label, num_cluster).type(dtype=torch.float32)

    return rand_label


def lpa(init_label, adj, num_cluster, max_iter=1000):  # max_iter: 最大迭代次数，默认值为1000。
    pseudo_label = copy.deepcopy(init_label)  # 使用 copy.deepcopy 创建初始标签字典的深拷贝 pseudo_label_dict。
    pseudo_label = pseudo_label.cuda()   # 转移到GPU上

    pseudo_label = pseudo_label.cuda()
    adj = adj.cuda()

    update_label_list = []  # 用于存储每次迭代后的标签状态。
    soft_label = 0
    for i in range(max_iter):
        soft_label = torch.spmm(adj, pseudo_label)

        new_label = torch.argmax(soft_label, dim=1)
        new_label = F.one_hot(new_label, num_cluster).type(dtype=torch.float32)
        pseudo_label = new_label

        update_label_list.append(new_label)

        if len(update_label_list) > 1:
            if update_label_list[-2].equal(update_label_list[-1]):  # 检查当前标签与上一次迭代的标签是否相同，如果相同则停止迭代。
                break

    return pseudo_label


def init_lpa(adj, ft, num_cluster):
    run_num = 1
    for i in range(run_num):
        init_label = gen_rand_label(ft, num_cluster)
        pseudo_label = lpa(init_label, adj, num_cluster)

    return pseudo_label


def lpaf(adj, init_pseudo_label, num_cluster, max_iter=1000):
    pseudo_label = copy.deepcopy(init_pseudo_label).cuda()
    current_label = copy.deepcopy(init_pseudo_label).cuda()

    pseudo_label = pseudo_label.cuda()
    adj = adj.cuda()

    target_label_list = []
    soft_label = 0

    for i in range(max_iter):
        soft_label = torch.spmm(adj, current_label)

        new_label = torch.argmax(soft_label, dim=1)
        new_label_one_hot = F.one_hot(new_label, num_cluster).type(dtype=torch.float32)
        pseudo_label = new_label_one_hot

        target_label_list.append(new_label)

        current_label = pseudo_label

        if len(target_label_list) > 1 and target_label_list[-2].equal(target_label_list[-1]):
            break

    return pseudo_label

