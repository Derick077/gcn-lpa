import argparse
import sys


def cora_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=250)
    parser.add_argument('--warm_epochs', type=int, default=10)
    parser.add_argument('--compress_ratio', type=int, default=0.01)

    args, _ = parser.parse_known_args()
    return args


def set_params(dataset):
    if dataset == "cora":
        args = cora_params()

    return args
