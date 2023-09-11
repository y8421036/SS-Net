import torch
import numpy as np
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
from networks import YNet_general
import surface_distance as surfdist


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_iterations', default=800, type=int)
        parser.add_argument('--learning_rate', default=5e-4, type=float)
        parser.add_argument('--edge_loss_lambda', default=0.5, type=float) #0.5
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--g_ratio', default=0.5, type=float)

        parser.add_argument('--image_dir', default="/data2/datasets/TubeTK/data_brain-fold1/")
        parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
        parser.add_argument('--gpu', type=int, nargs='+', default=[0], dest='gpu')
        parser.add_argument('--vis_cuda', type=str, dest='vis_cuda', default='0')
        parser.add_argument('--seed', default=7, type=int)

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt


def print_params(n_classes):
    """
    打印网络结构和各层的参数
    """
    input_shape = (1, 1, 96, 96, 96)

    ynet_model = YNet_general("y_net_gen", ratio=0.5).cuda()

    print("YNet")
    summary(ynet_model, input_shape)


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN

def metrics_3d(pred, gt):
    eps = 1e-10  # 极小的值，为了在分母非0情况下可以基本不影响计算结果！
    FP, FN, TP, TN = numeric_score(pred, gt)
    tpr = TP / (TP + FN + eps)  # recall查全率、sensitivity敏感性 ！！！！
    tnr = TN / (FP + TN + eps)  # specificity特异性
    precision = TP / (TP + FP + eps)  # 准确率、查准率
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)  # 精准率ACC
    # iou = TP / (TP + FN + FP + eps)  # jaccard
    dice_coe = 2 * TP / (FP + 2 * TP + FN + eps)  # ！！！
    # fnr = FN / (FN + TP + eps)
    fpr = FP / (FP + TN + eps)
    # F1 = (2*precision*tpr)/(precision+tpr)  # F1-score

    return tpr, tnr, precision, accuracy, fpr, dice_coe


def metrics3d(pred, label, batch_size):
    pred = (pred > 0.5).int()

    tp, tn, precision, accuracy, fp, dice, hd95 = 0, 0, 0, 0, 0, 0, 0
    for i in range(batch_size):
        img = pred[i].data.cpu().numpy()
        gt = label[i].data.cpu().numpy()
        tpr, tnr, prec, acc, fpr, dic = metrics_3d(img, gt)

        gt = gt[0, :, :, :].astype(np.bool8)
        img = img[0, :, :, :].astype(np.bool8)
        surface_distances = surfdist.compute_surface_distances(gt,
                                                               img,
                                                               spacing_mm=(0.8, 0.5, 0.5))  # z,y,x!!!!!!!!!
        hd = surfdist.compute_robust_hausdorff(surface_distances, 95)

        tp += tpr
        tn += tnr
        precision += prec
        accuracy += acc
        fp += fpr
        dice += dic
        hd95 += hd

    return tp / batch_size, tn / batch_size, precision / batch_size, accuracy / batch_size, fp / batch_size, dice / batch_size, hd95 / batch_size


