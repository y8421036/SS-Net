import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F


 
class FocalLoss(nn.Module):
    def __init__(self, num_labels=2, activation_type='sigmoid', gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type
 
    def forward(self, input, target):
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = input
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()
    

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target batch size don't match"

        # if predict.is_cuda:
        #     dsc = torch.FloatTensor(1).zero_().cuda()
        # else:
        #     dsc = torch.FloatTensor(1).zero_()
        dsc = 0
        for i in range(predict.shape[0]):  # batch内每个volume单独算dice，然后取平均。如果batch内所有volume一块算，结果有出入。
            img = predict[i].view(1, -1).contiguous()
            gt = target[i].view(1, -1).contiguous()
            intersection = torch.sum(torch.mul(img, gt)) * 2 + self.smooth
            denominator = torch.sum(img) + torch.sum(gt) + self.smooth
            dsc += intersection / denominator
        dice_loss = 1 - (dsc / predict.shape[0])

        return dice_loss
