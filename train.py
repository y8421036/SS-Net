import datetime
import glob
import os
import time
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from networks import YNet_general
from data import TestImage, get_data
from losses import BinaryDiceLoss, FocalLoss
from utils import BaseOptions
from torch.utils.data import DataLoader
import SimpleITK as sitk


def adjust_lr(optimizer, base_lr, current_epoch, max_epoch, power=0.9):
    lr = base_lr * (1 - float(current_epoch) / max_epoch) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval_random_patch(val_loader, criterion, model, device):
    model.eval()
    loss = 0
    counter = 0

    for img, label, edge in val_loader:
        img = img.to(device)
        label = label.to(device)
        pred_vessel, pred_edge = model(img)

        loss += criterion(pred_vessel, label).item()
        counter += 1

    loss = loss / counter
    print("val dice: ", 1-loss)
    return loss


def train(args):
    device = args.device
    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    batch_size = args.batch_size

    criterion_dice = BinaryDiceLoss().cuda()
    criterion_bce = nn.functional.binary_cross_entropy
    criterion_focal = FocalLoss().cuda()

    min_eval_loss = float("inf")
    best_iter = 0

    model = YNet_general(in_channels=1, ffc=True, ratio_in=ratio).to(device)
    model.train()

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader = get_data(data_path, batch_size, crop_size)

    for t in range(iterations):
        for img, label, edge in tqdm.tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            edge = edge.to(device)
            pred_vessel, pred_edge = model(img)

            y1 = criterion_dice(pred_vessel, label)
            y2 = criterion_bce(pred_vessel, label.float())
            y3 = criterion_focal(pred_edge,edge)
            loss = y1 + y2 + args.edge_loss_lambda * y3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        n = t + 1
        lr = adjust_lr(optimizer, base_lr=args.learning_rate, current_epoch=n, max_epoch=iterations, power=0.9)
        if n % 2 == 0:
            print("Epoch:", n, "/", iterations, '\tLoss:', loss.item(), '\tLr:', lr)

        if n % 10 == 0 and n > iterations/2:
            print("Validation")
            val_loss = eval_random_patch(val_loader, criterion_dice, model, device)
            if val_loss < min_eval_loss:
                min_eval_loss = val_loss
                best_iter = n
                print(colored_text("Updating model, epoch: "), n)
                torch.save(model.state_dict(), ckpt_path+"ss-net_"+str(best_iter)+"_"+str(round(1-min_eval_loss,4))+ ".pt")
            model.train()
           
    print("Best val:\t iteration", best_iter, "\t dice", 1-min_eval_loss)
    return model


if __name__ == "__main__":
    args = BaseOptions().parse()
    set_seed(args.seed)

    crop_size = (96, 96, 96)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    ckpt_path = "./MIDAS/3down/"+date+"baselineloss-random/"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_cuda
    
    t0 = time.time()
    train(args)
    print("It takes ", (time.time() - t0) / 60 / 60, "hours.")
    
