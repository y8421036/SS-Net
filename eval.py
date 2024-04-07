import argparse
import datetime
import glob
from os import path
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data import TestImage
from networks import SSNet
from utils import metrics3d


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--image_dir', default="/data2/datasets/TubeTK/data_brain-fold1/test/")
    parser.add_argument('--g_ratio', default=0.5, type=float)
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--pretrain_dir', default="./MIDAS/3down/best")
    parser.add_argument('--pretrain_name', default='bynet_720_0.8875.pt')

    return parser


def quant_eval(model, image_patch_loader, label, folder, device="cuda"):
    full_img_pred = torch.zeros_like(label).cuda()

    with torch.no_grad():
        for data in tqdm.tqdm(image_patch_loader):
            img_patch = data[0].to(device=device)
            if img_patch.sum() == 0:
                continue
            pred_vessel, _ = model(img_patch)
            z = int(data[1][0])
            y = int(data[2][0])
            x = int(data[3][0])

            full_img_pred[0, 0, z:z + crop_size[2], y:y + crop_size[1], x:x + crop_size[0]] = pred_vessel

    sens, spec, prec, acc, fpr, dic, hd95 = metrics3d(full_img_pred, label, full_img_pred.shape[0])
    print('full image       dsc:{0:.4f}, hd95:{1:.4f}'.format(dic, hd95))

    pred = full_img_pred[0, 0, :, :, :]
    pred_out_path = out_path + os.path.basename(folder) + '_pred_SS-Net.nii'
    pred = (pred > 0.5).int()
    pred = pred.data.cpu().numpy().astype(np.uint8)
    pred = sitk.GetImageFromArray(pred)
    pred.SetSpacing((0.5, 0.5, 0.8))  # x,y,z
    sitk.WriteImage(pred, pred_out_path)

    return sens, spec, prec, acc, dic, hd95


def eval_ynet(args):
    sensitivity, specificity, precision, accuracy, DSC, HD95 = [], [], [], [], [], []
    ssnet_model = SSNet(in_channels=1, ffc=True, ratio_in=args.g_ratio,).to(args.device)
    # ssnet_model = nn.DataParallel(ssnet_model, device_ids=[0]).cuda()  # 训练时如果用这个函数，则load参数时也要用，不然报错！！
    ynet_path = path.join(args.pretrain_dir, args.pretrain_name)
    trained_model = torch.load(ynet_path)
    ssnet_model.load_state_dict(trained_model)#, strict=False)
    ssnet_model.eval()

    for folder in glob.glob(os.path.join(args.image_dir+'image', '*')):
        test_image_patch = TestImage(folder)
        test_image_patch_loader = DataLoader(test_image_patch, batch_size=1)  # 由于后续一些设置，这里batch须为1
        label_path = os.path.join(args.image_dir + 'label', os.path.basename(folder) + '.nii')
        label = sitk.ReadImage(label_path)
        label = torch.from_numpy(sitk.GetArrayFromImage(label).astype(np.float32))
        label = label.unsqueeze(0).unsqueeze(0).cuda()  # b,c,z,y,x
        
        sens, spec, prec, acc, dic, ahd = quant_eval(ssnet_model, test_image_patch_loader, label, folder)
        sensitivity.append(sens)
        specificity.append(spec)
        precision.append(prec)
        accuracy.append(acc)
        DSC.append(dic)
        HD95.append(ahd)

    print("average：    sensitivity:{0:.4f}, specificity:{1:.4f}, precision:{2:.4f}, accuracy:{3:.4f}, DSC:{4:.4f}\
            HD95:{5:.4f}".format(np.mean(sensitivity), 
                                np.mean(specificity), 
                                np.mean(precision), 
                                np.mean(accuracy), 
                                np.mean(DSC), 
                                np.mean(HD95)))
    print('std:      sen:{0:.4f}, spe:{1:.4f}, pre:{2:.4f}, acc:{3:.4f}, dic:{4:.4f}, hd95:{5:.4f},'.format(
            np.std(sensitivity),
            np.std(specificity),
            np.std(precision),
            np.std(accuracy),
            np.std(DSC),
            np.std(HD95)))



if __name__ == "__main__":
    args = argument_parser().parse_args()
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    crop_size = (96, 96, 96)  # MIDAS
    out_path = date+"pred/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    eval_ynet(args)
