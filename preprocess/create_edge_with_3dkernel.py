"""
用3*3*3的kernel生成label volume前景的单体素edge volume。
条件：
1.kernel中心点为1
2.kernel中心点的6邻域中存在0

如果是对整个volume判断边缘，所以不需要考虑边界问题。
如果是子块，则是通过加镜像padding。
"""
import glob
import math
import os
import torch
from torch import nn
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F


def create_edge_volume_with_3dkernel(img_path, out_path):
    label = sitk.ReadImage(img_path)
    label_source = sitk.GetArrayFromImage(label).astype(np.float32)  # z,y,x !!!!!
    # edge = np.zeros_like(label)

    padding = 0  # 表示F.conv3d不补padding。因为在卷积前，手动加了1层镜像padding。
    kernel_size = 3
    out_channel = 1
    in_channel = 1
    group = 1

    # -------------------------------------------------------------------
    label_padding = torch.from_numpy(label_source[np.newaxis, np.newaxis, :])  # 变为b,c,z,y,x
    label_padding = nn.ReplicationPad3d(1)(label_padding)  # 使用复制的方式填充1层padding，因为F.conv3d无法设置padding的model
    kernel = torch.zeros(out_channel, int(in_channel / group), kernel_size, kernel_size, kernel_size)
    kernel[0][0][0][1][1] = 1  # b,c,z,y,x 设置6邻域
    kernel[0][0][1][0][1] = 1
    kernel[0][0][1][1][0] = 1
    kernel[0][0][1][1][2] = 1
    kernel[0][0][1][2][1] = 1
    kernel[0][0][2][1][1] = 1
    edge = F.conv3d(label_padding, kernel, padding=padding)  # 滤波相当于对kernel内每个元素求和
    edge = np.where((label_source == 0), 255,
                    edge)  # mask掉背景，为了后面只对前景判断边界。255是因为np.where用and时会报错:(edge>0 and edge<kernel_num-3)
    kernel_neighbourhood = 6  # 6邻域
    edge = np.where((edge < kernel_neighbourhood), 1, 0)
    # -------------------------------------------------------------------

    edge = edge.squeeze().squeeze()
    output = sitk.GetImageFromArray(edge.astype(np.uint8))
    output.SetSpacing(label.GetSpacing())
    sitk.WriteImage(output, os.path.join(out_path, 'edge-' + os.path.basename(img_path)))


if __name__ == '__main__':
    input_dir = '/data2/datasets/TubeTK/data_brain-fold1/training/label'
    output_dir = '/data2/datasets/TubeTK/data_brain-fold1/training/edge_label'

    for path in tqdm(glob.glob(os.path.join(input_dir, '*.nii'))):
        create_edge_volume_with_3dkernel(path, output_dir)
