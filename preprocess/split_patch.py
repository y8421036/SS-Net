"""
滑窗方式划分volume为patch，余数部分回缩。
"""

import glob
import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm
from data import normalization


def split(data_path, end, patch_shape, stride):  # patch-shape z=y=x
    out_dir = output_dir + os.path.basename(data_path)[:end]
    if not os.path.exists(out_dir):  # 待分割的volume，split后放到同一个文件夹
        os.makedirs(out_dir)

    data = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(data).astype(np.float32)
    data = normalization(data, 'float32')  # 先归一化，再crop！！！！！！！！！！！！！！！！！！

    for z in range(0, data.shape[0], stride[2]):
        for y in range(0, data.shape[1], stride[1]):
            for x in range(0, data.shape[2], stride[0]):
                z = np.where((z + patch_shape[2]) < data.shape[0], z, data.shape[0] - patch_shape[2])  # 让余数部分也提取出来
                y = np.where((y + patch_shape[1]) < data.shape[1], y, data.shape[1] - patch_shape[1])
                x = np.where((x + patch_shape[0]) < data.shape[2], x, data.shape[2] - patch_shape[0])
                data_patch = data[z:z + patch_shape[2], y:y + patch_shape[1], x:x + patch_shape[0]]
                data_patch = sitk.GetImageFromArray(np.ascontiguousarray(data_patch))
                sitk.WriteImage(data_patch, os.path.join(out_dir, 'image'+f'-{z:03}-{y:03}-{x:03}.nii'))  # 格式化字符串，并控制宽度


if __name__ == '__main__':
    crop_size = (64, 64, 64)  # z,y,x

    output_dir = '/data2/datasets/IXI/test/image/'
    input_dir = '/data2/datasets/IXI/test/image_full/'

    for path in tqdm(glob.glob(os.path.join(input_dir, '*.nii.gz'))):
        split(path, -7, crop_size, crop_size)
