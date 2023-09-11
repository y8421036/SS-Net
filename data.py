import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as tf
import random
import os
from typing import Callable
from torchvision import transforms
import SimpleITK as sitk
import glob


def normalization(dataset, dtype):  # min-max归一化
    _min = np.min(dataset)
    _range = np.max(dataset) - _min
    return ((dataset - _min) / _range).astype(dtype)


def load_single_path(path, file_type='*.nii'):
    """
    :return: 返回path下的所有.nii文件的绝对路径
    """
    files = []
    for file in glob.glob(os.path.join(path, file_type)):
        files.append(file)
    return files


def random_crop(image, label, edge, crop_factor=(0, 0, 0)):
    w, h, d = image.shape
    z = random.randint(0, w - crop_factor[0])
    y = random.randint(0, h - crop_factor[1])
    x = random.randint(0, d - crop_factor[2])

    image = image[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
    label = label[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
    edge = edge[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
    return image, label, edge


def load_dataset(root_dir, train=True):
    """
    读取root_dir下image、label和edge_label内各自的文件
    """
    images = []
    ground_truth = []
    edge_gt = []

    images_path = os.path.join(root_dir, 'image')
    ground_truth_path = os.path.join(root_dir, 'label')
    edge_gt_path = os.path.join(root_dir, 'edge_label')

    for file in glob.glob(os.path.join(images_path, '*.nii.gz')):
        ground_truth_name = os.path.basename(file)[:-13] + '.nii.gz'  # MIDAS
        # ground_truth_name = os.path.basename(file)[:-7] + '_GT.nii.gz'  # IXI
        edge_gt_name = 'edge-' + os.path.basename(ground_truth_name)

        images.append(file)
        ground_truth.append(os.path.join(ground_truth_path, ground_truth_name))
        edge_gt.append(os.path.join(edge_gt_path, edge_gt_name))

    return images, ground_truth, edge_gt


def get_data(data_path, batch_size, crop_size, val_batch_size=1):
    train_dataset_path = os.path.join(data_path, "training")
    val_dataset_path = os.path.join(data_path, "val_full")

    train_dataset = Data(train_dataset_path, train=True, scale=crop_size)
    val_dataset = Data(val_dataset_path, train=True, scale=crop_size)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return trainloader, valloader


class Data(Dataset):
    def __init__(self,
                 data_dir,
                 train=True,
                 flip=True,
                 scale=(0, 0, 0)):
        self.data_dir = data_dir
        self.train = train
        self.flip = flip
        self.transform = transforms.ToTensor()
        self.resize = scale
        self.images, self.ground_truth, self.edge_gt = load_dataset(self.data_dir, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.ground_truth[idx]
        edge_gt_path = self.edge_gt[idx]

        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image).astype(np.float32)  # [x,y,z] -> [z,y,x]
        label = sitk.ReadImage(gt_path)
        label = sitk.GetArrayFromImage(label).astype(np.int32)
        if 'val' in edge_gt_path:
            edge = np.zeros(image.shape, np.int32)
        else:
            edge = sitk.ReadImage(edge_gt_path)
            edge = sitk.GetArrayFromImage(edge).astype(np.int32)

        image = normalization(image, 'float32')

        image2, label2, edge2 = random_crop(image, label, edge, crop_factor=self.resize)
        while image2.sum() == 0:
            image2, label2, edge2 = random_crop(image, label, edge, crop_factor=self.resize)

        image2 = torch.from_numpy(np.ascontiguousarray(image2)).unsqueeze(0)
        label2 = torch.from_numpy(np.ascontiguousarray(label2)).unsqueeze(0)
        edge2 = torch.from_numpy(np.ascontiguousarray(edge2)).unsqueeze(0)

        return image2, label2, edge2  # c,z,y,x


class TestImage(Dataset):
    def __init__(self, brain_image_dir):
        self.images = load_single_path(brain_image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        z = os.path.basename(img_path)[6:9]
        y = os.path.basename(img_path)[10:13]
        x = os.path.basename(img_path)[14:17]

        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image).astype(np.float32)  # [z,y,x]

        # 测试集的patch提前划分，并做了min-max归一化。这里不需要第二遍！# image = normalization(image, 'float32') 

        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)  # c,z,y,x

        return image, z, y, x