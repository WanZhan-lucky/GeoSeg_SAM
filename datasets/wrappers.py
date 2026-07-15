
import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F

#训练包装层（模型输入处理）

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x=x.permute(2,0,1)
    return x

def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))
    # return transforms.ToTensor()(
    #     #transforms.Resize(size)(
    #     transforms.ToPILImage()(mask))

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))

def mask_to_onehot(mask, palette): #(3,1024,1024) → (5,1024,1024) #每个颜色形成一个类别通道
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    mask = mask.permute(1,2,0)*255 #mask 原来是 (C, H, W)，permute → (H, W, C)，更符合“图像像素”的直觉*255：因为 ToTensor 把像素值从 [0,255] 除以 255 变成 [0,1]，这里乘回去
    # mask.int()
    mask = mask.round().to(torch.uint8)
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = torch.all(equality, dim=-1) #[255,0,0] == [255,0,0] → [True,True,True] #[254,0,0] == [255,0,0] → [False,True,True]
        semantic_map.append(class_map)
    # (H, W, 3) - -->  (num_classes, H, W)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32) #.astype(np.float32) 兼容PyTorch 常见 loss（如 BCE、Dice、Focal 等）
    semantic_map = torch.as_tensor(semantic_map)
    semantic_map = semantic_map.permute(2,0,1)#(num_classes, H, W)
    map=torch.sum(semantic_map,dim=0)
    # assert torch.all(map == 1), "Mask error: some pixels not one-hot encoded correctly"
    # 这里加统计，而不是直接 assert
    invalid_zero = (map == 0).sum().item()
    invalid_multi = (map > 1).sum().item()
    # if invalid_zero > 0 or invalid_multi > 0:
    #     print("⚠ mask problem:",
    #           "no-class pixels:", invalid_zero,
    #           "multi-class pixels:", invalid_multi)
    # if invalid_zero > 0 or invalid_multi > 0:
        # raise AssertionError("mask illegal")
    return semantic_map


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([ ## 图像（用默认双线性）
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]) ## 将输入归一化到 [-1, 1]
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST), ## mask（必须明确最近邻）
                transforms.ToTensor(),#值 0~255 -> (C, H, W) , float32 , 值 0~1
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask,self.dataset.palette)
        return {
            'inp': self.img_transform(img),
            'gt': mask
        }


# PairedImageFolders
#     ↓
# PIL Image + PIL Label
#     ↓
# Resize → ToTensor → Normalize
#     ↓
# mask_to_onehot
# {
#   'inp': Tensor (3,H,W),
#   'gt': Tensor (C,H,W)
# }


# (Image, Mask)
# → Resize
# → mask_to_onehot
# → Normalize
# 磁盘文件
#    ↓
# ImageFolder
#    ↓ PIL.Image
# PairedImageFolders
#    ↓ (image, label)
# TrainDataset / ValDataset
#    ↓
# Tensor + One-hot
#    ↓
# 模型输入
@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        #random filp
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask,self.dataset.palette)
        return {
            'inp': self.img_transform(img),
            'gt': mask
        }

