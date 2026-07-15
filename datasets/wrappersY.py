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
import random
import torchvision.transforms.functional as TF


#训练包装层（模型输入处理）
def index_mask_to_onehot(mask, num_classes):
    """
    将 index mask 转为 one-hot：
    - 输入：mask (H, W)，每个像素是 [0, num_classes-1] 的整数
    - 输出：one_hot (num_classes, H, W)，float32
    """
    # 确保是 Long 类型
    mask_long = mask.long()  # (H, W)
    # F.one_hot 输出 (H, W, C)
    one_hot = F.one_hot(mask_long, num_classes=num_classes)  # (H, W, C)
    # 变为 (C, H, W)
    one_hot = one_hot.permute(2, 0, 1).float()  #(C, H, W) = (num_classes, H, W)
    check_map = torch.sum(one_hot, dim=0)  # (H, W)
    assert torch.all(check_map == 1), "ERROR: Some pixels do not belong to exactly one class (check mask values or num_classes)"
    return one_hot

@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]) ## 将输入归一化到 [-1, 1]
        # 注意：mask 是 index label，不能用 ToTensor（会/255）
        self.mask_resize = transforms.Resize(
            (inp_size, inp_size),
            interpolation=InterpolationMode.NEAREST
        )
        self.mask_to_tensor = transforms.PILToTensor()  # 保持 0~255，不除以 255

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]  # img: PIL RGB, mask: PIL L

        # 图像预处理
        img = self.img_transform(img)

        # 掩码预处理：Resize + 转为 Tensor，保持类别索引
        # mask_resize 输入输出都是 PIL
        mask = self.mask_resize(mask)  # PIL L
        mask = self.mask_to_tensor(mask)[0]  # (1,H,W) -> 取第 0 通道，得到 (H,W)
        # 现在 mask 是 uint8 Tensor，值 0~12
        mask = mask.long()
        invalid = (mask < 0) | (mask >= self.dataset.n_classes)
        mask[invalid] = 0
        gt = index_mask_to_onehot(mask, self.dataset.n_classes)  # (C,H,W)

        return {
            'inp': img,
            'gt': gt
        }

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
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])
        self.mask_resize = transforms.Resize(
            (inp_size, inp_size),
            interpolation=InterpolationMode.NEAREST
        )
        self.mask_to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]  # PIL RGB, PIL L # img: PIL RGB, mask: PIL L（单通道，值 0~12）
        # TODO: 如果未来要做随机翻转/旋转，这里对 img 和 mask 同时做
        # ===============================
        #  随机增强（几何增强一致作用于 img & mask）
        # ===============================
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # ✅ 只采样一次
            r = random.random()
            if r < 0.25:
                img = img.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)
            elif r < 0.50:
                img = img.transpose(Image.ROTATE_180)
                mask = mask.transpose(Image.ROTATE_180)
            elif r < 0.75:
                img = img.transpose(Image.ROTATE_270)
                mask = mask.transpose(Image.ROTATE_270)
            # else: no rotation

            if random.random() < 0.5:
                img = TF.adjust_brightness(img, 0.8 + 0.4 * random.random())
            if random.random() < 0.5:
                img = TF.adjust_contrast(img, 0.8 + 0.4 * random.random())

        # 图像处理
        img = self.img_transform(img)

        # 掩码处理：保持 index，0~12
        mask = self.mask_resize(mask)  # PIL L # PIL 图，NEAREST 插值，大小变为 (inp_size, inp_size)
        mask = self.mask_to_tensor(mask)[0]  # (H,W)，uint8  # (1, H, W) → 取第 0 通道，得到 (H, W)，dtype=uint8，值仍然是 0~12
        mask = mask.long()  # (H, W), dtype=int64
        invalid = (mask < 0) | (mask >= self.dataset.n_classes)
        mask[invalid] = 0
        gt = index_mask_to_onehot(mask, self.dataset.n_classes)

        return {
            'inp': img,
            'gt': gt
        }