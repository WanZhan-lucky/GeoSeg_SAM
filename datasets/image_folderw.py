import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import rasterio
#image_folder.py —— 数据读取层（原始图像加载）
@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False,ignore_bg = False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        """
               - 对于图像：3 通道 uint16 tif，但值在 0~255 之间，转为 RGB
               - 对于 mask：单通道 uint8 tif，值是 0~12，转为灰度 L
               """
        # img = Image.open(file)
        #
        # if self.mask:
        #     # 掩码保持为单通道灰度，内部是类别索引 0~12
        #     if img.mode != 'L':
        #         img = img.convert('L')
        #     return img
        # else:
        #     # 图像转 RGB（即使是 uint16，只要值在 0~255 内，转换不会丢信息）
        #     if img.mode != 'RGB':
        #         img = img.convert('RGB')
        #     return img

        with rasterio.open(file) as src:
            if self.mask:
                # mask：单通道，类别 0~12
                mask = src.read(1).astype(np.uint8)  # [H, W]
                img = Image.fromarray(mask, mode='L')
                return img
            else:
                # image：3 通道，uint16 但值在 0~255
                img = src.read().astype(np.uint8)  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))  # [H, W, C]
                img = Image.fromarray(img, mode='RGB')
                return img
        # img: PIL.Image，RGB
        # mask: PIL.Image，L（单通道）
        # rasterio读 → numpy → PIL

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, classes, palette,**kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
        self.n_classes = len(classes)
        self.classes = classes
        self.palette = palette

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
