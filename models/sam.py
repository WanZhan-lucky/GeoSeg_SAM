import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU, DiceLoss
from typing import Any, Optional, Tuple


def onehot_to_mask(mask, palette): #one-hot → 彩色 mask
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x=x.permute(2,0,1)
    return x

def init_weights(layer): #初始化网络权重 Conv2d / Linear：权重 ~ N(0, 0.02)，bias=0  BatchNorm2d：权重 ~ N(1, 0.02)，bias=0
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module): #平衡二分类 BCE 针对二值分割设计的损失：Balanced BCE。
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target): #简单 IoU Loss #所以当前版本没用它，只留作备选。
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)
    return iou.mean()

class PositionEmbeddingRandom(nn.Module): #随机频率的位置编码  用随机高斯矩阵来生成 2D 位置编码
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)), #num_pos_feats 是编码维度的一半
        )
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor: #归一化到 [0,1] 的坐标（x,y）。
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1)) #类似随机 Fourier Features 的位置编码方式。
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None,num_classes = None, loss_weight = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes = num_classes
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False #逻辑上是想 只训练解码头 / prompt。

        self.loss_mode = loss
        if self.loss_mode == 'iou':
            print("loss: CrossEntropy + Dice")
            # class_weights = torch.tensor(
            #     [1.136, 0.646, 0.353, 0.931, 0.772,
            #      0.998, 0.629, 1.020, 1.120, 1.198,
            #      0.674, 3.147, 0.375],
            #     dtype=torch.float32
            # ).cuda()  # 如果你用 GPU
            self.criterionCE =  torch.nn.CrossEntropyLoss()
            self.criterionIOU = IOU()
            self.criterionDice = DiceLoss()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs = 1
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device) #为空
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )#用 no_mask_embed 填充，相当于“无提示”
        self.features, self.pyramid = self.image_encoder(self.input)

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            pyramid_embeddings = self.pyramid,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        ) #low_res_masks: (B,1,C,64,64) #iou_predictions: (B,1)
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) #把低分辨率 mask 上采样成 1024×1024。
        self.pred_mask = masks

    def infer(self, input):
        bs = 1
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        self.features, self.pyramid = self.image_encoder(input)  #第一个val 第二张图推理循环 显存+5G
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            pyramid_embeddings=self.pyramid,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) ##self.pred_mask shape = (B, C, 1024, 1024)
        return masks
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # masks = masks[0]
        masks = masks[:, 0]
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self): #计算损失并反向传播
        #你前面 wrapper 里生成的 one-hot/索引标签。  CrossEntropyLoss → gt_mask 应该是 (1, H, W) 的整型标签。
        gt_index = torch.argmax(self.gt_mask, dim=1)
        ce_loss = self.criterionCE(self.pred_mask, gt_index)
        # iou_loss = self.criterionIOU(self.pred_mask, gt_index)
        dice_loss = self.criterionDice(self.pred_mask, gt_index)
        # lovasz_loss = self.criterionLovasz(self.pred_mask, gt_index)
        lambda_ce = 1.0
        lambda_dice = 1.0
        self.loss_G = lambda_ce * ce_loss + lambda_dice * dice_loss
        self.loss_G.backward()
        #CrossEntropy：像素分类精准度 IoU loss：区域重叠质量

    def optimize_parameters(self): #一整步训练
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False): #控制网络是否参与训练
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
