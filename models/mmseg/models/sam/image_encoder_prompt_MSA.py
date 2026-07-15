# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Type
from timm.models.layers import DropPath
from .common import LayerNorm2d, MLPBlock, Adapter
import math
import warnings
from itertools import repeat
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


class MiniASPP(nn.Module):
    """
    轻量版 ASPP，适合放在 encoder 最后一层之后：
    - 输入: [B, H, W, C]
    - 输出: [B, H, W, C]（残差）
    """
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = dim
        self.hidden_dim = hidden_dim

        # 先降到较小通道（比如 256），减轻计算
        self.proj_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        # 3 个不同膨胀率的 depthwise conv
        dilations = [1, 2, 4]
        self.branches = nn.ModuleList([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3,
                padding=d, dilation=d, groups=hidden_dim, bias=False
            )
            for d in dilations
        ])

        # 聚合后投回 dim 维度
        self.proj_out = nn.Conv2d(hidden_dim * len(dilations), dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B,H,W,C]
        B, H, W, C = x.shape
        feat = x.permute(0, 3, 1, 2).contiguous()   # [B,C,H,W]

        feat = self.proj_in(feat)                   # [B,hidden,H,W]

        multi_feats = []
        for conv in self.branches:
            multi_feats.append(conv(feat))          # 每个都是 [B,hidden,H,W]

        feat = torch.cat(multi_feats, dim=1)        # [B,hidden*3,H,W]
        feat = self.proj_out(feat)                  # [B,dim,H,W]
        feat = self.act(self.norm(feat))            # [B,dim,H,W]

        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        return x + feat   # 残差连接

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        # adaptor
        self.scale_factor = 32
        self.prompt_type = 'highpass'
        self.tuning_stage = 1234
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        # self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dim,
        #                                         self.tuning_stage, self.depth,
        #                                         self.input_type, self.freq_nums,
        #                                         self.handcrafted_tune, self.embedding_tune, self.adaptor,
        #                                         img_size, patch_size)
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

        # self.out_indices = [1, 4, 7] #而是 Transformer 的指定 block 输出。
        self.out_indices = [2,5,8,11]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.patch_embed(x)

        # ################# adaptor
        # embedding_feature = self.prompt_generator.init_embeddings(x)  # [1,4096,24]
        # handcrafted_feature = self.prompt_generator.init_handcrafted(inp)  # [1,24,64,64]
        # prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)
        # ################ adaptor end

        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        if self.pos_embed is not None:
            if self.pos_embed.size(2) != x.size(2):
                pos_embed = self.pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic')
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed

        outs = []

        for i, blk in enumerate(self.blocks):
            x = blk(x, i)

            if i in self.out_indices:
                outs.append(self.neck(x.permute(0, 3, 1, 2)))

        x = self.neck(x.permute(0, 3, 1, 2))
        return x, outs
        # return x, None

def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dim = embed_dim
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depth = depth
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        for i in range(self.depth):

            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
                nn.GELU(),
                #nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
            )



            ###### add by pxy 230706 #########
            # lightweight_mlp = MLP(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor,self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor,3)

            ###########


            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.prompt_generator = PatchEmbed2(img_size=img_size,
                                                   patch_size=patch_size, in_chans=3,
                                                   embed_dim=self.embed_dim//self.scale_factor)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape  #[1,64,64,768]
        x = x.reshape(N, C, H*W).permute(0, 2, 1) #[1,4096,768]
        return self.embedding_generator(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums) #[1,3,1024,1024]
        return self.prompt_generator(x)

    def get_prompt(self, handcrafted_feature, embedding_feature):
        N, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(N, C, H*W).permute(0, 2, 1) #[1,4096,24]
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            # prompt = proj_prompt(prompt)
            prompt = lightweight_mlp(handcrafted_feature + embedding_feature) #[1,4096,24]
            prompts.append(self.shared_mlp(prompt))
            #prompts.append(prompt)
        return prompts #12*[1,4096,768]

    def forward(self, x):
        if self.input_type == 'laplacian':
            pyr_A = self.lap_pyramid.pyramid_decom(img=x, num=self.freq_nums)
            x = pyr_A[:-1]
            laplacian = x[0]
            for x_i in x[1:]:
                x_i = F.interpolate(x_i, size=(laplacian.size(2), laplacian.size(3)), mode='bilinear', align_corners=True)
                laplacian = torch.cat([laplacian, x_i], dim=1)
            x = laplacian
        elif self.input_type == 'fft':
            x = self.fft(x, self.freq_nums)
        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        # get prompting
        prompt = self.prompt_generator(x)

        if self.mode == 'input':
            prompt = self.proj(prompt)
            return prompt
        elif self.mode == 'stack':
            prompts = []
            for i in range(self.depth):
                proj = getattr(self, 'proj_{}'.format(str(i)))
                prompts.append(proj(prompt))
            return prompts
        elif self.mode == 'hierarchical':
            prompts = []
            for i in range(self.depth):
                proj_prompt = getattr(self, 'proj_prompt_{}'.format(str(i)))
                prompt = proj_prompt(prompt)
                prompts.append(self.proj_token(prompt))
            return prompts

    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        #rate=1/16
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        #fft = fft * (1 - mask)
        fft_low = fft * mask
        fr_low = fft_low.real
        fi_low = fft_low.imag

        fft_hires_low = torch.fft.ifftshift(torch.complex(fr_low, fi_low))
        inv_low = torch.fft.ifft2(fft_hires_low, norm="forward").real

        inv_low = torch.abs(inv_low)

        fft_high = fft * (1-mask)
        fr_high = fft_high.real
        fi_high = fft_high.imag

        fft_hires_high = torch.fft.ifftshift(torch.complex(fr_high, fi_high))
        inv_high = torch.fft.ifft2(fft_hires_high, norm="forward").real

        inv_high = torch.abs(inv_high)
        return inv_high

    #     ran_id=random.randint(1,10000)

    #     array = inv_low.squeeze().cpu().numpy()
    #     # 将像素值缩放到0-255之间
    #     array = (array * 255).astype(np.uint8)
    #     # 转换为RGB顺序
    #     array = np.transpose(array, (1, 2, 0))
    #     # 创建图像对象
    #     image = Image.fromarray(array, mode='RGB')
    #     # 保存图像
    #     #os.makedirs('/remote-home/pxy/CWSAM/vis_after_fft_low/')
    #     image.save('/remote-home/pxy/CWSAM/vis_after_fft_low/'+str(ran_id)+'.jpg')

    #     array_h = inv_high.squeeze().cpu().numpy()
    #     # 将像素值缩放到0-255之间
    #     array_h = (array_h * 255).astype(np.uint8)
    #     # 转换为RGB顺序

    #     array_h = np.transpose(array_h, (1, 2, 0))
    #     # 创建图像对象
    #     image = Image.fromarray(array_h, mode='RGB')
    #     # 保存图像
    #    # os.makedirs('/remote-home/pxy/CWSAM/vis_after_fft_high/')
    #     image.save('/remote-home/pxy/CWSAM/vis_after_fft_high/'+str(ran_id)+'.jpg')

    #     x_array = x.squeeze().cpu().numpy()
    #     x_array = (x_array * 255).astype(np.uint8)
    #     x_array = np.transpose(x_array, (1, 2, 0))
    #     x_image = Image.fromarray(x_array, mode='RGB')
    #     #os.makedirs('/remote-home/pxy/CWSAM/vis_inp/')
    #     x_image.save('/remote-home/pxy/CWSAM/vis_inp/'+str(ran_id)+'.jpg')

        #return torch.cat((inv_low,inv_high), dim=1)
        return inv_low
class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)#,groups=2)

    def forward(self, x):
        B, C, H, W = x.shape #1,3,1024,1024
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)  #[1,24,64,64]
        return x
########## adaptor end

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        pool_ratios: Tuple[int, int]=[1, 2, 4, 8]
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )


        self.norm3 = norm_layer(dim//12)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim//12)
        self.norm6 = norm_layer(dim)
        #
        drop_path=0.0
        
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        if window_size > 0:
            self.down_proj = nn.Linear(dim, dim // 12)
            self.attn_0 = Attention(dim // 12, num_heads=8, qkv_bias=qkv_bias, use_rel_pos=True,
                                    rel_pos_zero_init=rel_pos_zero_init,
                                    input_size=(7, 7))
            self.attn_1 = Attention(dim // 12, num_heads=8, qkv_bias=qkv_bias, use_rel_pos=True,
                                    rel_pos_zero_init=rel_pos_zero_init,
                                    input_size=(28, 28))
            self.up_proj = nn.Linear(dim // 12, dim)
            self.p2t_mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.multi_scale_layers = [9, 10]
        self.MLP_Adapter = Adapter(dim, skip_connect=False)  # MLP-adapter, no skip connection
        self.Space_Adapter = Adapter(dim)  # with skip connection\

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        shortcut = x
        
        x = self.norm1(x) #(B, H, W, C)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)     

        x = self.attn(x)
        x = self.Space_Adapter(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        if i in self.multi_scale_layers:
            x = shortcut + x
            H, W = x.shape[1], x.shape[2]
            _x = self.down_proj(x)
            n_x = self.norm5(_x)
            w_x, w_pad_hw = window_partition(n_x, 7)
            up_w_x, up_w_pad_hw = window_partition(n_x, 28)
            w_x = self.drop_path(self.attn_0(w_x))
            up_w_x = self.drop_path(self.attn_1(up_w_x))
            w_x = window_unpartition(w_x, 7, w_pad_hw, (H, W)) + window_unpartition(up_w_x, 28, up_w_pad_hw,
                                                                                    (H, W)) + _x
            w_x = self.up_proj(w_x) + x
            x = w_x + self.mlp(self.norm2(x)) + self.drop_path(self.p2t_mlp(self.norm6(w_x)))
            return x
        else:
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))
            x = x + self.MLP_Adapter(x)
            return x

        # x = shortcut + x
        # x = x + self.mlp(self.norm2(x))
        # x = x + self.MLP_Adapter(x)
        # return x

        # xn = self.norm2(x)
        # x = x + self.mlp(xn) + self.scale * self.MLP_Adapter(xn)
        #
        # return x
        
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
        pool_ratios=[1,2,3,6], eps=1.0e-15):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.eps = eps

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, d_convs=None):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        # q = self.q(x).reshape(B, N, C)
        q = self.q(x).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # print("self.pool_ratios", self.pool_ratios)
        # print("d_convs", d_convs)
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            # pool = nn.AvgPool2d(pool_ratio[0], stride=pool_ratio[1])(x_) + nn.MaxPool2d(pool_ratio[0], stride=pool_ratio[1])(x_)
            pool = F.adaptive_avg_pool2d(x_, (round(H/pool_ratio), round(W/pool_ratio)))  + F.adaptive_max_pool2d(x_, (round(H/pool_ratio), round(W/pool_ratio)))
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, H, W, C)
        
        x = self.proj(x)
        # x = x.view(B, H, W, C)

        return x


class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize//2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0,3,1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.permute(0,2,3,1)


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
