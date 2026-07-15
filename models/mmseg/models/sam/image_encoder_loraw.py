# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

from .common import LayerNorm2d  # 只保留需要的
import math


# ----------------------------
# LoRA building blocks
# ----------------------------
class LoRALinear(nn.Linear):
    """
    nn.Linear with LoRA injected.
    Keeps original parameter names: weight, bias
    so SAM checkpoint keys (e.g. qkv.weight) match directly.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 0.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if self.r > 0:
            # A: (r, in), B: (out, r)
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        if freeze_base:
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base
        y = F.linear(x, self.weight, self.bias)
        # lora
        if self.r > 0:
            x_d = self.lora_dropout(x)
            # (..., in) @ (in, r) -> (..., r)
            z = x_d @ self.lora_A.t()
            # (..., r) @ (r, out) -> (..., out)
            y = y + (z @ self.lora_B.t()) * self.scaling
        return y


class LoRAMLP(nn.Module):
    """MLPBlock 的 LoRA 版本（fc1 + act + fc2），两层都用 LoRALinear。"""
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.fc1 = LoRALinear(
            embedding_dim, mlp_dim, bias=True,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
        )
        self.act = act()
        self.fc2 = LoRALinear(
            mlp_dim, embedding_dim, bias=True,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ----------------------------
# ViT Image Encoder (LoRA)
# ----------------------------
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
        # ---- LoRA options ----
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
        lora_in_attn: bool = True,
        lora_in_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = LoRABlock(
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
                # lora
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                freeze_base=freeze_base,
                lora_in_attn=lora_in_attn,
                lora_in_mlp=lora_in_mlp,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.patch_embed(x)  # [B, H', W', C]
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))  # -> [B, out_chans, H', W']
        return x, None


class LoRABlock(nn.Module):
    """Transformer block without Adapter; use LoRA in Attention/MLP."""

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
        # lora
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
        lora_in_attn: bool = True,
        lora_in_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            # lora
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
            enable_lora=lora_in_attn,
        )

        self.norm2 = norm_layer(dim)
        if lora_in_mlp:
            self.mlp = LoRAMLP(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                act=act_layer,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                freeze_base=freeze_base,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                act_layer(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )
            if freeze_base:
                for p in self.mlp.parameters():
                    p.requires_grad_(False)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head Attention with optional relative pos and optional LoRA."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,  # kept for signature compatibility
        input_size: Optional[Tuple[int, int]] = None,
        # lora
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
        enable_lora: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if enable_lora:
            self.qkv = LoRALinear(
                dim, dim * 3, bias=qkv_bias,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
            )
            self.proj = LoRALinear(
                dim, dim, bias=True,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
            )
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim, bias=True)
            if freeze_base:
                for p in self.qkv.parameters():
                    p.requires_grad_(False)
                for p in self.proj.parameters():
                    p.requires_grad_(False)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            # rel_pos_zero_init: 如果你想保持原行为，可以在外面控制初始化；这里默认就是 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C]
        return: [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert C % self.num_heads == 0, f"C={C} must be divisible by num_heads={self.num_heads}"
        head_dim = C // self.num_heads

        # flatten spatial
        x_flat = x.reshape(B, H * W, C)  # [B, HW, C]

        # qkv projection
        qkv = self.qkv(x_flat)  # [B, HW, 3C]
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4).contiguous()
        # q, k, v: [B, nH, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # merge heads into batch for attention
        q = q.reshape(B * self.num_heads, H * W, head_dim)  # [B*nH, HW, head_dim]
        k = k.reshape(B * self.num_heads, H * W, head_dim)
        v = v.reshape(B * self.num_heads, H * W, head_dim)

        # attention
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B*nH, HW, HW]

        # relative positional encoding (SAM uses this on decomposed rel pos)
        if self.use_rel_pos:
            # add_decomposed_rel_pos expects q shape [B*nH, HW, head_dim]
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)

        # apply attention to v
        out = attn @ v  # [B*nH, HW, head_dim]

        # restore heads
        out = out.reshape(B, self.num_heads, H, W, head_dim).permute(0, 2, 3, 1, 4).contiguous()
        out = out.reshape(B, H, W, C).contiguous()  # [B, H, W, C]

        # output projection (proj expects [B, HW, C])
        out_flat = out.reshape(B, H * W, C)
        out_flat = self.proj(out_flat)  # [B, HW, C]
        out = out_flat.reshape(B, H, W, C)

        return out


# ----------------------------
# helpers (same as original)
# ----------------------------
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
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
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

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
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)         # [B, C, H', W']
        x = x.permute(0, 2, 3, 1)  # -> [B, H', W', C]
        return x
