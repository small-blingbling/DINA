# ============================================================
# DS-Net （Facebook 原始版本 + tiny/small 两个模型）
# 支持 MixVisionTransformer: tiny / small
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict


# ============================================================
# Utilities
# ============================================================

def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor.normal_().fmod_(2).mul_(std).add_(mean)


def to_2tuple(x):
    return (x, x)


class DropPath(nn.Module):
    """Stochastic depth"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================
# MLP & CMlp
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# Attention 子模块（纯注意力）
# ============================================================

class Attention_pure(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // 3
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(x)


class Cross_Attention_pure(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, shape):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(shape[0], shape[1], shape[2])
        return self.proj_drop(x)


# ============================================================
# MixBlock（核心模块）
# ============================================================

class MixBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, downsample=2):
        super().__init__()

        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.dim = dim
        self.dim_conv = dim // 2
        self.dim_sa = dim - self.dim_conv

        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

        self.norm_conv1 = nn.BatchNorm2d(self.dim_conv)
        self.norm_sa1 = nn.LayerNorm(self.dim_sa)

        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, padding=1, groups=self.dim_conv)
        self.channel_up = nn.Linear(self.dim_sa, 3 * self.dim_sa)

        self.cross_channel_up_conv = nn.Conv2d(self.dim_conv, 3 * self.dim_conv, 1)
        self.cross_channel_up_sa = nn.Linear(self.dim_sa, 3 * self.dim_sa)

        self.fuse_channel_conv = nn.Linear(self.dim_conv, self.dim_conv)
        self.fuse_channel_sa = nn.Linear(self.dim_sa, self.dim_sa)

        self.attn = Attention_pure(self.dim_sa, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Cross_Attention_pure(self.dim_sa, num_heads=num_heads,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               attn_drop=attn_drop, proj_drop=drop)

        self.norm_conv2 = nn.BatchNorm2d(self.dim_conv)
        self.norm_sa2 = nn.LayerNorm(self.dim_sa)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.downsample = downsample

    def forward(self, x):

        x = x + self.pos_embed(x)
        B, _, H, W = x.shape
        residual = x

        x = self.norm1(x)
        x = self.conv1(x)

        # Conv branch
        conv = x[:, self.dim_sa:, :, :]
        conv = conv + self.conv(self.norm_conv1(conv))

        # SA branch
        qkv = x[:, :self.dim_sa, :]
        sa = F.interpolate(qkv, size=(H // self.downsample, W // self.downsample))

        B, C, Hs, Ws = sa.shape
        sa = sa.flatten(2).transpose(1, 2)
        residual_sa = sa
        sa = self.norm_sa1(sa)
        sa = self.channel_up(sa)
        sa = residual_sa + self.attn(sa)

        # Cross attention
        conv_qkv = self.cross_channel_up_conv(self.norm_conv2(conv)).flatten(2).transpose(1, 2)
        sa_qkv = self.cross_channel_up_sa(self.norm_sa2(sa))

        Bc, Nc, Cc = conv_qkv.shape
        Cc = Cc // 3
        conv_q, conv_k, conv_v = conv_qkv.reshape(Bc, Nc, 3, -1).permute(2, 0, 1, 3)

        Bs, Ns, Cs = sa_qkv.shape
        Cs = Cs // 3
        sa_q, sa_k, sa_v = sa_qkv.reshape(Bs, Ns, 3, -1).permute(2, 0, 1, 3)

        conv = self.cross_attn(conv_q, sa_k, sa_v, shape=(Bc, Nc, Cc))
        conv = self.fuse_channel_conv(conv)
        conv = conv.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        sa = self.cross_attn(sa_q, conv_k, conv_v, shape=(Bs, Ns, Cs))
        sa = self.fuse_channel_sa(sa)
        sa = sa.reshape(B, Hs, Ws, -1).permute(0, 3, 1, 2)
        sa = F.interpolate(sa, size=(H, W))

        x = torch.cat([conv, sa], dim=1)
        x = residual + self.drop_path(self.conv2(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        H, W = x.shape[-2:]

        # 如果图像尺寸不是 img_size，自动 padding 成最近的 patch 对齐尺寸
        pad_h = (self.img_size[0] - H) if H < self.img_size[0] else 0
        pad_w = (self.img_size[1] - W) if W < self.img_size[1] else 0

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return self.proj(x)


# ============================================================
# MixVisionTransformer（主干骨架）
# ============================================================

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=[64, 128, 320, 512], depth=[3, 4, 8, 3],
                 num_heads=[1, 2, 5, 8], mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        # 生成 block 列表（注意：depth）
        self.blocks1 = nn.ModuleList([
            MixBlock(embed_dim[0], num_heads[0], mlp_ratio, qkv_bias,
                     qk_scale, drop_rate, attn_drop_rate,
                     drop_path_rate, norm_layer=norm_layer, downsample=8)
            for _ in range(depth[0])
        ])

        self.blocks2 = nn.ModuleList([
            MixBlock(embed_dim[1], num_heads[1], mlp_ratio, qkv_bias,
                     qk_scale, drop_rate, attn_drop_rate,
                     drop_path_rate, norm_layer=norm_layer, downsample=4)
            for _ in range(depth[1])
        ])

        self.blocks3 = nn.ModuleList([
            MixBlock(embed_dim[2], num_heads[2], mlp_ratio, qkv_bias,
                     qk_scale, drop_rate, attn_drop_rate,
                     drop_path_rate, norm_layer=norm_layer, downsample=2)
            for _ in range(depth[2])
        ])

        self.blocks4 = nn.ModuleList([
            MixBlock(embed_dim[3], num_heads[3], mlp_ratio, qkv_bias,
                     qk_scale, drop_rate, attn_drop_rate,
                     drop_path_rate, norm_layer=norm_layer, downsample=2)
            for _ in range(depth[3])
        ])

        self.norm = nn.BatchNorm2d(embed_dim[-1])
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):

        x = self.patch_embed1(x)

        for blk in self.blocks1:
            x = blk(x)

        x = self.patch_embed2(x)

        for blk in self.blocks2:
            x = blk(x)

        x = self.patch_embed3(x)

        for blk in self.blocks3:
            x = blk(x)

        x = self.patch_embed4(x)

        for blk in self.blocks4:
            x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x


# ============================================================
# Tiny version
# ============================================================

def ds_net_tiny(pretrained=False, **kwargs):
    model = MixVisionTransformer(
        patch_size=16, depth=[2, 2, 4, 1], mlp_ratio=4, qkv_bias=True,
        img_size=100, **kwargs
    )
    return model


# ============================================================
# Small version（你的模型）
# ============================================================

def ds_net_small(pretrained=False, **kwargs):
    # 如果外部没有传 img_size，则默认 100
    kwargs.setdefault("img_size", 68)
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("depth", [3, 4, 12, 3])
    kwargs.setdefault("mlp_ratio", 4)
    kwargs.setdefault("qkv_bias", True)

    model = MixVisionTransformer(**kwargs)
    return model

