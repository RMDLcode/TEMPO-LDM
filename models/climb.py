# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""
Modified UNet and ControlNet with Mamba2 and temporal conditioning.
Only the components required by init_latent_diffusion_4 and init_controlnet_4 are kept.
"""
from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import numpy as np
from monai.networks.blocks import Convolution, MLPBlock
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep
from torch import nn
from mamba_ssm import Mamba2

if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops
    has_xformers = True
else:
    xformers = None
    has_xformers = False

__all__ = ["DiffusionModelUNet", "ControlNet"]


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class CrossAttention(nn.Module):
    """Cross attention layer with optional flash attention."""
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads
        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(self, query, key, value):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        return xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

    def _attention(self, query, key, value):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1).to(dtype)
        return torch.bmm(attention_probs, value)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)
        return self.to_out(x)


class BasicTransformerBlock(nn.Module):
    """Transformer block with Mamba2 and cross-attention."""
    def __init__(
        self,
        d_state: int,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=num_channels,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
        self.mamba = Mamba2(d_model=num_channels, d_state=d_state, d_conv=4, expand=4)
        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, act="GEGLU", dropout_rate=dropout)
        self.attn2 = CrossAttention(
            query_dim=num_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = self.mamba(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """Transformer block for image-like data."""
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        d_state: int,
        num_attention_heads: int,
        num_head_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                d_state=d_state,
                num_channels=inner_dim,
                num_attention_heads=num_attention_heads,
                num_head_channels=num_head_channels,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
            ) for _ in range(num_layers)
        ])
        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        B, C = x.shape[:2]
        spatial_dims = x.shape[2:]
        x = self.norm(x)
        x = self.proj_in(x)
        inner_dim = x.shape[1]
        x = x.flatten(2).permute(0, 2, 1)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.permute(0, 2, 1).reshape(B, inner_dim, *spatial_dims).contiguous()
        x = self.proj_out(x)
        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block."""
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_heads = num_channels // num_head_channels if num_head_channels else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)
        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(self, query, key, value):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        return xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

    def _attention(self, query, key, value):
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        return torch.bmm(attention_probs, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, C = x.shape[:2]
        spatial_dims = x.shape[2:]
        x = self.norm(x)
        x = x.flatten(2).permute(0, 2, 1)
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)
        x = x.permute(0, 2, 1).reshape(B, C, *spatial_dims)
        return x + residual


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


class Downsample(nn.Module):
    def __init__(self, spatial_dims, num_channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must match when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x, emb=None):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, spatial_dims, num_channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            self.conv = None

    def forward(self, x, emb=None):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        temb_channels,
        up=False,
        down=False,
        norm_num_groups=32,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims, in_channels, out_channels, strides=1, kernel_size=3, padding=1, conv_only=True
        )
        self.upsample = None
        self.downsample = None
        if self.up:
            self.upsample = Upsample(spatial_dims, in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample(spatial_dims, in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(spatial_dims, out_channels, out_channels, strides=1, kernel_size=3, padding=1, conv_only=True)
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims, in_channels, out_channels, strides=1, kernel_size=1, padding=0, conv_only=True
            )

    def forward(self, x, emb):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)
        temb = self.time_emb_proj(self.nonlinearity(emb))
        if self.spatial_dims == 2:
            temb = temb[:, :, None, None]
        else:
            temb = temb[:, :, None, None, None]
        h = h + temb
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        return self.skip_connection(x) + h


class DownBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, temb_channels, num_res_blocks=1,
                 norm_num_groups=32, norm_eps=1e-6, add_downsample=True, resblock_updown=False, downsample_padding=1):
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []
        for i in range(num_res_blocks):
            cin = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock(spatial_dims, cin, out_channels, temb_channels,
                                       norm_num_groups=norm_num_groups, norm_eps=norm_eps))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(spatial_dims, out_channels, out_channels, temb_channels,
                                               down=True, norm_num_groups=norm_num_groups, norm_eps=norm_eps)
            else:
                self.downsampler = Downsample(spatial_dims, out_channels, use_conv=True, out_channels=out_channels,
                                              padding=downsample_padding)
        else:
            self.downsampler = None

    def forward(self, hidden_states, temb, context=None):
        output_states = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)
        return hidden_states, output_states


class AttnDownBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, temb_channels, num_res_blocks=1,
                 norm_num_groups=32, norm_eps=1e-6, add_downsample=True, resblock_updown=False,
                 downsample_padding=1, num_head_channels=1, use_flash_attention=False):
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            cin = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock(spatial_dims, cin, out_channels, temb_channels,
                                       norm_num_groups=norm_num_groups, norm_eps=norm_eps))
            attentions.append(AttentionBlock(spatial_dims, out_channels, num_head_channels,
                                             norm_num_groups, norm_eps, use_flash_attention))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(spatial_dims, out_channels, out_channels, temb_channels,
                                               down=True, norm_num_groups=norm_num_groups, norm_eps=norm_eps)
            else:
                self.downsampler = Downsample(spatial_dims, out_channels, use_conv=True, out_channels=out_channels,
                                              padding=downsample_padding)
        else:
            self.downsampler = None

    def forward(self, hidden_states, temb, context=None):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states.append(hidden_states)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)
        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, temb_channels, d_state, num_res_blocks=1,
                 norm_num_groups=32, norm_eps=1e-6, add_downsample=True, resblock_updown=False,
                 downsample_padding=1, num_head_channels=1, transformer_num_layers=1,
                 cross_attention_dim=None, upcast_attention=False, use_flash_attention=False, dropout_cattn=0.0):
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            cin = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock(spatial_dims, cin, out_channels, temb_channels,
                                       norm_num_groups=norm_num_groups, norm_eps=norm_eps))
            attentions.append(SpatialTransformer(
                spatial_dims, out_channels, d_state,
                num_attention_heads=out_channels // num_head_channels,
                num_head_channels=num_head_channels,
                num_layers=transformer_num_layers,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout=dropout_cattn,
            ))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(spatial_dims, out_channels, out_channels, temb_channels,
                                               down=True, norm_num_groups=norm_num_groups, norm_eps=norm_eps)
            else:
                self.downsampler = Downsample(spatial_dims, out_channels, use_conv=True, out_channels=out_channels,
                                              padding=downsample_padding)
        else:
            self.downsampler = None

    def forward(self, hidden_states, temb, context=None):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)
            output_states.append(hidden_states)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)
        return hidden_states, output_states


class AttnMidBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, temb_channels, norm_num_groups=32, norm_eps=1e-6,
                 num_head_channels=1, use_flash_attention=False):
        super().__init__()
        self.resnet_1 = ResnetBlock(spatial_dims, in_channels, in_channels, temb_channels,
                                    norm_num_groups=norm_num_groups, norm_eps=norm_eps)
        self.attention = AttentionBlock(spatial_dims, in_channels, num_head_channels,
                                        norm_num_groups, norm_eps, use_flash_attention)
        self.resnet_2 = ResnetBlock(spatial_dims, in_channels, in_channels, temb_channels,
                                    norm_num_groups=norm_num_groups, norm_eps=norm_eps)

    def forward(self, hidden_states, temb, context=None):
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states


class CrossAttnMidBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, temb_channels, d_state, norm_num_groups=32, norm_eps=1e-6,
                 num_head_channels=1, transformer_num_layers=1, cross_attention_dim=None,
                 upcast_attention=False, use_flash_attention=False, dropout_cattn=0.0):
        super().__init__()
        self.resnet_1 = ResnetBlock(spatial_dims, in_channels, in_channels, temb_channels,
                                    norm_num_groups=norm_num_groups, norm_eps=norm_eps)
        self.attention = SpatialTransformer(
            spatial_dims, in_channels, d_state,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout=dropout_cattn,
        )
        self.resnet_2 = ResnetBlock(spatial_dims, in_channels, in_channels, temb_channels,
                                    norm_num_groups=norm_num_groups, norm_eps=norm_eps)

    def forward(self, hidden_states, temb, context=None):
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states


class UpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels,
                 num_res_blocks=1, norm_num_groups=32, norm_eps=1e-6, add_upsample=True, resblock_updown=False):
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []
        for i in range(num_res_blocks):
            res_skip_channels = in_channels if i == num_res_blocks - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock(
                spatial_dims,
                resnet_in_channels + res_skip_channels,
                out_channels,
                temb_channels,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps
            ))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(spatial_dims, out_channels, out_channels, temb_channels,
                                             up=True, norm_num_groups=norm_num_groups, norm_eps=norm_eps)
            else:
                self.upsampler = Upsample(spatial_dims, out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsampler = None

    def forward(self, hidden_states, res_hidden_states_list, temb, context=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)
        return hidden_states


class AttnUpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels,
                 num_res_blocks=1, norm_num_groups=32, norm_eps=1e-6, add_upsample=True, resblock_updown=False,
                 num_head_channels=1, use_flash_attention=False):
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            res_skip_channels = in_channels if i == num_res_blocks - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock(
                spatial_dims,
                resnet_in_channels + res_skip_channels,
                out_channels,
                temb_channels,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps
            ))
            attentions.append(AttentionBlock(spatial_dims, out_channels, num_head_channels,
                                             norm_num_groups, norm_eps, use_flash_attention))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(spatial_dims, out_channels, out_channels, temb_channels,
                                             up=True, norm_num_groups=norm_num_groups, norm_eps=norm_eps)
            else:
                self.upsampler = Upsample(spatial_dims, out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsampler = None

    def forward(self, hidden_states, res_hidden_states_list, temb, context=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)
        return hidden_states


class CrossAttnUpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels, d_state,
                 num_res_blocks=1, norm_num_groups=32, norm_eps=1e-6, add_upsample=True, resblock_updown=False,
                 num_head_channels=1, transformer_num_layers=1, cross_attention_dim=None,
                 upcast_attention=False, use_flash_attention=False, dropout_cattn=0.0):
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            res_skip_channels = in_channels if i == num_res_blocks - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock(
                spatial_dims,
                resnet_in_channels + res_skip_channels,
                out_channels,
                temb_channels,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps
            ))
            attentions.append(SpatialTransformer(
                spatial_dims, out_channels, d_state,
                num_attention_heads=out_channels // num_head_channels,
                num_head_channels=num_head_channels,
                num_layers=transformer_num_layers,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout=dropout_cattn,
            ))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(spatial_dims, out_channels, out_channels, temb_channels,
                                             up=True, norm_num_groups=norm_num_groups, norm_eps=norm_eps)
            else:
                self.upsampler = Upsample(spatial_dims, out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsampler = None

    def forward(self, hidden_states, res_hidden_states_list, temb, context=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)
        return hidden_states


def get_down_block(
    spatial_dims, in_channels, out_channels, temb_channels, d_state, num_res_blocks,
    norm_num_groups, norm_eps, add_downsample, resblock_updown,
    with_attn, with_cross_attn, num_head_channels, transformer_num_layers,
    cross_attention_dim, upcast_attention=False, use_flash_attention=False, dropout_cattn=0.0
):
    if with_attn:
        return AttnDownBlock(
            spatial_dims, in_channels, out_channels, temb_channels,
            num_res_blocks, norm_num_groups, norm_eps, add_downsample,
            resblock_updown, 1, num_head_channels, use_flash_attention
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            spatial_dims, in_channels, out_channels, temb_channels, d_state,
            num_res_blocks, norm_num_groups, norm_eps, add_downsample,
            resblock_updown, 1, num_head_channels, transformer_num_layers,
            cross_attention_dim, upcast_attention, use_flash_attention, dropout_cattn
        )
    else:
        return DownBlock(
            spatial_dims, in_channels, out_channels, temb_channels,
            num_res_blocks, norm_num_groups, norm_eps, add_downsample, resblock_updown
        )


def get_mid_block(
    spatial_dims, in_channels, temb_channels, d_state, norm_num_groups, norm_eps,
    with_conditioning, num_head_channels, transformer_num_layers,
    cross_attention_dim, upcast_attention=False, use_flash_attention=False, dropout_cattn=0.0
):
    if with_conditioning:
        return CrossAttnMidBlock(
            spatial_dims, in_channels, temb_channels, d_state,
            norm_num_groups, norm_eps, num_head_channels,
            transformer_num_layers, cross_attention_dim,
            upcast_attention, use_flash_attention, dropout_cattn
        )
    else:
        return AttnMidBlock(
            spatial_dims, in_channels, temb_channels,
            norm_num_groups, norm_eps, num_head_channels, use_flash_attention
        )


def get_up_block(
    spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels, d_state,
    num_res_blocks, norm_num_groups, norm_eps, add_upsample, resblock_updown,
    with_attn, with_cross_attn, num_head_channels, transformer_num_layers,
    cross_attention_dim, upcast_attention=False, use_flash_attention=False, dropout_cattn=0.0
):
    if with_attn:
        return AttnUpBlock(
            spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels,
            num_res_blocks, norm_num_groups, norm_eps, add_upsample,
            resblock_updown, num_head_channels, use_flash_attention
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels, d_state,
            num_res_blocks, norm_num_groups, norm_eps, add_upsample,
            resblock_updown, num_head_channels, transformer_num_layers,
            cross_attention_dim, upcast_attention, use_flash_attention, dropout_cattn
        )
    else:
        return UpBlock(
            spatial_dims, in_channels, prev_output_channel, out_channels, temb_channels,
            num_res_blocks, norm_num_groups, norm_eps, add_upsample, resblock_updown
        )


class DiffusionModelUNet(nn.Module):
    """
    UNet with timestep embedding and cross-attention.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        d_states: Sequence[int] = (256, 128, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        if with_conditioning and cross_attention_dim is None:
            raise ValueError("cross_attention_dim required when with_conditioning=True")
        if cross_attention_dim is not None and not with_conditioning:
            raise ValueError("with_conditioning must be True when cross_attention_dim is set")
        if any(c % norm_num_groups != 0 for c in num_channels):
            raise ValueError("num_channels must be divisible by norm_num_groups")
        if len(num_channels) != len(attention_levels):
            raise ValueError("num_channels and attention_levels length mismatch")
        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))
        if len(num_head_channels) != len(attention_levels):
            raise ValueError("num_head_channels length mismatch")
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
        if len(num_res_blocks) != len(num_channels):
            raise ValueError("num_res_blocks length mismatch")
        if use_flash_attention and not has_xformers:
            raise ValueError("xformers not installed for flash attention")
        if use_flash_attention and not torch.cuda.is_available():
            raise ValueError("Flash attention requires CUDA")

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims, in_channels, num_channels[0], strides=1, kernel_size=3, padding=1, conv_only=True
        )
        # time embedding
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # down blocks
        self.down_blocks = nn.ModuleList()
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final = i == len(num_channels) - 1
            down_block = get_down_block(
                spatial_dims, input_channel, output_channel, time_embed_dim, d_states[i],
                num_res_blocks[i], norm_num_groups, norm_eps, not is_final, resblock_updown,
                attention_levels[i] and not with_conditioning,
                attention_levels[i] and with_conditioning,
                num_head_channels[i], transformer_num_layers, cross_attention_dim,
                upcast_attention, use_flash_attention, dropout_cattn
            )
            self.down_blocks.append(down_block)

        # mid block
        self.middle_block = get_mid_block(
            spatial_dims, num_channels[-1], time_embed_dim, d_states[-1],
            norm_num_groups, norm_eps, with_conditioning,
            num_head_channels[-1], transformer_num_layers, cross_attention_dim,
            upcast_attention, use_flash_attention, dropout_cattn
        )

        # up blocks
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(num_channels))
        reversed_res_blocks = list(reversed(num_res_blocks))
        reversed_d_states = list(reversed(d_states))
        reversed_attn = list(reversed(attention_levels))
        reversed_head = list(reversed(num_head_channels))
        output_channel = reversed_channels[0]
        for i in range(len(reversed_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            input_channel = reversed_channels[min(i+1, len(num_channels)-1)]
            is_final = i == len(num_channels) - 1
            up_block = get_up_block(
                spatial_dims, input_channel, prev_output_channel, output_channel, time_embed_dim,
                reversed_d_states[i], reversed_res_blocks[i] + 1, norm_num_groups, norm_eps,
                not is_final, resblock_updown,
                reversed_attn[i] and not with_conditioning,
                reversed_attn[i] and with_conditioning,
                reversed_head[i], transformer_num_layers, cross_attention_dim,
                upcast_attention, use_flash_attention, dropout_cattn
            )
            self.up_blocks.append(up_block)

        # output
        self.out = nn.Sequential(
            nn.GroupNorm(norm_num_groups, num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(spatial_dims, num_channels[0], out_channels, strides=1, kernel_size=3, padding=1, conv_only=True)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        use_positional_encoding: bool = False,
        positional_encoding_dim: int = 256,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_outputs = []
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0]).to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels required")
            class_emb = self.class_embedding(class_labels).to(dtype=x.dtype)
            emb = emb + class_emb

        # Optional positional encoding for context
        if context is not None and use_positional_encoding:
            B, _, C = context.shape
            temporal_feature = context[:, :, 0].squeeze(1)
            pos_enc = get_sinusoidal_positional_encoding(temporal_feature, positional_encoding_dim)
            context_modified = context.clone()
            context_modified[:, :, 0] = pos_enc
            context = context_modified

        h = self.conv_in(x)
        all_outputs.append(h)

        down_block_res_samples = [h]
        for down_block in self.down_blocks:
            h, res_samples = down_block(h, emb, context)
            all_outputs.append(h)
            down_block_res_samples.extend(res_samples)

        if down_block_additional_residuals is not None:
            new_res = []
            for res, add_res in zip(down_block_res_samples, down_block_additional_residuals):
                new_res.append(res + add_res)
            down_block_res_samples = new_res

        h = self.middle_block(h, emb, context)
        all_outputs.append(h)
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            h = up_block(h, res_samples, emb, context)
            all_outputs.append(h)

        h = self.out(h)
        return h, all_outputs


class ControlNetConditioningEmbedding(nn.Module):
    """
    Conditioning embedding network for ControlNet.
    """
    def __init__(self, spatial_dims, in_channels, out_channels, num_channels=(16, 32, 96, 256)):
        super().__init__()
        self.conv_in = Convolution(
            spatial_dims, in_channels, num_channels[0], strides=1, kernel_size=3, padding=1, conv_only=True
        )
        self.blocks = nn.ModuleList()
        for i in range(len(num_channels) - 1):
            self.blocks.append(
                Convolution(spatial_dims, num_channels[i], num_channels[i], strides=1, kernel_size=3, padding=1, conv_only=True)
            )
            self.blocks.append(
                Convolution(spatial_dims, num_channels[i], num_channels[i+1], strides=1, kernel_size=3, padding=1, conv_only=True)
            )
        self.conv_out = zero_module(
            Convolution(spatial_dims, num_channels[-1], out_channels, strides=1, kernel_size=3, padding=1, conv_only=True)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


def get_sinusoidal_positional_encoding(position: torch.Tensor, d_model: int, max_period: int = 10000) -> torch.Tensor:
    """Generate sinusoidal positional encoding for given position values (1D or 2D)."""
    original_shape = position.shape
    device = position.device
    if position.dim() == 1:
        position = position.unsqueeze(1)
    flat = position.reshape(-1, 1)
    B_T = flat.shape[0]
    half_dim = d_model // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim, device=device) / half_dim)
    args = flat * freqs.unsqueeze(0)
    pe_flat = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if d_model % 2 == 1:
        pe_flat = F.pad(pe_flat, (0, 1, 0, 0))
    if len(original_shape) == 1:
        return pe_flat.reshape(original_shape[0], d_model)
    else:
        return pe_flat.reshape(original_shape[0], original_shape[1], d_model)


class TemporalAwareConditioningEmbedding_T(nn.Module):
    """
    Temporal-aware conditioning embedding that separates image and time features.
    This version includes target time as an extra group.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: Sequence[int],
        out_channels: int,
        img_channels: int = 3,
        time_dim: int = 1,
        mode: str = 'conv',
        num_heads: int = 8,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.img_channels = img_channels
        self.time_dim = time_dim
        self.mode = mode
        self.num_heads = num_heads

        self.num_groups = in_channels // (img_channels + time_dim)
        if in_channels % (img_channels + time_dim) != 0:
            raise ValueError("in_channels must be divisible by (img_channels + time_dim)")

        self.img_proj = Convolution(
            spatial_dims,
            img_channels,
            num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.d_model = num_channels[0]

        if mode == 'conv':
            self.final_proj = zero_module(
                Convolution(
                    spatial_dims,
                    self.d_model * (self.num_groups + 1),
                    out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )
        elif mode == 'attention':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=self.d_model * 4,
                batch_first=True,
                norm_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.final_proj = zero_module(
                Convolution(
                    spatial_dims,
                    self.d_model,
                    out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )
        else:
            raise ValueError("mode must be 'conv' or 'attention'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial_dims = x.shape
        N = np.prod(spatial_dims)

        # split into image and history time, and target time
        img_end_idx = self.num_groups * self.img_channels
        img_features = x[:, :img_end_idx, ...]
        history_time_features = x[:, img_end_idx:-1, ...]
        target_time_features = x[:, -1:, ...]

        img_features = img_features.view(B, self.num_groups, self.img_channels, *spatial_dims)
        history_time_features = history_time_features.view(B, self.num_groups, self.time_dim, *spatial_dims)
        target_time_features = target_time_features.view(B, 1, self.time_dim, *spatial_dims)

        history_time_scalars = history_time_features.mean(dim=(3,4,5), keepdim=False)
        target_time_scalars = target_time_features.mean(dim=(3,4,5), keepdim=False)
        if self.time_dim == 1:
            history_time_scalars = history_time_scalars.squeeze(-1)
            target_time_scalars = target_time_scalars.squeeze(-1)

        if self.mode == 'conv':
            processed = []
            for t in range(self.num_groups):
                img_t = img_features[:, t, ...]
                img_emb = self.img_proj(img_t)
                time_val = history_time_scalars[:, t] if self.time_dim == 1 else history_time_scalars[:, t, :]
                pos_enc = get_sinusoidal_positional_encoding(time_val, self.d_model)
                pos_enc_exp = pos_enc.view(B, self.d_model, 1, 1, 1).expand(-1, -1, *spatial_dims)
                processed.append(img_emb + pos_enc_exp)

            # target time
            target_pos_enc = get_sinusoidal_positional_encoding(target_time_scalars, self.d_model)
            target_pos_exp = target_pos_enc.view(B, self.d_model, 1, 1, 1).expand(-1, -1, *spatial_dims)
            target_img_placeholder = torch.zeros(B, self.d_model, *spatial_dims, device=x.device)
            processed.append(target_img_placeholder + target_pos_exp)

            x_out = torch.cat(processed, dim=1)
        else:  # attention
            sequence_list = []
            for t in range(self.num_groups):
                img_t = img_features[:, t, ...]
                img_emb = self.img_proj(img_t)
                seq = img_emb.flatten(2).permute(0, 2, 1)
                sequence_list.append(seq)

            target_img_placeholder = torch.zeros(B, self.d_model, *spatial_dims, device=x.device)
            target_seq = target_img_placeholder.flatten(2).permute(0, 2, 1)
            sequence_list.append(target_seq)
            sequence = torch.stack(sequence_list, dim=1)  # [B, T, N, D]

            all_time_scalars = torch.cat([history_time_scalars, target_time_scalars.unsqueeze(1)], dim=1)
            pos_enc = get_sinusoidal_positional_encoding(all_time_scalars, self.d_model)
            pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, sequence.shape[2], -1)
            sequence = sequence + pos_enc

            B_, T, N, D = sequence.shape
            sequence = sequence.reshape(B_ * T, N, D)
            sequence = self.temporal_encoder(sequence)
            sequence = sequence.reshape(B_, T, N, D)
            x_out = sequence.mean(dim=1)  # [B, N, D]
            x_out = x_out.permute(0, 2, 1).reshape(B, self.d_model, *spatial_dims)

        return self.final_proj(x_out)


class ControlNet(nn.Module):
    """
    ControlNet network.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        d_states: Sequence[int] = (256, 128, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        use_group_conv: bool = False,
        conditioning_embedding_in_channels: int = 1,
        conditioning_embedding_num_channels: Sequence[int] | None = (16, 32, 96, 256),
    ) -> None:
        super().__init__()
        if with_conditioning and cross_attention_dim is None:
            raise ValueError("cross_attention_dim required with with_conditioning")
        if cross_attention_dim is not None and not with_conditioning:
            raise ValueError("with_conditioning must be True if cross_attention_dim set")
        if any(c % norm_num_groups != 0 for c in num_channels):
            raise ValueError("num_channels must be divisible by norm_num_groups")
        if len(num_channels) != len(attention_levels):
            raise ValueError("num_channels and attention_levels length mismatch")
        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))
        if len(num_head_channels) != len(attention_levels):
            raise ValueError("num_head_channels length mismatch")
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
        if len(num_res_blocks) != len(num_channels):
            raise ValueError("num_res_blocks length mismatch")
        if use_flash_attention and not torch.cuda.is_available():
            raise ValueError("Flash attention requires CUDA")

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        self.conv_in = Convolution(
            spatial_dims, in_channels, num_channels[0], strides=1, kernel_size=3, padding=1, conv_only=True
        )
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # conditioning embedding
        if not use_group_conv:
            self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
                spatial_dims,
                conditioning_embedding_in_channels,
                num_channels[0],
                num_channels=conditioning_embedding_num_channels,
            )
        else:
            self.controlnet_cond_embedding = TemporalAwareConditioningEmbedding_T(
                spatial_dims,
                conditioning_embedding_in_channels,
                conditioning_embedding_num_channels,
                num_channels[0],
                img_channels=3,
                time_dim=1,
                mode='conv',
            )

        # down blocks and controlnet outputs
        self.down_blocks = nn.ModuleList()
        self.controlnet_down_blocks = nn.ModuleList()
        output_channel = num_channels[0]

        # first controlnet block (after conv_in)
        cb = zero_module(
            Convolution(spatial_dims, output_channel, output_channel, strides=1, kernel_size=1, padding=0, conv_only=True)
        )
        self.controlnet_down_blocks.append(cb)

        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final = i == len(num_channels) - 1
            down_block = get_down_block(
                spatial_dims, input_channel, output_channel, time_embed_dim, d_states[i],
                num_res_blocks[i], norm_num_groups, norm_eps, not is_final, resblock_updown,
                attention_levels[i] and not with_conditioning,
                attention_levels[i] and with_conditioning,
                num_head_channels[i], transformer_num_layers, cross_attention_dim,
                upcast_attention, use_flash_attention, 0.0
            )
            self.down_blocks.append(down_block)

            for _ in range(num_res_blocks[i]):
                cb = zero_module(
                    Convolution(spatial_dims, output_channel, output_channel, strides=1, kernel_size=1, padding=0, conv_only=True)
                )
                self.controlnet_down_blocks.append(cb)
            if not is_final:
                cb = zero_module(
                    Convolution(spatial_dims, output_channel, output_channel, strides=1, kernel_size=1, padding=0, conv_only=True)
                )
                self.controlnet_down_blocks.append(cb)

        # mid block
        self.middle_block = get_mid_block(
            spatial_dims, num_channels[-1], time_embed_dim, d_states[-1],
            norm_num_groups, norm_eps, with_conditioning,
            num_head_channels[-1], transformer_num_layers, cross_attention_dim,
            upcast_attention, use_flash_attention, 0.0
        )
        cb = zero_module(
            Convolution(spatial_dims, output_channel, output_channel, strides=1, kernel_size=1, padding=0, conv_only=True)
        )
        self.controlnet_mid_block = cb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0]).to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels required")
            class_emb = self.class_embedding(class_labels).to(dtype=x.dtype)
            emb = emb + class_emb

        h = self.conv_in(x)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        h += controlnet_cond

        down_res = [h]
        for down_block in self.down_blocks:
            h, res_samples = down_block(h, emb, context)
            down_res.extend(res_samples)

        h = self.middle_block(h, emb, context)

        # produce controlnet outputs
        controlnet_down_res = tuple(
            block(res) for res, block in zip(down_res, self.controlnet_down_blocks)
        )
        mid_res = self.controlnet_mid_block(h)

        controlnet_down_res = tuple(r * conditioning_scale for r in controlnet_down_res)
        mid_res = mid_res * conditioning_scale

        return controlnet_down_res, mid_res


# ========== Helper to load checkpoints ==========
def load_if(checkpoints_path, network, map_location=None):
    if checkpoints_path is not None:
        network.load_state_dict(torch.load(checkpoints_path, map_location=map_location))
    return network


# ========== Public initializers ==========
def init_latent_diffusion_4(checkpoints_path=None, map_location=None, multi_heads=False):
    """Initialize UNet with 4 resolution levels."""
    if multi_heads:
        num_head_channels = (16, 32, 64, 128)
    else:
        num_head_channels = (128, 256, 512, 1024)
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        num_channels=(128, 256, 512, 1024),
        d_states=(512, 256, 128, 64),
        attention_levels=(True, True, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=num_head_channels,
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=7,
        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False,
    )
    return load_if(checkpoints_path, model, map_location)


def init_controlnet_4(checkpoints_path=None, imput_time_num=3, map_location=None, input_T=False, multi_heads=False):
    """Initialize ControlNet with 4 resolution levels."""
    if multi_heads:
        num_head_channels = (16, 32, 64, 128)
    else:
        num_head_channels = (128, 256, 512, 1024)
    model = ControlNet(
        spatial_dims=3,
        in_channels=3,
        num_res_blocks=2,
        num_channels=(128, 256, 512, 1024),
        d_states=(512, 256, 128, 64),
        attention_levels=(True, True, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=num_head_channels,
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=7,
        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False,
        use_group_conv=input_T,
        conditioning_embedding_in_channels=imput_time_num * 4,
        conditioning_embedding_num_channels=(256,),
    )
    return load_if(checkpoints_path, model, map_location) 