import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal


class TriAxialStripGating(nn.Module):
    """Tri-Axial Strip Gating (TASG) for 3D feature maps.

    The input and output use ``(B, C, D, H, W)`` layout. A local depth-wise
    convolution is followed by depth-, height-, and width-axis strip
    convolutions. The sigmoid gate is applied with residual modulation.
    """

    def __init__(self, dim, k=11, k_local=5, use_sigmoid=True, residual=True):
        super().__init__()
        assert k % 2 == 1 and k_local % 2 == 1, "k and k_local must be odd"

        self.conv_local = nn.Conv3d(dim, dim, kernel_size=k_local, padding=k_local // 2, groups=dim, bias=False)
        self.conv_d = nn.Conv3d(dim, dim, kernel_size=(k, 1, 1), padding=(k // 2, 0, 0), groups=dim, bias=False)
        self.conv_h = nn.Conv3d(dim, dim, kernel_size=(1, k, 1), padding=(0, k // 2, 0), groups=dim, bias=False)
        self.conv_w = nn.Conv3d(dim, dim, kernel_size=(1, 1, k), padding=(0, 0, k // 2), groups=dim, bias=False)
        self.pw = nn.Conv3d(dim, dim, kernel_size=1, bias=True)

        self.act = nn.Sigmoid() if use_sigmoid else nn.Identity()
        self.residual = residual

    def forward(self, x):
        attn = self.conv_local(x)
        attn = self.conv_d(attn)
        attn = self.conv_h(attn)
        attn = self.conv_w(attn)
        attn = self.pw(attn)
        attn = self.act(attn)

        if self.residual:
            return x * (1.0 + attn)
        return x * attn

class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode
    def forward(self, src, flow):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        # Explicit indexing to avoid future torch.meshgrid warning
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2,1,0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode
    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x

def window_partition(x, window_size):
    B, H, W, T, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], T // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0]*window_size[1]*window_size[2], C)
    return windows
def window_reverse(windows, window_size, dims):
    B, H, W, T = dims
    x = windows.view(B, H // window_size[0], W // window_size[1], T // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, T, -1)
    return x

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim * 2, dim)
        self.gule1 = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.dim = dim
    def forward(self, x):
        x = self.linear1(x)
        x = self.gule1(x)
        x = self.linear2(x)
        return x


class HeterogeneousAxialMixingAttention(nn.Module):
    """Heterogeneous Axial Mixing Attention (HAMA).

    A value branch is modulated by a structural branch whose channel groups
    use heterogeneous depth-, height-, and width-axis kernels.
    """

    def __init__(
        self,
        dim,
        ca_num_heads=1,
        sa_num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ca_attention=1,
        expand_ratio=1,
        init_cfg=None,
    ):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)

        for i in range(self.ca_num_heads):
            local_conv_1 = nn.Conv3d(
                dim // self.ca_num_heads,
                dim // self.ca_num_heads,
                kernel_size=(3 + i * 2, 1, 1),
                padding=(1 + i, 0, 0),
                stride=1,
                groups=dim // self.ca_num_heads,
            )
            local_conv_2 = nn.Conv3d(
                dim // self.ca_num_heads,
                dim // self.ca_num_heads,
                kernel_size=(1, 3 + i * 2, 1),
                padding=(0, 1 + i, 0),
                stride=1,
                groups=dim // self.ca_num_heads,
            )
            local_conv_3 = nn.Conv3d(
                dim // self.ca_num_heads,
                dim // self.ca_num_heads,
                kernel_size=(1, 1, 3 + i * 2),
                padding=(0, 0, 1 + i),
                stride=1,
                groups=dim // self.ca_num_heads,
            )
            setattr(self, f"local_conv_{i + 1}_1", local_conv_1)
            setattr(self, f"local_conv_{i + 1}_2", local_conv_2)
            setattr(self, f"local_conv_{i + 1}_3", local_conv_3)

        self.proj0 = nn.Conv3d(
            dim,
            dim * expand_ratio,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=self.split_groups,
        )

        self.norm = nn.InstanceNorm3d(dim * expand_ratio, affine=True)

        self.proj1 = nn.Conv3d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        self.dw_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape

        v = self.v(x)
        s = (
            self.s(x)
            .reshape(B, H, W, D, self.ca_num_heads, C // self.ca_num_heads)
            .permute(4, 0, 5, 1, 2, 3)
        )

        for i in range(self.ca_num_heads):
            local_conv_1 = getattr(self, f"local_conv_{i + 1}_1")
            local_conv_2 = getattr(self, f"local_conv_{i + 1}_2")
            local_conv_3 = getattr(self, f"local_conv_{i + 1}_3")

            s_i = s[i]
            s_i = local_conv_1(s_i)
            s_i = local_conv_2(s_i)
            s_i = local_conv_3(s_i).reshape(B, self.split_groups, -1, H, W, D)

            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)

        s_out = s_out.reshape(B, C, H, W, D)

        s_out = (
            self.proj1(self.act(self.norm(self.proj0(s_out))))
            .reshape(B, C, N)
            .permute(0, 2, 1)
        )

        x = s_out * v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x_out = self.act2(x)
        return x_out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out
class HAMAEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        sca_heads=4,
        sca_expand_ratio=1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.1)

        self.attn = HeterogeneousAxialMixingAttention(
            dim=out_channels,
            ca_num_heads=sca_heads,
            sa_num_heads=1,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            ca_attention=1,
            expand_ratio=sca_expand_ratio,
            init_cfg=None,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        B, C, D, H, W = x.shape
        x_tok = x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)

        y = self.attn(x_tok, D, H, W)
        y = y.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        return x + y
class SharedHierarchicalEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channel_num=16,
        use_sc: bool = True,
        sc_ks=(5, 5, 5, 5, 5),
        sc_k_local: int = 5,
        hama_heads=(2, 2, 4, 4),
        hama_expand_ratio=1,
    ):
        super().__init__()

        _, _, h4, h5 = hama_heads

        self.conv_1 = DConvBlock(in_channels, channel_num)
        self.conv_2 = DConvBlock(channel_num, channel_num * 2)
        self.conv_3 = DConvBlock(channel_num * 2, channel_num * 4)

        self.conv_4 = HAMAEncoderBlock(
            channel_num * 4,
            channel_num * 8,
            sca_heads=h4,
            sca_expand_ratio=hama_expand_ratio,
        )
        self.conv_5 = HAMAEncoderBlock(
            channel_num * 8,
            channel_num * 16,
            sca_heads=h5,
            sca_expand_ratio=hama_expand_ratio,
        )

        self.downsample = nn.AvgPool3d(2, stride=2)

        self.use_sc = use_sc
        if use_sc:
            k1, k2, k3, _, _ = sc_ks
            self.sc_1 = TriAxialStripGating(channel_num, k=k1, k_local=sc_k_local)
            self.sc_2 = TriAxialStripGating(channel_num * 2, k=k2, k_local=sc_k_local)
            self.sc_3 = TriAxialStripGating(channel_num * 4, k=k3, k_local=sc_k_local)

            self.sc_4 = nn.Identity()
            self.sc_5 = nn.Identity()
        else:
            self.sc_1 = nn.Identity()
            self.sc_2 = nn.Identity()
            self.sc_3 = nn.Identity()
            self.sc_4 = nn.Identity()
            self.sc_5 = nn.Identity()

    def forward(self, x_in):
        x_1 = self.sc_1(self.conv_1(x_in))
        x = self.downsample(x_1)

        x_2 = self.sc_2(self.conv_2(x))
        x = self.downsample(x_2)

        x_3 = self.sc_3(self.conv_3(x))
        x = self.downsample(x_3)

        x_4 = self.sc_4(self.conv_4(x))
        x = self.downsample(x_4)

        x_5 = self.sc_5(self.conv_5(x))
        return [x_1, x_2, x_3, x_4, x_5]

def get_winsize(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)
def get_aff(xk, yk):
    b, n, c = xk.shape
    xk = xk.permute(0, 2, 1) #b, c, n
    yk = yk.permute(0, 2, 1) #b, c, n
    a_sq = xk.pow(2).sum(1).unsqueeze(2)
    ab = xk.transpose(1, 2) @ yk
    affinity = (2 * ab - a_sq) / math.sqrt(c)
    # softmax operation; aligned the evaluation style
    maxes = torch.max(affinity, dim=1, keepdim=True)[0]
    x_exp = torch.exp(affinity - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    affinity = x_exp / x_exp_sum
    affinity = affinity.permute(0, 2, 1)
    return affinity
class BidirectionalFeatureInteraction(nn.Module):
    """Bidirectional feature interaction used by ACM.

    Takes two feature maps (x, y) with shape (B, C, D, H, W) and returns gated
    versions (x', y'). The design follows ADSF's idea:
    global pooling -> channel interaction via lightweight conv1d -> bidirectional
    association -> learnable mixing factor.
    """
    def __init__(self, channel: int, m: float = -0.80, b: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([m], dtype=torch.float32), requires_grad=True)
        self.mix_block = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if (t % 2 == 1) else (t + 1)
        k = max(k, 1)

        # ECA-like channel interaction (1D conv over channels)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        # pointwise projection for the second branch (3D version of paper's 1x1 conv)
        self.fc = nn.Conv3d(channel, channel, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def _conv1d_over_c(self, v_bc: torch.Tensor) -> torch.Tensor:
        """v_bc: (B,C) -> (B,C)"""
        v = v_bc.unsqueeze(1)          # (B,1,C)
        v = self.conv1(v)              # (B,1,C)
        return v.squeeze(1)            # (B,C)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x, y: (B,C,D,H,W)
        B, C, _, _, _ = x.shape

        ax = self.avg_pool(x).view(B, C)          # (B,C)
        ay = self.avg_pool(y)                     # (B,C,1,1,1)

        ax = self._conv1d_over_c(ax).unsqueeze(-1)   # (B,C,1)
        ay = self.fc(ay).view(B, C).unsqueeze(-1)    # (B,C,1)

        # bidirectional association
        # out_xy: x-guided weights, out_yx: y-guided weights
        out_xy = torch.matmul(ax, ay.transpose(1, 2))    # (B,C,C)
        out_xy = self.sigmoid(out_xy.sum(dim=1))         # (B,C)

        out_yx = torch.matmul(ay, ax.transpose(1, 2))    # (B,C,C)
        out_yx = self.sigmoid(out_yx.sum(dim=1))         # (B,C)

        mix = self.mix_block(self.w)   # scalar in (0,1)
        # produce two (slightly) different gates for x and y (cross direction)
        g_y = out_xy * mix + out_yx * (1.0 - mix)
        g_x = out_yx * mix + out_xy * (1.0 - mix)

        # refine gates with conv1d-over-channel (keeps it lightweight)
        g_y = self.sigmoid(self._conv1d_over_c(g_y)).view(B, C, 1, 1, 1)
        g_x = self.sigmoid(self._conv1d_over_c(g_x)).view(B, C, 1, 1, 1)

        # residual-style modulation for stability
        return x * (1.0 + g_x), y * (1.0 + g_y)

class Sim(nn.Module):
    def __init__(self, dim, window_size=[2, 2, 2], plug=None):
        super().__init__()
        self.channel = window_size[0] * window_size[1] * window_size[2]
        self.window_size = window_size
        self.normx = nn.LayerNorm(dim)
        self.normy = nn.LayerNorm(dim)
        self.plug = plug
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in window_size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)
    def makeV(self, N):
        v = self.grid.reshape(self.channel, 3).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0)
        return v
    def forward(self, x_in, y_in):
        b, c, d, h, w = x_in.shape
        n = d*h*w
        x = x_in.permute(0, 2, 3, 4, 1)
        y = y_in.permute(0, 2, 3, 4, 1)
        x = self.normx(x)
        y = self.normy(y)

        # --- ADSF cross-gate plug (optional) ---
        if self.plug is not None:
            xb = x.permute(0, 4, 1, 2, 3).contiguous()  # B,C,D,H,W
            yb = y.permute(0, 4, 1, 2, 3).contiguous()
            try:
                xb, yb = self.plug(xb, yb)              # dual-input plug
            except TypeError:
                xb = self.plug(xb)                      # single-input plug
                yb = self.plug(yb)
            x = xb.permute(0, 2, 3, 4, 1).contiguous()  # B,D,H,W,C
            y = yb.permute(0, 2, 3, 4, 1).contiguous()
        # --------------------------------------
        window_size = get_winsize((d, h, w), self.window_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)
        affinity = get_aff(x_windows, y_windows) # b_, n_, c_
        affinity = affinity.view(-1, *(window_size + (self.channel,)))
        affinity = window_reverse(affinity, window_size, dims)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            affinity = affinity[:, :d, :h, :w, :].contiguous()
        affinity = affinity.view(b, n, self.channel).reshape(b, n, self.channel, 1).transpose(2, 3) # b, n, 1, 8
        v = self.makeV(n)  # b, n, 8, 3
        out = (affinity @ v)  # b, n, 1, 3
        out = out.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3)
        #out = out.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
        return out

class RegHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.reg_head = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    def forward(self, x):
        x_out = self.reg_head(x)
        return x_out

def _grad_l2(flow: torch.Tensor) -> torch.Tensor:
    """Simple smoothness: mean squared spatial flow gradient."""
    dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()


def _check(name: str, t: torch.Tensor):
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    if torch.isnan(t).any():
        raise RuntimeError(f"{name} has NaN")
    if torch.isinf(t).any():
        raise RuntimeError(f"{name} has Inf")


def _cubic_bspline_weights(u: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline basis weights for u in [0,1].
    Returns (...,4) for B0..B3.
    """
    # u: (...)
    u2 = u * u
    u3 = u2 * u
    w0 = ((1 - u) ** 3) / 6.0
    w1 = (3 * u3 - 6 * u2 + 4) / 6.0
    w2 = (-3 * u3 + 3 * u2 + 3 * u + 1) / 6.0
    w3 = u3 / 6.0
    return torch.stack([w0, w1, w2, w3], dim=-1)


class AdaptiveCorrelationMatching(Sim):
    """Adaptive Correlation Matching (ACM).

    - corr: expected offset (B,3,D,H,W) same as Sim
    - conf: (B,1,D,H,W) in [0,1] (roughly)

    Confidence is derived from inverse normalized entropy by default. The
    maximum-probability variant is retained for the reliability ablation.
    """

    def __init__(self, dim, window_size=[2, 2, 2], conf_type: str = 'entropy', plug=None):
        super().__init__(dim=dim, window_size=window_size, plug=plug)
        assert conf_type in ('max', 'entropy')
        self.conf_type = conf_type

    def forward(self, x_in, y_in):
        b, c, d, h, w = x_in.shape
        n = d * h * w

        x = x_in.permute(0, 2, 3, 4, 1)
        y = y_in.permute(0, 2, 3, 4, 1)
        x = self.normx(x)
        y = self.normy(y)

        # --- ADSF cross-gate plug (optional) ---
        if self.plug is not None:
            xb = x.permute(0, 4, 1, 2, 3).contiguous()  # B,C,D,H,W
            yb = y.permute(0, 4, 1, 2, 3).contiguous()
            try:
                xb, yb = self.plug(xb, yb)              # dual-input plug
            except TypeError:
                xb = self.plug(xb)                      # single-input plug
                yb = self.plug(yb)
            x = xb.permute(0, 2, 3, 4, 1).contiguous()  # B,D,H,W,C
            y = yb.permute(0, 2, 3, 4, 1).contiguous()
        # --------------------------------------

        window_size = get_winsize((d, h, w), self.window_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]

        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)

        affinity = get_aff(x_windows, y_windows)  # (Bwin, winN, C)
        affinity = affinity.view(-1, *(window_size + (self.channel,)))
        affinity = window_reverse(affinity, window_size, dims)  # (B, Dp, Hp, Wp, C)

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            affinity = affinity[:, :d, :h, :w, :].contiguous()

        # (B, N, C)
        aff = affinity.view(b, n, self.channel)

        # confidence
        if self.conf_type == 'max':
            conf = aff.max(dim=-1).values  # (B,N)
        else:
            eps = 1e-8
            p = aff.clamp_min(eps)
            ent = -(p * p.log()).sum(dim=-1)  # (B,N)
            ent_norm = ent / math.log(self.channel)
            conf = (1.0 - ent_norm).clamp(0.0, 1.0)

        # expected offset
        aff_m = aff.reshape(b, n, self.channel, 1).transpose(2, 3)  # (B,N,1,C)
        v = self.makeV(n).to(aff_m.device)  # (1,N,C,3)
        out = (aff_m @ v)  # (B,N,1,3)
        out = out.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3)

        conf = conf.reshape(b, 1, d, h, w)
        return out, conf


class ConfidenceGuidedBFFD(nn.Module):
    """Confidence-guided B-spline FFD solver using conjugate gradients.

    Solve control coefficients c (on a control lattice) from dense/sampled displacement observations b:
        argmin_c || W^(1/2) (A c - b) ||^2 + lam * ||c||^2
    where A is cubic B-spline interpolation operator.

    Notes:
    - Uses Conjugate Gradient on normal equations without explicitly forming (A^T W A).
    - Designed to be drop-in inside the model forward (no training loop change).
    """

    def __init__(
        self,
        ctrl_grid: Union[int, Tuple[int, int, int]] = 24,
        max_points: int = 32768,
        sample_stride: Union[int, Tuple[int, int, int]] = 2,
        sample_by_conf: bool = True,
        lam: float = 1e-2,
        cg_iters: int = 20,
        cg_tol: float = 1e-6,
        conf_power: float = 1.0,
        force_fp32: bool = True,
        eval_chunk_z: int = 16,
    ):
        super().__init__()
        if isinstance(ctrl_grid, int):
            ctrl_grid = (ctrl_grid, ctrl_grid, ctrl_grid)
        if isinstance(sample_stride, int):
            sample_stride = (sample_stride, sample_stride, sample_stride)

        self.ctrl_grid = tuple(int(x) for x in ctrl_grid)
        self.max_points = int(max_points)
        self.sample_stride = tuple(int(x) for x in sample_stride)
        self.sample_by_conf = bool(sample_by_conf)
        self.lam = float(lam)
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)
        self.conf_power = float(conf_power)
        self.force_fp32 = bool(force_fp32)
        self.eval_chunk_z = int(eval_chunk_z)

    @staticmethod
    def _build_ctrl_params(size: int, n_ctrl_target: int, device, dtype):
        """Return n_ctrl (>=5) and spacing so that base=floor(x/spacing) ranges safely.

        We use neighbor indices: base + {0,1,2,3}, so we need n_ctrl >= base_max + 4.
        Choose n_ctrl <= size+3 (practical cap) and >=5.
        Spacing is set as (size-1)/(n_ctrl-4).
        """
        if size <= 1:
            n_ctrl = 5
            spacing = torch.tensor(1.0, device=device, dtype=dtype)
            return n_ctrl, spacing

        n_ctrl = int(min(n_ctrl_target, size + 3))
        n_ctrl = max(n_ctrl, 5)
        denom = max(n_ctrl - 4, 1)
        spacing = torch.tensor((size - 1) / denom, device=device, dtype=dtype)
        return n_ctrl, spacing

    @staticmethod
    def _axis_idx_w(size_out: int, spacing: torch.Tensor, n_ctrl: int, device, dtype):
        coord = torch.arange(size_out, device=device, dtype=dtype)
        t = coord / spacing
        base = torch.floor(t).to(torch.long)
        u = (t - base.to(dtype)).clamp(0.0, 1.0)
        w = _cubic_bspline_weights(u)  # (size_out,4)
        offsets = torch.arange(4, device=device, dtype=torch.long)
        idx = base[:, None] + offsets[None, :]
        idx = idx.clamp(0, n_ctrl - 1)
        return idx, w

    def _sample_linear_indices(self, conf: torch.Tensor, shape: Tuple[int, int, int]):
        """Return linear indices into flattened DHW, shared across batch.

        conf: (B,1,D,H,W)
        """
        B, _, D, H, W = conf.shape
        N = D * H * W
        dz, dy, dx = self.sample_stride

        # stride-based candidate set
        zz = torch.arange(0, D, dz, device=conf.device)
        yy = torch.arange(0, H, dy, device=conf.device)
        xx = torch.arange(0, W, dx, device=conf.device)
        Z, Y, X = torch.meshgrid(zz, yy, xx, indexing='ij')
        lin = (Z * (H * W) + Y * W + X).reshape(-1)

        if lin.numel() <= self.max_points:
            return lin

        # too many -> subsample
        if self.sample_by_conf:
            conf_flat = conf.reshape(B, N)
            conf_mean = conf_flat.mean(dim=0)
            # get conf on candidate set only
            w = conf_mean[lin].clamp_min(1e-6)
            w = w / w.sum()
            idx = torch.multinomial(w, self.max_points, replacement=False)
            return lin[idx]
        else:
            perm = torch.randperm(lin.numel(), device=conf.device)
            return lin[perm[: self.max_points]]

    def _coords_from_linear(self, lin: torch.Tensor, shape: Tuple[int, int, int], device, dtype):
        D, H, W = shape
        z = lin // (H * W)
        rem = lin % (H * W)
        y = rem // W
        x = rem % W
        coords = torch.stack([z, y, x], dim=-1).to(device=device, dtype=dtype)
        return coords  # (Ns,3) in z,y,x

    def _neighbors_idx_w(
        self,
        coords_zyx: torch.Tensor,
        ctrl_shape: Tuple[int, int, int],
        spacing_zyx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """For each sample coordinate, return 64 neighbor ctrl indices + weights.

        coords_zyx: (Ns,3) float
        returns:
          idx64: (Ns,64) long in [0,M)
          w64:   (Ns,64) float
        """
        device = coords_zyx.device
        dtype = coords_zyx.dtype
        Ns = coords_zyx.shape[0]

        Dc, Hc, Wc = ctrl_shape
        sz, sy, sx = spacing_zyx

        z = coords_zyx[:, 0]
        y = coords_zyx[:, 1]
        x = coords_zyx[:, 2]

        # base indices
        tz = z / sz
        ty = y / sy
        tx = x / sx

        bz = torch.floor(tz).to(torch.long)
        by = torch.floor(ty).to(torch.long)
        bx = torch.floor(tx).to(torch.long)

        uz = (tz - bz.to(dtype)).clamp(0.0, 1.0)
        uy = (ty - by.to(dtype)).clamp(0.0, 1.0)
        ux = (tx - bx.to(dtype)).clamp(0.0, 1.0)

        wz = _cubic_bspline_weights(uz)  # (Ns,4)
        wy = _cubic_bspline_weights(uy)
        wx = _cubic_bspline_weights(ux)

        off = torch.arange(4, device=device, dtype=torch.long)
        iz = (bz[:, None] + off[None, :]).clamp(0, Dc - 1)  # (Ns,4)
        iy = (by[:, None] + off[None, :]).clamp(0, Hc - 1)
        ix = (bx[:, None] + off[None, :]).clamp(0, Wc - 1)

        # combine to 64
        iz3 = iz[:, :, None, None]
        iy3 = iy[:, None, :, None]
        ix3 = ix[:, None, None, :]

        idx = (iz3 * (Hc * Wc) + iy3 * Wc + ix3).reshape(Ns, 64)

        w = (wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]).reshape(Ns, 64)
        return idx, w
    def _apply_A(self, c: torch.Tensor, idx64: torch.Tensor, w64: torch.Tensor) -> torch.Tensor:
        """Compute Ac at sampled points.

        c: (B,3,M)
        idx64: (Ns,64)
        w64: (Ns,64)
        returns: (B,3,Ns)
        """
        B, C, M = c.shape
        Ns = idx64.shape[0]

        # Flatten 64 basis contributions per sample point to satisfy gather/index_select shape rules.
        idx_flat = idx64.reshape(-1).long()  # (Ns*64,)
        gathered = torch.index_select(c, 2, idx_flat)  # (B,3,Ns*64)
        gathered = gathered.reshape(B, C, Ns, 64)      # (B,3,Ns,64)

        pred = (gathered * w64[None, None, :, :]).sum(dim=-1)  # (B,3,Ns)
        return pred
    def _apply_At(self, r: torch.Tensor, idx64: torch.Tensor, w64: torch.Tensor, M: int) -> torch.Tensor:
        """Compute A^T r for sampled points.

        r: (B,3,Ns)
        returns: (B,3,M)
        """
        B, C, Ns = r.shape
        out = torch.zeros((B, C, M), device=r.device, dtype=r.dtype)

        idx = idx64[None, None, :, :].expand(B, C, Ns, 64).reshape(B, C, Ns * 64)
        src = (r[:, :, :, None] * w64[None, None, :, :]).reshape(B, C, Ns * 64)
        out.scatter_add_(dim=2, index=idx, src=src)
        return out

    def _matvec(self, x: torch.Tensor, idx64: torch.Tensor, w64: torch.Tensor, W: torch.Tensor, M: int) -> torch.Tensor:
        """Compute (A^T W A + lam I) x.

        x: (B,3,M)
        W: (B,Ns)
        """
        Ax = self._apply_A(x, idx64, w64)  # (B,3,Ns)
        Ax = Ax * W[:, None, :]
        AtWAx = self._apply_At(Ax, idx64, w64, M)
        return AtWAx + self.lam * x

    def _cg_solve(self, rhs: torch.Tensor, idx64: torch.Tensor, w64: torch.Tensor, W: torch.Tensor, M: int) -> torch.Tensor:
        """Batched Conjugate Gradient for SPD system."""
        B = rhs.shape[0]
        x = torch.zeros_like(rhs)
        r = rhs - self._matvec(x, idx64, w64, W, M)  # = rhs
        p = r

        rr = (r * r).sum(dim=(1, 2)).clamp_min(1e-12)  # (B,)
        rhs_norm = rr.sqrt().clamp_min(1e-12)

        for _ in range(self.cg_iters):
            Ap = self._matvec(p, idx64, w64, W, M)
            pAp = (p * Ap).sum(dim=(1, 2)).clamp_min(1e-12)
            alpha = (rr / pAp).view(B, 1, 1)
            x = x + alpha * p
            r = r - alpha * Ap

            rr_new = (r * r).sum(dim=(1, 2)).clamp_min(1e-12)
            if (rr_new.sqrt() / rhs_norm).max().item() < self.cg_tol:
                break

            beta = (rr_new / rr).view(B, 1, 1)
            p = r + beta * p
            rr = rr_new
        return x

    def _evaluate_dense(self, ctrl: torch.Tensor, out_shape: Tuple[int, int, int], spacing_zyx, ctrl_shape):
        """Separable cubic B-spline interpolation from control lattice to dense field.

        ctrl: (B,3,Dc,Hc,Wc)
        returns: (B,3,D,H,W)
        """
        B, C, Dc, Hc, Wc = ctrl.shape
        D, H, W = out_shape
        device = ctrl.device
        dtype = ctrl.dtype

        sz, sy, sx = spacing_zyx

        # precompute y/x indices+weights (independent of z)
        iy, wy = self._axis_idx_w(H, sy, Hc, device, dtype)  # (H,4)
        ix, wx = self._axis_idx_w(W, sx, Wc, device, dtype)

        chunks: List[torch.Tensor] = []
        z0 = 0
        while z0 < D:
            z1 = min(z0 + self.eval_chunk_z, D)
            Z = z1 - z0
            iz, wz = self._axis_idx_w(Z, sz, Dc, device, dtype)
            # shift coords by z0
            # Instead of recomputing with absolute coords, we recompute correctly by using absolute z coords.
            z_coord = torch.arange(z0, z1, device=device, dtype=dtype)
            t = z_coord / sz
            base = torch.floor(t).to(torch.long)
            u = (t - base.to(dtype)).clamp(0.0, 1.0)
            wz = _cubic_bspline_weights(u)
            offsets = torch.arange(4, device=device, dtype=torch.long)
            iz = (base[:, None] + offsets[None, :]).clamp(0, Dc - 1)

            # along z: (B,3,Z,Hc,Wc)
            tmp = 0.0
            for l in range(4):
                tmp = tmp + wz[:, l].view(1, 1, Z, 1, 1) * ctrl[:, :, iz[:, l], :, :]

            # along y: (B,3,Z,H,Wc)
            tmp2 = 0.0
            for l in range(4):
                tmp2 = tmp2 + wy[:, l].view(1, 1, 1, H, 1) * tmp[:, :, :, iy[:, l], :]

            # along x: (B,3,Z,H,W)
            out = 0.0
            for l in range(4):
                out = out + wx[:, l].view(1, 1, 1, 1, W) * tmp2[:, :, :, :, ix[:, l]]

            chunks.append(out)
            z0 = z1

        return torch.cat(chunks, dim=2)

    def forward(self, obs_flow: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve and return dense flow with same shape as obs_flow."""
        assert obs_flow.dim() == 5 and obs_flow.shape[1] == 3
        B, _, D, H, W = obs_flow.shape
        device = obs_flow.device

        if conf is None:
            conf = torch.ones((B, 1, D, H, W), device=device, dtype=obs_flow.dtype)
        else:
            assert conf.shape[:2] == (B, 1)

        # weights
        conf_w = conf.clamp(0.0, 1.0) ** self.conf_power

        # numeric stability: solve in fp32
        if self.force_fp32:
            solve_dtype = torch.float32
        else:
            solve_dtype = obs_flow.dtype

        # numeric stability: run solver in fp32 and disable AMP (torch.amp is the newer API)
        try:
            from torch.amp import autocast  # PyTorch >= 2.0
            _autocast_ctx = autocast(device_type='cuda', enabled=False)
        except Exception:
            _autocast_ctx = torch.cuda.amp.autocast(enabled=False)

        with _autocast_ctx:
            b = obs_flow.to(dtype=solve_dtype)
            w = conf_w.to(dtype=solve_dtype)

            # control lattice parameters
            ncz, sz = self._build_ctrl_params(D, self.ctrl_grid[0], device, solve_dtype)
            ncy, sy = self._build_ctrl_params(H, self.ctrl_grid[1], device, solve_dtype)
            ncx, sx = self._build_ctrl_params(W, self.ctrl_grid[2], device, solve_dtype)
            ctrl_shape = (ncz, ncy, ncx)
            M = ncz * ncy * ncx

            # sample points
            lin = self._sample_linear_indices(w, (D, H, W))  # (Ns,)
            coords = self._coords_from_linear(lin, (D, H, W), device, solve_dtype)
            idx64, w64 = self._neighbors_idx_w(coords, ctrl_shape, (sz, sy, sx))

            # gather observations and weights at samples
            N = D * H * W
            b_flat = b.reshape(B, 3, N)
            w_flat = w.reshape(B, N)
            b_s = b_flat[:, :, lin]  # (B,3,Ns)
            W_s = w_flat[:, lin]  # (B,Ns)

            # rhs = A^T (W b)
            rhs = self._apply_At(b_s * W_s[:, None, :], idx64, w64, M)

            # solve
            c = self._cg_solve(rhs, idx64, w64, W_s, M)

            ctrl = c.view(B, 3, ncz, ncy, ncx)
            dense = self._evaluate_dense(ctrl, (D, H, W), (sz, sy, sx), ctrl_shape)

        return dense.to(dtype=obs_flow.dtype)


class ACMNet(nn.Module):
    """Adaptive Correlation Matching Network (ACM-Net).

    The default return value is ``(warped_moving, displacement)`` for backward
    compatibility with the original training scripts. Set ``return_aux=True``
    to additionally obtain the multi-scale correspondence and confidence maps.
    """

    def __init__(
        self,
        in_channel: int = 16,
        channel_num: int = 16,
        # solver hyperparams (defaults are conservative)
        ctrl_grids: Tuple[Union[int, Tuple[int,int,int]], ...] = (10, 12, 16, 20, 24),
        max_points: Tuple[int, ...] = (4096, 8192, 16384, 32768, 32768),
        lam: float = 1e-2,
        cg_iters: int = 20,
        cg_tol: float = 1e-6,
        conf_power: float = 1.0,
        conf_type: str = 'entropy',
        matching_window: Tuple[int, int, int] = (2, 2, 2),
        sample_stride: Tuple[int, int, int] = (2, 2, 2),
        sample_by_conf: bool = True,
    ):
        super().__init__()

        self.encoder = SharedHierarchicalEncoder(
            channel_num=in_channel,
            hama_heads=(2, 2, 4, 4),
            hama_expand_ratio=1,
        )

        self.conv_1 = DConvBlock(in_channel * 1 * 2 + channel_num * 1 + 3, channel_num * 1)
        self.conv_2 = DConvBlock(in_channel * 2 * 2 + channel_num * 2 + 3, channel_num * 2)
        self.conv_3 = DConvBlock(in_channel * 4 * 2 + channel_num * 4 + 3, channel_num * 4)
        self.conv_4 = DConvBlock(in_channel * 8 * 2 + channel_num * 8 + 3, channel_num * 8)
        self.conv_5 = DConvBlock(in_channel * 16 * 2 + 3, channel_num * 16)

        self.corr_1 = AdaptiveCorrelationMatching(in_channel * 1, window_size=matching_window, conf_type=conf_type, plug=BidirectionalFeatureInteraction(in_channel * 1))
        self.corr_2 = AdaptiveCorrelationMatching(in_channel * 2, window_size=matching_window, conf_type=conf_type, plug=BidirectionalFeatureInteraction(in_channel * 2))
        self.corr_3 = AdaptiveCorrelationMatching(in_channel * 4, window_size=matching_window, conf_type=conf_type, plug=BidirectionalFeatureInteraction(in_channel * 4))
        self.corr_4 = AdaptiveCorrelationMatching(in_channel * 8, window_size=matching_window, conf_type=conf_type, plug=BidirectionalFeatureInteraction(in_channel * 8))
        self.corr_5 = AdaptiveCorrelationMatching(in_channel * 16, window_size=matching_window, conf_type=conf_type, plug=BidirectionalFeatureInteraction(in_channel * 16))

        # upsample conv features
        self.upsample_1 = DeconvBlock(channel_num * 2, channel_num * 1)
        self.upsample_2 = DeconvBlock(channel_num * 4, channel_num * 2)
        self.upsample_3 = DeconvBlock(channel_num * 8, channel_num * 4)
        self.upsample_4 = DeconvBlock(channel_num * 16, channel_num * 8)

        # observation heads (used as solver 'measurements')
        self.reghead_1 = RegHead(channel_num * 1)
        self.reghead_2 = RegHead(channel_num * 2)
        self.reghead_3 = RegHead(channel_num * 4)
        self.reghead_4 = RegHead(channel_num * 8)
        self.reghead_5 = RegHead(channel_num * 16)

        # residual heads (dense residual flow); initialized near-zero by default
        self.reshead_1 = RegHead(channel_num * 1)
        self.reshead_2 = RegHead(channel_num * 2)
        self.reshead_3 = RegHead(channel_num * 4)
        self.reshead_4 = RegHead(channel_num * 8)
        self.reshead_5 = RegHead(channel_num * 16)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

        # start close to pure-solver behavior
        for m in [self.reshead_1, self.reshead_2, self.reshead_3, self.reshead_4, self.reshead_5]:
            for p in m.parameters():
                p.data.mul_(0.0)

        # solvers per level (5->1)
        # ctrl_grids order: (L5, L4, L3, L2, L1)
        assert len(ctrl_grids) == 5 and len(max_points) == 5
        solver_kwargs = {
            "lam": lam,
            "cg_iters": cg_iters,
            "cg_tol": cg_tol,
            "conf_power": conf_power,
            "sample_stride": sample_stride,
            "sample_by_conf": sample_by_conf,
        }
        self.solver_5 = ConfidenceGuidedBFFD(ctrl_grid=ctrl_grids[0], max_points=max_points[0], **solver_kwargs)
        self.solver_4 = ConfidenceGuidedBFFD(ctrl_grid=ctrl_grids[1], max_points=max_points[1], **solver_kwargs)
        self.solver_3 = ConfidenceGuidedBFFD(ctrl_grid=ctrl_grids[2], max_points=max_points[2], **solver_kwargs)
        self.solver_2 = ConfidenceGuidedBFFD(ctrl_grid=ctrl_grids[3], max_points=max_points[3], **solver_kwargs)
        self.solver_1 = ConfidenceGuidedBFFD(ctrl_grid=ctrl_grids[4], max_points=max_points[4], **solver_kwargs)

        self.resize_transformer = nn.ModuleList()
        self.spatial_transformer = nn.ModuleList()
        for i in range(5):
            self.resize_transformer.append(ResizeTransformer_block(resize_factor=2, mode='trilinear'))
            self.spatial_transformer.append(SpatialTransformer_block(mode='bilinear'))

    def forward(self, moving, fixed, return_aux: bool = False):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)

        # Level 5 (coarsest)
        corr_5, conf_5 = self.corr_5(x_mov_5, x_fix_5)
        cat = torch.cat([x_mov_5, corr_5, x_fix_5], dim=1)
        conv_corr_5 = self.conv_5(cat)
        obs_5 = self.reghead_5(conv_corr_5)
        flow_solve_5 = self.solver_5(obs_5, conf_5)
        res_5 = self.reshead_5(conv_corr_5) * self.res_scale
        flow_5 = flow_solve_5 + res_5

        # Level 4
        flow_5_up = self.resize_transformer[3](flow_5)
        x_mov_4 = self.spatial_transformer[3](x_mov_4, flow_5_up)
        conv_corr_5_up = self.upsample_4(conv_corr_5)

        corr_4, conf_4 = self.corr_4(x_mov_4, x_fix_4)
        cat = torch.cat([x_mov_4, corr_4, x_fix_4, conv_corr_5_up], dim=1)
        conv_corr_4 = self.conv_4(cat)
        obs_4 = self.reghead_4(conv_corr_4)
        flow_solve_4 = self.solver_4(obs_4, conf_4)
        res_4 = self.reshead_4(conv_corr_4) * self.res_scale
        delta_4 = flow_solve_4 + res_4
        flow_4 = delta_4 + flow_5_up

        # Level 3
        flow_4_up = self.resize_transformer[2](flow_4)
        x_mov_3 = self.spatial_transformer[2](x_mov_3, flow_4_up)
        conv_corr_4_up = self.upsample_3(conv_corr_4)

        corr_3, conf_3 = self.corr_3(x_mov_3, x_fix_3)
        cat = torch.cat([x_mov_3, corr_3, x_fix_3, conv_corr_4_up], dim=1)
        conv_corr_3 = self.conv_3(cat)
        obs_3 = self.reghead_3(conv_corr_3)
        flow_solve_3 = self.solver_3(obs_3, conf_3)
        res_3 = self.reshead_3(conv_corr_3) * self.res_scale
        delta_3 = flow_solve_3 + res_3
        flow_3 = delta_3 + flow_4_up

        # Level 2
        flow_3_up = self.resize_transformer[1](flow_3)
        x_mov_2 = self.spatial_transformer[1](x_mov_2, flow_3_up)
        conv_corr_3_up = self.upsample_2(conv_corr_3)

        corr_2, conf_2 = self.corr_2(x_mov_2, x_fix_2)
        cat = torch.cat([x_mov_2, corr_2, x_fix_2, conv_corr_3_up], dim=1)
        conv_corr_2 = self.conv_2(cat)
        obs_2 = self.reghead_2(conv_corr_2)
        flow_solve_2 = self.solver_2(obs_2, conf_2)
        res_2 = self.reshead_2(conv_corr_2) * self.res_scale
        delta_2 = flow_solve_2 + res_2
        flow_2 = delta_2 + flow_3_up

        # Level 1 (finest)
        flow_2_up = self.resize_transformer[0](flow_2)
        x_mov_1 = self.spatial_transformer[0](x_mov_1, flow_2_up)
        conv_corr_2_up = self.upsample_1(conv_corr_2)

        corr_1, conf_1 = self.corr_1(x_mov_1, x_fix_1)
        cat = torch.cat([x_mov_1, corr_1, x_fix_1, conv_corr_2_up], dim=1)
        conv_corr_1 = self.conv_1(cat)
        obs_1 = self.reghead_1(conv_corr_1)
        flow_solve_1 = self.solver_1(obs_1, conf_1)
        res_1 = self.reshead_1(conv_corr_1) * self.res_scale
        delta_1 = flow_solve_1 + res_1
        flow_1 = delta_1 + flow_2_up

        moved = self.spatial_transformer[0](moving, flow_1)
        if return_aux:
            aux: Dict[str, List[torch.Tensor]] = {
                "soft_correspondence": [corr_1, corr_2, corr_3, corr_4, corr_5],
                "confidence": [conf_1, conf_2, conf_3, conf_4, conf_5],
                "bffd_flow": [flow_solve_1, flow_solve_2, flow_solve_3, flow_solve_4, flow_solve_5],
                "learned_residual": [res_1, res_2, res_3, res_4, res_5],
            }
            return moved, flow_1, aux
        return moved, flow_1


# Backward-compatible aliases. They preserve imports used by the original
# experiment scripts while the public API follows the terminology in the paper.
StripBlock3D = TriAxialStripGating
ScaAttention = HeterogeneousAxialMixingAttention
DualConvBlock = DConvBlock
ScaEncoderBlock = HAMAEncoderBlock
SharedEncoder = SharedHierarchicalEncoder
ADSFCrossGate3D = BidirectionalFeatureInteraction
SimWithConf = AdaptiveCorrelationMatching
BSplineFFDLeastSquaresCG = ConfidenceGuidedBFFD
SLNet_BSplineSolve = ACMNet

__all__ = [
    "ACMNet",
    "TriAxialStripGating",
    "HeterogeneousAxialMixingAttention",
    "AdaptiveCorrelationMatching",
    "BidirectionalFeatureInteraction",
    "ConfidenceGuidedBFFD",
    "SLNet_BSplineSolve",
]

def _grad_l2(flow: torch.Tensor) -> torch.Tensor:
    dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()


def _check(name: str, t: torch.Tensor):
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    if torch.isnan(t).any():
        raise RuntimeError(f"{name} has NaN")
    if torch.isinf(t).any():
        raise RuntimeError(f"{name} has Inf")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    B, C, D, H, W = 1, 1, 32, 64, 64
    moving = torch.randn(B, C, D, H, W, device=device)
    fixed = torch.randn(B, C, D, H, W, device=device)

    model = ACMNet(in_channel=8, channel_num=8).to(device)

    model.eval()
    with torch.no_grad():
        moved, flow = model(moving, fixed)

    _check("moved", moved)
    _check("flow", flow)
    assert moved.shape == moving.shape
    assert flow.shape == (B, 3, D, H, W)
    print("Forward OK")

    model.train()
    moved, flow = model(moving, fixed)
    loss_img = nnf.mse_loss(moved, fixed)
    loss_smooth = _grad_l2(flow)
    loss = loss_img + 0.01 * loss_smooth
    model.zero_grad(set_to_none=True)
    loss.backward()
    print(f"Backward OK: loss={loss.item():.6f}")
