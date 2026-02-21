# ============================================================
# PyTorch implementation: Hierarchical Multiscale CNN (HMCN)
# - Input x: (B, C, H, W)
# - Pad channels so C' divisible by s
# - Duplicate into Left/Right wings
# - Each wing splits into s channel chunks: ϕ1..ϕs
# - Hierarchical depthwise-separable conv with increasing kernels
# - After each stage: split output into two halves:
#     - half goes to wing output
#     - half is carried and concatenated with next ϕ (⊕) for next stage
# - Skip connection per wing
# - Concatenate features from both wings
# - Lightweight attention (SE) at the end
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Depthwise Separable Conv2D
# -----------------------------
class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise: groups=in_channels
    Pointwise: 1x1 to mix channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            # "same-ish" padding for odd kernels
            padding = ((kernel_size - 1) // 2) * dilation

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# -----------------------------
# Simple SE Attention (lightweight)
# -----------------------------
class SEBlock(nn.Module):
    """
    Channel attention: squeeze (GAP) -> MLP -> scale.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


# -----------------------------
# One Wing: Left or Right
# -----------------------------
@dataclass
class WingConfig:
    s: int                      # number of splits/stages
    K: int                      # input-to-output channel ratio (larger K => fewer output channels)
    kernels: Optional[List[int]] = None  # kernel sizes per stage (len=s). If None => 3,5,7,...
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1
    keep_last_carry: bool = True  # include the final carry into outputs to avoid dropping features


class HMCNWing(nn.Module):
    """
    Implements equations like:
      Left:
        Y_l = S(R(BN(Conv_dwp(ϕ_l)))), l=1
        Y_l = S(R(BN(Conv_dwp(Y_{l-1} ⊕ ϕ_l)))), 1<l<=s
      Right (mirrored order):
        Y_l = S(R(BN(Conv_dwp(ϕ_l)))), l=s
        Y_l = S(R(BN(Conv_dwp(Y_{l+1} ⊕ ϕ_l)))), 1<=l<s

    Here S(.) is "split into two halves along channels":
      out_direct, out_carry = split(out)
      out_direct -> collected output
      out_carry -> concatenated with next input chunk
    """
    def __init__(self, chunk_channels: int, cfg: WingConfig):
        super().__init__()
        assert cfg.s >= 1
        assert cfg.K >= 1

        self.cfg = cfg
        self.chunk_channels = chunk_channels

        # Output width per stage (w) controlled by K
        # Larger K => smaller w
        self.w = max(1, chunk_channels // cfg.K)

        # kernel schedule
        if cfg.kernels is None:
            kernels = [2 * i + 1 for i in range(1, cfg.s + 1)]  # 3,5,7,...
        else:
            assert len(cfg.kernels) == cfg.s
            kernels = cfg.kernels

        self.kernels = kernels

        # Each stage gets input channels either:
        # stage 0: chunk_channels
        # stage l>0: chunk_channels + carry_channels
        # carry_channels is w//2 (because we split w into direct/carry)
        carry_ch = max(1, self.w // 2)

        self.stages = nn.ModuleList()
        for l in range(cfg.s):
            in_ch = chunk_channels if l == 0 else (chunk_channels + carry_ch)
            stage = nn.Sequential(
                DepthwiseSeparableConv2d(in_ch, self.w, kernel_size=kernels[l], bias=True),
                nn.BatchNorm2d(self.w, eps=cfg.bn_eps, momentum=cfg.bn_momentum),
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)

        # per-wing skip: project input (chunk_channels*s) -> match concat output channels
        # output channels per wing = s * direct_half + (optional) last_carry
        direct_ch = self.w - carry_ch
        out_ch = cfg.s * direct_ch + (carry_ch if cfg.keep_last_carry else 0)

        self.skip_proj = nn.Sequential(
            nn.Conv2d(chunk_channels * cfg.s, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch, eps=cfg.bn_eps, momentum=cfg.bn_momentum),
        )

        self.out_channels = out_ch
        self.carry_channels = carry_ch
        self.direct_channels = direct_ch

    def _split_direct_carry(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # split channels into (direct, carry)
        c = x.shape[1]
        carry_ch = self.carry_channels
        direct_ch = c - carry_ch
        if direct_ch <= 0:
            # fallback: if w==1, keep direct=1 and carry=0
            return x, x[:, :0]
        direct = x[:, :direct_ch, :, :]
        carry = x[:, direct_ch:direct_ch + carry_ch, :, :]
        return direct, carry

    def forward(self, chunks: List[torch.Tensor], direction: str) -> torch.Tensor:
        """
        chunks: list length s, each (B, chunk_channels, H, W)
        direction: "left" or "right"
        """
        assert len(chunks) == self.cfg.s
        assert direction in ("left", "right")

        # order
        idxs = list(range(self.cfg.s))
        if direction == "right":
            idxs = list(reversed(idxs))

        outs: List[torch.Tensor] = []
        carry: Optional[torch.Tensor] = None

        for stage_i, chunk_i in enumerate(idxs):
            phi = chunks[chunk_i]  # (B, chunk_channels, H, W)

            if carry is None:
                inp = phi
            else:
                inp = torch.cat([carry, phi], dim=1)  # (B, carry+chunk_channels, H, W)

            y = self.stages[stage_i](inp)  # (B, w, H, W)
            direct, carry = self._split_direct_carry(y)
            outs.append(direct)

        if self.cfg.keep_last_carry and carry is not None and carry.shape[1] > 0:
            outs.append(carry)

        y_wing = torch.cat(outs, dim=1)  # (B, out_ch, H, W)

        # skip connection
        x_full = torch.cat(chunks, dim=1)  # (B, chunk_channels*s, H, W)
        y_wing = y_wing + self.skip_proj(x_full)
        return F.relu(y_wing, inplace=True)


# -----------------------------
# Full HMCN Block (Left + Right + Attention)
# -----------------------------
class HMCNBlock(nn.Module):
    """
    x: (B, C, H, W)
    - pad C -> C' divisible by s
    - duplicate for left/right wings (conceptual; we just reuse chunks)
    - split into s chunks, run both wings
    - concat left+right
    - attention
    """
    def __init__(
        self,
        in_channels: int,
        s: int = 4,
        K: int = 2,
        kernels: Optional[List[int]] = None,
        attn_reduction: int = 8,
        keep_last_carry: bool = True,
    ):
        super().__init__()
        assert s >= 1
        assert K >= 1

        self.in_channels = in_channels
        self.s = s

        self.pad_channels = (s - (in_channels % s)) % s
        self.c_prime = in_channels + self.pad_channels
        self.chunk_channels = self.c_prime // s

        wing_cfg = WingConfig(
            s=s,
            K=K,
            kernels=kernels,
            keep_last_carry=keep_last_carry,
        )

        self.left = HMCNWing(self.chunk_channels, wing_cfg)
        self.right = HMCNWing(self.chunk_channels, wing_cfg)

        fused_channels = self.left.out_channels + self.right.out_channels

        self.attn = SEBlock(fused_channels, reduction=attn_reduction)

        # Optional final 1x1 to return something close to original channel count
        # (you can change this depending on your network design)
        self.out_proj = nn.Sequential(
            nn.Conv2d(fused_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def _pad_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_channels == 0:
            return x
        b, c, h, w = x.shape
        pad = x.new_zeros((b, self.pad_channels, h, w))
        return torch.cat([x, pad], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, in_channels, H, W)
        """
        assert x.dim() == 4, "Expected (B,C,H,W)"
        assert x.shape[1] == self.in_channels, f"Expected C={self.in_channels}, got {x.shape[1]}"

        x_pad = self._pad_channels(x)  # (B, C', H, W)

        # split into s chunks ϕ1..ϕs along channel dim
        chunks = list(torch.chunk(x_pad, self.s, dim=1))  # each (B, C'/s, H, W)

        y_left = self.left(chunks, direction="left")
        y_right = self.right(chunks, direction="right")

        y = torch.cat([y_left, y_right], dim=1)
        y = self.attn(y)
        y = self.out_proj(y)

        # global skip (optional but usually helps)
        return y + x


# -----------------------------
# Helpers
# -----------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


#