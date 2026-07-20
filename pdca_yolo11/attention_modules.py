"""Custom attention blocks for Ultralytics YOLO YAML (ECA, CBAM, PDCA, etc.)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("ECA", "DCN", "SE", "SADA", "PDCA", "SAWA", "KUG", "LSAM", "register_attention_modules")


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al.) — channel-only, very low cost."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log2(max(channels, 1)) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1) -> 1D conv along channel index
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.act(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class SE(nn.Module):
    """Squeeze-and-excitation (channel gate)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        r = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SADA(nn.Module):
    """Stem-Aware Directional Attention — row/column/local gates on P3 neck features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.conv_s = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z_h = x.mean(dim=3, keepdim=True)
        z_w = x.mean(dim=2, keepdim=True)
        z_w_t = z_w.permute(0, 1, 3, 2)
        u = torch.cat([z_h, z_w_t], dim=2)
        f = F.silu(self.bn(self.conv1(u)))
        f_h, f_w = torch.split(f, [h, w], dim=2)
        f_w = f_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(f_h))
        a_w = torch.sigmoid(self.conv_w(f_w))
        a_s = torch.sigmoid(self.conv_s(x))
        return x * a_h * a_w * a_s


class PDCA(nn.Module):
    """Pose-Detection Co-Attention — cross-gate box/pose paths before the Pose head."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 32)
        self.box_path = nn.Conv2d(channels, mid, 1, bias=False)
        self.pose_path = nn.Conv2d(channels, mid, 1, bias=False)
        self.box_gate = nn.Conv2d(mid, channels, 1, bias=True)
        self.pose_gate = nn.Conv2d(mid, channels, 1, bias=True)
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_box = self.box_path(x)
        x_pose = self.pose_path(x)
        box_g = torch.sigmoid(self.box_gate(x_pose))
        pose_g = torch.sigmoid(self.pose_gate(x_box))
        return self.fuse(x * box_g + x * pose_g)


class SAWA(nn.Module):
    """Scale-Adaptive Weed Attention — multi-scale DW conv + avg/max channel gate."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.pool_fc = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.dw3(x) + self.dw5(x) + self.dw7(x)
        gate = self.pool_fc(torch.cat([x.mean(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)], dim=1))
        return x * gate + ms * 0.1


class KUG(nn.Module):
    """Keypoint Uncertainty Gate — feature confidence gate (inference-safe forward)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class LSAM(nn.Module):
    """Local Stem Affinity Module — criss-cross style local affinity on P3."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 16)
        self.q = nn.Conv2d(channels, mid, 1, bias=False)
        self.k = nn.Conv2d(channels, mid, 1, bias=False)
        self.v = nn.Conv2d(channels, mid, 1, bias=False)
        self.proj = nn.Conv2d(mid, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scale = q.size(1) ** 0.5
        k_h = k.mean(dim=3, keepdim=True).expand_as(k)
        v_h = v.mean(dim=3, keepdim=True).expand_as(v)
        attn_h = torch.sigmoid((q * k_h).sum(dim=1, keepdim=True) / scale)
        out_h = attn_h * v_h
        k_w = k.mean(dim=2, keepdim=True).expand_as(k)
        v_w = v.mean(dim=2, keepdim=True).expand_as(v)
        attn_w = torch.sigmoid((q * k_w).sum(dim=1, keepdim=True) / scale)
        out_w = attn_w * v_w
        return x + self.gamma * self.proj(out_h + out_w)


class DCN(nn.Module):
    """★#9 neck block — depthwise separable conv (DeformConv2d segfaults in this env)."""

    def __init__(self, channels: int, kernel_size: int = 3, bottleneck: int = 256) -> None:
        super().__init__()
        p = kernel_size // 2
        mid = min(channels, bottleneck)
        self.proj_in = nn.Conv2d(channels, mid, 1, bias=False) if channels > mid else nn.Identity()
        self.proj_out = nn.Conv2d(mid, channels, 1, bias=False) if channels > mid else nn.Identity()
        self.dw = nn.Conv2d(mid, mid, kernel_size, padding=p, groups=mid, bias=False)
        self.pw = nn.Conv2d(mid, mid, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = self.act(self.pw(self.dw(h)))
        return self.proj_out(h)


def register_attention_modules() -> None:
    """Expose custom/built-in attn modules to ultralytics.nn.tasks.parse_model."""
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules import CBAM

    mods = (
        ("ECA", ECA),
        ("SE", SE),
        ("DCN", DCN),
        ("SADA", SADA),
        ("PDCA", PDCA),
        ("SAWA", SAWA),
        ("KUG", KUG),
        ("LSAM", LSAM),
        ("CBAM", CBAM),
    )
    for name, mod in mods:
        if getattr(tasks, name, None) is not mod:
            setattr(tasks, name, mod)
