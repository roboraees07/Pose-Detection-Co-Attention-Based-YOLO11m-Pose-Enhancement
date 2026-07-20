"""SRD-YOLO blocks (DWR + RepViT/RVB + MultiSEAM) adapted for Ultralytics YOLO11-Pose."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, C3k2, Conv, DFL, Pose
from ultralytics.nn.modules.block import Bottleneck

__all__ = (
    "DWR",
    "MultiSEAM",
    "RVBottleneck",
    "C3k2_RVB",
    "C3k2_DWR",
    "Pose_SEAM",
    "register_srd_modules",
)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: float = 0.25) -> None:
        super().__init__()
        rd = max(int(channels * reduction), 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, rd, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(rd, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        groups: int = 1,
        bn_weight_init: float = 1,
    ) -> None:
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, groups=groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class RepVGGDW(nn.Module):
    def __init__(self, ed: int) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x) + self.conv1(x) + x)


class RepViTBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        hidden_dim: int,
        oup: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        assert stride in (1, 2)
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, 1, 1, 0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU(),
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                )
            )
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU(),
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


class RVBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        self.block = RepViTBlock(c1, c1, c2, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWR(nn.Module):
    """DWRM block from SRD-YOLO (Liu et al., Plant Phenomics 2025)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv_3x3 = Conv(dim, dim // 2, 3)
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)
        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_3x3(x)
        y1 = self.conv_3x3_d1(y)
        y2 = self.conv_3x3_d3(y)
        y3 = self.conv_3x3_d5(y)
        out = self.conv_1x1(torch.cat([y1, y2, y3], dim=1))
        return out + x


class C3k2_RVB(C3k2):
    """C3k2 with RepViT (RVB) bottlenecks — YOLO11 analogue of C2f_RVB."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super(C3k2, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RVBottleneck(self.c, self.c) for _ in range(n))


class C3k2_DWR(C3k2):
    """C3k2 with DWRM bottlenecks — YOLO11 analogue of C2f_DWR."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super(C3k2, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DWR(self.c) for _ in range(n))


class MultiSEAM(nn.Module):
    """Multi-scale SEAM attention (SRD-YOLO paper)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        depth: int = 1,
        kernel_size: int = 3,
        patch_size: tuple[int, int, int] = (3, 5, 7),
        reduction: int = 16,
    ) -> None:
        super().__init__()
        c2 = c1 if c1 != c2 else c2
        self.DCovN0 = self._dcovn(c1, c2, depth, kernel_size, patch_size[0])
        self.DCovN1 = self._dcovn(c1, c2, depth, kernel_size, patch_size[1])
        self.DCovN2 = self._dcovn(c1, c2, depth, kernel_size, patch_size[2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(c2 // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(c2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c2, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _dcovn(c1: int, c2: int, depth: int, kernel_size: int, patch_size: int) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(c1, c2, patch_size, patch_size, 0, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(c2),
        ]
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(c2, c2, kernel_size, 1, kernel_size // 2, groups=c2, bias=False),
                            nn.SiLU(),
                            nn.BatchNorm2d(c2),
                        )
                    ),
                    nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
                    nn.SiLU(),
                    nn.BatchNorm2d(c2),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y4 = self.avg_pool(x).view(b, c)
        y = (
            self.avg_pool(y0).view(b, c)
            + self.avg_pool(y1).view(b, c)
            + self.avg_pool(y2).view(b, c)
            + y4
        ) / 4.0
        y = self.fc(y).view(b, c, 1, 1)
        return x * torch.exp(y)


class Pose_SEAM(Pose):
    """Pose head with MultiSEAM on box/cls branches (SRD-YOLO)."""

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (1, 3),
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ) -> None:
        nn.Module.__init__(self)
        self.legacy = True
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                MultiSEAM(c2, c2, depth=1),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                MultiSEAM(c3, c3, depth=1),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)


def _patch_parse_model() -> None:
    import ultralytics.nn.tasks as tasks

    if getattr(tasks, "_srd_parse_patched", False):
        return

    _orig = tasks.parse_model

    def parse_model(d, ch, verbose=True):
        import ast
        import contextlib

        import ultralytics.nn.tasks as tm
        from ultralytics.nn.tasks import (
            A2C2f,
            ADown,
            AConv,
            AIFI,
            BottleneckCSP,
            C1,
            C2,
            C2PSA,
            C2f,
            C2fAttn,
            C2fCIB,
            C2fPSA,
            C3,
            C3Ghost,
            C3TR,
            C3k2,
            C3x,
            CBFuse,
            CBLinear,
            Classify,
            Concat,
            Conv,
            ConvTranspose,
            DWConv,
            DWConvTranspose2d,
            Detect,
            ELAN1,
            Focus,
            GhostBottleneck,
            GhostConv,
            HGBlock,
            HGStem,
            ImagePoolingAttn,
            Index,
            OBB,
            OBB26,
            PSA,
            Pose,
            Pose26,
            ResNetLayer,
            RTDETRDecoder,
            RepC3,
            RepNCSPELAN4,
            SCDown,
            Segment,
            Segment26,
            SPP,
            SPPELAN,
            SPPF,
            TorchVision,
            WorldDetect,
            YOLOEDetect,
            YOLOESegment,
            YOLOESegment26,
            make_divisible,
            v10Detect,
        )
        from ultralytics.utils import LOGGER, colorstr
        from ultralytics.utils.torch_utils import initialize_weights

        tm.C3k2_RVB = C3k2_RVB
        tm.C3k2_DWR = C3k2_DWR
        tm.Pose_SEAM = Pose_SEAM

        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        base_modules = frozenset(
            {
                Classify,
                Conv,
                ConvTranspose,
                GhostConv,
                Bottleneck,
                GhostBottleneck,
                SPP,
                SPPF,
                C2fPSA,
                C2PSA,
                DWConv,
                Focus,
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C3k2_RVB,
                C3k2_DWR,
                RepNCSPELAN4,
                ELAN1,
                ADown,
                AConv,
                SPPELAN,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                torch.nn.ConvTranspose2d,
                DWConvTranspose2d,
                C3x,
                RepC3,
                PSA,
                SCDown,
                C2fCIB,
                A2C2f,
            }
        )
        repeat_modules = frozenset(
            {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C3k2_RVB,
                C3k2_DWR,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                A2C2f,
            }
        )
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else tm.__dict__[m]
            )
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                    )
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if m in {C3k2, C3k2_RVB, C3k2_DWR}:
                    legacy = False
                    if scale in "mlx" and m is C3k2:
                        args[3] = True
                if m is A2C2f:
                    legacy = False
                    if scale in "lx":
                        args.extend((True, 1.2))
                if m is C2fCIB:
                    legacy = False
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in frozenset({HGStem, HGBlock}):
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in frozenset(
                {
                    Detect,
                    WorldDetect,
                    YOLOEDetect,
                    Segment,
                    Segment26,
                    YOLOESegment,
                    YOLOESegment26,
                    Pose,
                    Pose_SEAM,
                    Pose26,
                    OBB,
                    OBB26,
                }
            ):
                args.extend([reg_max, end2end, [ch[x] for x in f]])
                if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                if m in {
                    Detect,
                    YOLOEDetect,
                    Segment,
                    Segment26,
                    YOLOESegment,
                    YOLOESegment26,
                    Pose,
                    Pose_SEAM,
                    Pose26,
                    OBB,
                    OBB26,
                }:
                    m.legacy = legacy
            elif m is v10Detect:
                args.append([ch[x] for x in f])
            elif m is ImagePoolingAttn:
                args.insert(1, [ch[x] for x in f])
            elif m is RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            elif m in frozenset({TorchVision, Index}):
                c2 = args[0]
                c1 = ch[f]
                args = [*args[1:]]
            else:
                c2 = ch[f]

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return torch.nn.Sequential(*layers), sorted(save)

    tasks.parse_model = parse_model
    tasks._srd_parse_patched = True
    tasks._srd_parse_model_orig = _orig


def register_srd_modules() -> None:
    """Register SRD blocks for YOLO11 YAML parsing and training."""
    import ultralytics.nn.tasks as tasks

    for name, mod in (
        ("C3k2_RVB", C3k2_RVB),
        ("C3k2_DWR", C3k2_DWR),
        ("Pose_SEAM", Pose_SEAM),
        ("MultiSEAM", MultiSEAM),
        ("DWR", DWR),
    ):
        setattr(tasks, name, mod)
    _patch_parse_model()
