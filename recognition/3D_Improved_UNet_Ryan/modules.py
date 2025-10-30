# modules.py — 3D Improved UNet for Prostate Segmentation (PyTorch)
# Lightweight, training-ready components mirroring your 2D pipeline.
# - Residual Conv blocks with GroupNorm + LeakyReLU
# - Downsampling via strided conv; upsampling via trilinear + 1x1 conv (safer on memory)
# - Optional deep supervision; binary or multi-class compatible
# - Dice / BCE-Dice losses and Dice metric helpers

from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------- blocks -----------------------
class ConvNormAct3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 8, act: Optional[nn.Module] = None, bias: bool = False):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = act if act is not None else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))

class ResBlock3d(nn.Module):
    def __init__(self, ch: int, mid_ch: Optional[int] = None, groups: int = 8):
        super().__init__()
        mid = mid_ch or ch
        self.conv1 = ConvNormAct3d(ch, mid, k=3, s=1, groups=groups)
        self.conv2 = ConvNormAct3d(mid, ch, k=3, s=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))

class DownBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int = 1, groups: int = 8):
        super().__init__()
        self.down = ConvNormAct3d(in_ch, out_ch, k=3, s=2, groups=groups)  # stride-2 downsample
        self.body = nn.Sequential(*[ResBlock3d(out_ch, groups=groups) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.body(x)

class UpBlock3d(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, depth: int = 1, groups: int = 8):
        super().__init__()
        # upsample + 1x1 to align channels; avoids transpose artifacts
        self.up_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.post = nn.Sequential(
            ConvNormAct3d(out_ch + skip_ch, out_ch, k=3, s=1, groups=groups),
            *[ResBlock3d(out_ch, groups=groups) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # trilinear upsample by 2
        x = F.interpolate(x, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
        x = self.up_conv(x)
        # handle odd shapes by center-cropping skip to x
        if skip.shape[-3:] != x.shape[-3:]:
            dh = skip.shape[-3] - x.shape[-3]
            hh = skip.shape[-2] - x.shape[-2]
            wh = skip.shape[-1] - x.shape[-1]
            sD = dh // 2; sH = hh // 2; sW = wh // 2
            skip = skip[..., sD: sD + x.shape[-3], sH: sH + x.shape[-2], sW: sW + x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.post(x)

# ----------------------- UNet 3D -----------------------
class UNet3D_Improved(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,  # 1 for binary; >1 for multi-class
                 base_channels: int = 32,
                 depth_blocks: Tuple[int, int, int, int] = (1, 1, 1, 1),
                 groups: int = 8,
                 deep_supervision: bool = False):
        super().__init__()
        B = base_channels
        d0, d1, d2, d3 = depth_blocks
        act = nn.LeakyReLU(0.1, inplace=True)

        # stem
        self.stem = nn.Sequential(
            ConvNormAct3d(in_channels, B, k=3, s=1, groups=groups, act=act),
            ResBlock3d(B, groups=groups),
        )

        # encoder (3 downs to x3, then one more down inside bottleneck to xb)
        self.down1 = DownBlock3d(B,   2*B, depth=d0, groups=groups)  # x1
        self.down2 = DownBlock3d(2*B, 4*B, depth=d1, groups=groups)  # x2
        self.down3 = DownBlock3d(4*B, 8*B, depth=d2, groups=groups)  # x3
        self.bottleneck = nn.Sequential(                              # xb (4th down)
            DownBlock3d(8*B, 16*B, depth=d3, groups=groups),
            ResBlock3d(16*B, groups=groups)
        )

        # decoder (now 4 ups to mirror 4 downs)
        self.up3 = UpBlock3d(16*B, 8*B, 8*B, depth=1, groups=groups)  # with x3
        self.up2 = UpBlock3d( 8*B, 4*B, 4*B, depth=1, groups=groups)  # with x2
        self.up1 = UpBlock3d( 4*B, 2*B, 2*B, depth=1, groups=groups)  # with x1
        self.up0 = UpBlock3d( 2*B,   B,   B, depth=1, groups=groups)  # NEW: with x0

        self.head = nn.Sequential(
            ConvNormAct3d(B, B, k=3, s=1, groups=groups, act=act),
            nn.Conv3d(B, num_classes, kernel_size=1)
        )

        # deep supervision heads (optional)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.ds2 = nn.Conv3d(4*B, num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(8*B, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: [B, C, D, H, W]
        x0 = self.stem(x)           # [B, B,   D,H,W]
        x1 = self.down1(x0)         # [B, 2B,  D/2,H/2,W/2]
        x2 = self.down2(x1)         # [B, 4B,  D/4,H/4,W/4]
        x3 = self.down3(x2)         # [B, 8B,  D/8,H/8,W/8]
        xb = self.bottleneck(x3)    # [B,16B,  D/16,H/16,W/16]

        u3 = self.up3(xb, x3)       # [B, 8B,  D/8,H/8,W/8]
        u2 = self.up2(u3, x2)       # [B, 4B,  D/4,H/4,W/4]
        u1 = self.up1(u2, x1)       # [B, 2B,  D/2,H/2,W/2]
        u0 = self.up0(u1, x0)       # [B,  B,   D,H,W]  <- restored full res
        out = self.head(u0)

        if self.deep_supervision and self.training:
            # match spatial size to final out
            def up_to(sz_from: torch.Tensor, logits: torch.Tensor):
                return F.interpolate(logits, size=out.shape[-3:], mode="trilinear", align_corners=False)
            ds2 = up_to(out, self.ds2(u2))
            ds3 = up_to(out, self.ds3(u3))
            return out, ds2, ds3
        return out

# ----------------------- losses / metrics -----------------------
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred: logits or probs [B,1,D,H,W] (binary) or [B,C,D,H,W] (one-hot/softmax probs)
    target: binary mask [B,1,D,H,W] or int labels [B, D,H,W]
    """
    if pred.shape[1] == 1:  # binary
        prob = torch.sigmoid(pred)
        if target.dim() == 4:  # [B,D,H,W]
            target = target.unsqueeze(1)
        target = target.float()
        inter = torch.sum(prob * target)
        union = torch.sum(prob) + torch.sum(target)
        return (2 * inter + eps) / (union + eps)
    else:  # multi-class mean Dice (exclude background=0 if desired)
        prob = F.softmax(pred, dim=1)
        if target.dim() == 5:  # already one-hot
            tgt_oh = target
        else:
            C = prob.shape[1]
            tgt_oh = F.one_hot(target.long().clamp_min(0), num_classes=C).permute(0,4,1,2,3).float()
        dims = (0,2,3,4)
        inter = torch.sum(prob * tgt_oh, dim=dims)
        union = torch.sum(prob + tgt_oh, dim=dims)
        dice = (2*inter + eps) / (union + eps)
        return dice.mean()

def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    if pred.shape[1] != 1:
        raise ValueError("bce_dice_loss is for binary outputs; use CE/DiceCE for multi-class.")
    if target.dim() == 4:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(pred, target.float())
    dice = 1.0 - dice_coefficient(pred, target)
    return bce_weight * bce + (1.0 - bce_weight) * dice

class DiceCELoss(nn.Module):
    """ Combined multi-class Dice + CrossEntropy. """
    def __init__(self, ce_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target.long())
        # compute soft Dice against one-hot target
        C = pred.shape[1]
        prob = F.softmax(pred, dim=1)
        tgt_oh = F.one_hot(target.long().clamp_min(0), num_classes=C).permute(0,4,1,2,3).float()
        dims = (0,2,3,4)
        inter = torch.sum(prob * tgt_oh, dim=dims)
        union = torch.sum(prob + tgt_oh, dim=dims)
        dice = (2*inter + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + (1.0 - self.ce_weight) * dice_loss

# ----------------------- utils -----------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # quick smoke test
    net = UNet3D_Improved(in_channels=1, num_classes=1, base_channels=32)
    x = torch.randn(2, 1, 96, 128, 128)
    y = net(x)
    print("out:", y.shape, "params:", count_parameters(net))
