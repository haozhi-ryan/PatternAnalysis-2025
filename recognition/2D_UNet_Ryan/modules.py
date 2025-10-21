# modules.py
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Building Blocks
# ----------------------
def kaiming_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.01)  # LeakyReLU a=0.01
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class ConvBlock(nn.Module):
    """
    Two 3x3 convs each followed by InstanceNorm + LeakyReLU.
    (Isensee-style: IN + LeakyReLU worked well for med seg.)
    """
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.in1   = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1  = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.in2   = nn.InstanceNorm2d(out_ch, affine=True)
        self.act2  = nn.LeakyReLU(0.01, inplace=True)

        self.dropout = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.act1(self.in1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.in2(self.conv2(x)))
        return x

class Down(nn.Module):
    """
    Strided conv (2x downsample) instead of maxpool (tends to help on med data),
    then a ConvBlock.
    """
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.LeakyReLU(0.01, inplace=True)
        self.block = ConvBlock(out_ch, out_ch, p_drop)

    def forward(self, x):
        x = self.act(self.norm(self.down(x)))
        x = self.block(x)
        return x

class Up(nn.Module):
    """
    TransposeConv upsample (2x), concat skip, then ConvBlock.
    """
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        # in_ch is (skip_ch + up_ch). We upsample to out_ch, then concat -> (out_ch + skip) so we
        # set the transposed conv to produce out_ch, then ConvBlock expects (out_ch + skip)->out_ch.
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)  # up input channels
        self.block = ConvBlock(in_ch, out_ch, p_drop)  # after concat

    def forward(self, x, skip):
        x = self.up(x)
        # pad/crop if needed (just in case odd dims)
        if x.shape[-2] != skip.shape[-2] or x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x

# ----------------------
# Improved UNet (2D)
# ----------------------
class ImprovedUNet2D(nn.Module):
    """
    Isensee-inspired 2D U-Net:
      - InstanceNorm2d + LeakyReLU
      - Strided-conv downs, Transpose-conv ups
      - Optional deep supervision (aux heads at decoder levels)
    Args:
      in_channels: 1 for HipMRI slices
      out_channels: 1 (binary) or C (multi-class)
      base_ch: width of first level (e.g., 32)
      depth: number of levels (4 is common)
      p_drop: dropout rate in ConvBlocks
      deep_supervision: if True, returns list of logits [out, aux1, aux2, ...]
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        p_drop: float = 0.0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        chs = [base_ch * (2 ** i) for i in range(depth)]
        # Encoder
        self.enc0 = ConvBlock(in_channels, chs[0], p_drop)
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(Down(chs[i-1], chs[i], p_drop))

        # Bottleneck
        self.bottleneck = ConvBlock(chs[-1], chs[-1], p_drop)

        # Decoder
        self.ups = nn.ModuleList()
        dec_chs = list(reversed(chs))
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 1):
            up_out = dec_chs[i+1]                         # e.g., 128, 64, 32
            self.ups.append(nn.ConvTranspose2d(dec_chs[i], up_out, kernel_size=2, stride=2))
            in_block = up_out * 2                         # concat: skip(up_out) + up(up_out)
            self.dec_blocks.append(ConvBlock(in_block, up_out, p_drop))


        # Heads
        self.head = nn.Conv2d(dec_chs[-1], out_channels, kernel_size=1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(dec_chs[i+1], out_channels, kernel_size=1) for i in range(depth - 2)
            ])
        else:
            self.aux_heads = None

        kaiming_init(self)

    def forward(self, x) -> torch.Tensor | List[torch.Tensor]:
        # Encoder
        skips = []
        x0 = self.enc0(x); skips.append(x0)
        x = x0
        for d in self.downs:
            x = d(x); skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        aux_logits = []
        for i in range(len(self.ups)):
            up = self.ups[i](x)
            skip = skips[-(i+2)]  # pull corresponding skip
            if up.shape[-2:] != skip.shape[-2:]:
                up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, up], dim=1)
            x = self.dec_blocks[i](x)
            # deep supervision from intermediate decoder features (except last)
            if self.deep_supervision and i < len(self.ups) - 1:
                aux_logits.append(self.aux_heads[i](x))

        out = self.head(x)
        if self.deep_supervision:
            # upsample aux to final size
            final_hw = out.shape[-2:]
            aux_logits = [F.interpolate(a, size=final_hw, mode="bilinear", align_corners=False)
                          for a in aux_logits]
            return [out] + aux_logits
        return out

# ----------------------
# Losses / Metrics
# ----------------------
def dice_coef(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Binary Dice on logits (B,1,H,W) and targets (B,1,H,W) in {0,1}.
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

class DiceLoss(nn.Module):
    """
    Binary Dice loss on logits.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 1.0 - dice_coef(logits, targets, self.eps)

class BCEWithLogitsDice(nn.Module):
    """
    BCE + Dice (binary). Good default for class imbalance.
    """
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.dw = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets.float()) * (1 - self.dw) + self.dice(logits, targets) * self.dw

# ----------------------
# Factory + Quick Tests
# ----------------------
def build_model(
    num_classes: int = 1,
    in_channels: int = 1,
    base_ch: int = 32,
    depth: int = 4,
    p_drop: float = 0.0,
    deep_supervision: bool = False,
) -> ImprovedUNet2D:
    """
    Helper to build the model with consistent params.
    Binary: num_classes=1; Multi-class: num_classes=C (use CE later).
    """
    return ImprovedUNet2D(
        in_channels=in_channels,
        out_channels=num_classes,
        base_ch=base_ch,
        depth=depth,
        p_drop=p_drop,
        deep_supervision=deep_supervision,
    )

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Minimal forward tests (CPU)
    x = torch.randn(2, 1, 256, 128)
    # Binary example
    m_bin = build_model(num_classes=1, base_ch=32, depth=4, p_drop=0.1, deep_supervision=True)
    y_list = m_bin(x)
    if isinstance(y_list, list):
        print("Binary w/ deep supervision heads:", [t.shape for t in y_list])
        y = y_list[0]
    else:
        y = y_list
    print("Binary main logits:", y.shape, "params:", count_params(m_bin))

    # Fake target + quick loss
    target = (torch.rand_like(y) > 0.7).float()
    loss_fn = BCEWithLogitsDice(dice_weight=0.5)
    loss = loss_fn(y, target)
    print("Sample loss (BCE+Dice):", float(loss))

    # Multiclass example (e.g., 5 classes)
    m_mc = build_model(num_classes=5, base_ch=32, depth=4, p_drop=0.0, deep_supervision=False)
    y_mc = m_mc(x)   # (B,5,H,W)
    print("Multiclass logits:", y_mc.shape, "params:", count_params(m_mc))
