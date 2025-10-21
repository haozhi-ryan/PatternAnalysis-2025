# dataset.py
import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random

import torch.nn.functional as F


# ----- helpers -----
def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    # HipMRI slices sometimes have an extra singleton dim; drop it. (See Appendix B.) 
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr

def _basename_niigz(p: Path) -> str:
    """Return filename without .nii or .nii.gz."""
    s = p.name
    if s.endswith(".nii.gz"):
        return s[:-7]
    if s.endswith(".nii"):
        return s[:-4]
    return p.stem

def _normalize_stem(p: Path) -> str:
    """
    Make a comparable key so 'case_004_week_0_slice_0' matches 'seg_004_week_0_slice_0'.
    Also strips common mask suffixes like _seg/_mask.
    """
    s = _basename_niigz(p)
    # remove known prefixes
    for prefix in ("case_", "seg_", "image_", "mask_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # remove known suffixes
    for suf in ("_seg", "-seg", "_mask", "-mask", "_label", "-label"):
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s

def _pair_lists(img_dir: Path, seg_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = sorted(img_dir.glob("*.nii*"))
    segs = sorted(seg_dir.glob("*.nii*"))

    seg_map = {_normalize_stem(s): s for s in segs}

    pairs, missing = [], []
    for im in imgs:
        key = _normalize_stem(im)
        if key in seg_map:
            pairs.append((im, seg_map[key]))
        else:
            missing.append(im.name)

    if missing:
        print(f"[dataset] Warning: {len(missing)} images had no matching mask. First few:")
        for m in missing[:10]:
            print("   -", m)

    if not pairs:
        raise RuntimeError(f"No NIfTI pairs found under {img_dir} and {seg_dir}")
    return pairs


# ----- core dataset -----
class HipMRISlicesDataset(Dataset):
    """
    Loads 2D HipMRI slice images and their segmentation masks from NIfTI files.
    - If binary=True, mask is binarized for a single label (e.g., prostate).
    - If binary=False, mask is returned as integer class ids (multi-class).
    Returns:
      image: float32 tensor [1,H,W] (z-scored)
      mask:  uint8  tensor [1,H,W] if binary else [H,W] (long)
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        binary: bool = True,
        target_label: int = 4,    # change if your prostate label id differs
        augment: bool = False,
        seed: Optional[int] = None,
        fixed_size: Optional[tuple[int, int]] = (256, 128)
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.pairs = _pair_lists(self.image_dir, self.mask_dir)
        self.binary = binary
        self.target_label = target_label
        self.augment = augment
        self.fixed_size = fixed_size
        if seed is not None:
            random.seed(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _maybe_augment(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Simple geometry-only augments to keep pixel spacing consistent
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            msk = np.flip(msk, axis=1)
        if random.random() < 0.5:
            img = np.flip(img, axis=0)
            msk = np.flip(msk, axis=0)
        return img, msk

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]

        # --- load nifti (read-from-disk strategy as in Appendix B) ---
        img = nib.load(str(img_path)).get_fdata(caching='unchanged')  # 
        msk = nib.load(str(msk_path)).get_fdata(caching='unchanged')  # labels

        img = _ensure_2d(img).astype(np.float32)
        msk = _ensure_2d(msk).astype(np.uint8)

        # normalise image (z-score recommended in Appendix B) 
        img = _zscore(img)

        if self.binary:
            # Create binary mask for target_label (e.g., prostate)
            msk = (msk == self.target_label).astype(np.uint8)  # [H,W]
            msk_t = torch.from_numpy(msk[None, ...].copy())    # [1,H,W] uint8
        else:
            # keep integer classes; use LongTensor for CE/DiceCE losses
            msk_t = torch.from_numpy(msk.copy()).long()        # [H,W]

        if self.augment:
            img, msk_np = self._maybe_augment(img, msk_t.numpy() if self.binary else msk)
            if self.binary:
                msk_t = torch.from_numpy(msk_np.copy())
            else:
                msk_t = torch.from_numpy(msk_np.copy()).long()

        img_t = torch.from_numpy(img[None, ...].copy())  # [1,H,W] float32

        if self.fixed_size is not None:
          H, W = self.fixed_size
          # resize image (bilinear)
          img_t = F.interpolate(img_t.unsqueeze(0),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False).squeeze(0)
          # resize mask (nearest)
          if self.binary:
              msk_t = F.interpolate(msk_t.unsqueeze(0).float(),
                                    size=(H, W),
                                    mode="nearest").squeeze(0).to(torch.uint8)
          else:
              msk_t = F.interpolate(msk_t.unsqueeze(0).unsqueeze(0).float(),
                                    size=(H, W),
                                    mode="nearest").squeeze(0).squeeze(0).long()

        return img_t, msk_t, img_path.stem  # include ID for bookkeeping

# ----- convenience factory for your exact folder layout -----
def make_dataloaders(root="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
                     batch_size=8, num_workers=4,
                     binary=True, target_label=4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root)
    train_ds = HipMRISlicesDataset(
        image_dir=root/"keras_slices_train",
        mask_dir=root/"keras_slices_seg_train",
        binary=binary, target_label=target_label, augment=True)
    val_ds = HipMRISlicesDataset(
        image_dir=root/"keras_slices_validate",
        mask_dir=root/"keras_slices_seg_validate",
        binary=binary, target_label=target_label, augment=False)
    test_ds = HipMRISlicesDataset(
        image_dir=root/"keras_slices_test",
        mask_dir=root/"keras_slices_seg_test",
        binary=binary, target_label=target_label, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# Quick sanity check (run on Rangpur login node w/o GPU)
if __name__ == "__main__":
  train_loader, val_loader, test_loader = make_dataloaders(batch_size=2, num_workers=0)

  print(f"Pairs found: {len(train_loader.dataset)} (train)")
  print("Example pair:\n ", train_loader.dataset.pairs[0][0].name, "<->", train_loader.dataset.pairs[0][1].name)

  x, y, ids = next(iter(train_loader))
  print("Batch image shape:", x.shape)
  print("Batch mask shape:", y.shape)
  print("Any NaNs in images?", x.isnan().any().item())
  print("Unique mask values:", y.unique())
  print("Positive pixels in first mask:", int(y[0].sum()))
