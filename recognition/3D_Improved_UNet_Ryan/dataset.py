# dataset.py — Prostate 3D (HipMRI_Study_open / semantic_MRs + semantic_labels_only)
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import re

# ---------- config (paths you gave) ----------
DEFAULT_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")
DEFAULT_IMG_DIR = DEFAULT_ROOT / "semantic_MRs"
DEFAULT_MSK_DIR = DEFAULT_ROOT / "semantic_labels_only"

# ---------- helpers ----------
def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)

def _basename_niigz(p: Path) -> str:
    s = p.name
    if s.endswith(".nii.gz"): return s[:-7]
    if s.endswith(".nii"):    return s[:-4]
    return p.stem

def _normalize_stem(p: Path) -> str:
    s = p.name
    if s.endswith(".nii.gz"): s = s[:-7]
    elif s.endswith(".nii"):  s = s[:-4]
    # strip the terminal modality token so keys match:
    # ..._LFOV  ↔  ..._SEMANTIC
    s = s.replace("_LFOV", "").replace("_SEMANTIC", "")
    return s

def _pair_lists(img_dir: Path, seg_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = sorted(list(img_dir.glob("*.nii")) + list(img_dir.glob("*.nii.gz")))
    segs = sorted(list(seg_dir.glob("*.nii")) + list(seg_dir.glob("*.nii.gz")))
    seg_map = {_normalize_stem(s): s for s in segs}
    pairs, missing = [], []
    for im in imgs:
        key = _normalize_stem(im)
        if key in seg_map:
            pairs.append((im, seg_map[key]))
        else:
            missing.append(im.name)
    if missing:
        print(f"[dataset] Warning: {len(missing)} images had no matching mask (showing up to 10):")
        for m in missing[:10]:
            print("   -", m)
    if not pairs:
        raise RuntimeError(f"No NIfTI pairs under {img_dir} and {seg_dir}")
    return pairs

def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 4 and 1 in a.shape:
        a = np.squeeze(a)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D array, got {a.shape}")
    return a  # assume [D,H,W]

def _resize3d(img_t: torch.Tensor, msk_t: torch.Tensor, out_size: Tuple[int,int,int], binary: bool):
    D,H,W = out_size
    img_t = F.interpolate(img_t.unsqueeze(0), size=(D,H,W), mode="trilinear", align_corners=False).squeeze(0)
    if binary:
        m = F.interpolate(msk_t.unsqueeze(0).float(), size=(D,H,W), mode="nearest").squeeze(0).to(torch.uint8)
    else:
        m = F.interpolate(msk_t.unsqueeze(0).unsqueeze(0).float(), size=(D,H,W), mode="nearest").squeeze(0).squeeze(0).long()
    return img_t, m

def _patient_id_from_stem(stem: str) -> str:
    # e.g. "B006_Week0_LFOV" -> "B006"
    return stem.split("_")[0]

# ---------- datasets ----------
class Prostate3DVolumeDataset(Dataset):
    """
    Loads full 3D volumes from semantic_MRs and semantic_labels_only.
    Returns image [1,D,H,W] float32 (z-scored), mask [1,D,H,W] uint8 (binary) or [D,H,W] long.
    """
    def __init__(
        self,
        image_dir: Path = DEFAULT_IMG_DIR,
        mask_dir: Path = DEFAULT_MSK_DIR,
        binary: bool = True,
        target_label: int = 1,               # many HipMRI labels already binary; change if needed
        augment: bool = False,
        fixed_size: Optional[Tuple[int,int,int]] = (128,256,256),  # adjust to your downsampled dims
        intensity_jitter: bool = False,
        restrict_ids: Optional[List[str]] = None,   # stems to include (for split)
        seed: Optional[int] = 1337,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.pairs = _pair_lists(self.image_dir, self.mask_dir)
        if restrict_ids is not None:
            keep = set(restrict_ids)
            self.pairs = [(i,m) for (i,m) in self.pairs if _basename_niigz(i) in keep]
        if not self.pairs:
            raise RuntimeError("No pairs after filtering by restrict_ids")

        self.binary = binary
        self.target_label = target_label
        self.augment = augment
        self.fixed_size = fixed_size
        self.intensity_jitter = intensity_jitter
        if seed is not None:
            random.seed(seed)

    def __len__(self): return len(self.pairs)

    def _maybe_augment_geom(self, img: torch.Tensor, msk: torch.Tensor):
        # flips in D/H/W; random 90° rotations in H-W
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1]); msk = torch.flip(msk, dims=[1] if msk.dim()==4 else [0])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2]); msk = torch.flip(msk, dims=[2] if msk.dim()==4 else [1])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[3]); msk = torch.flip(msk, dims=[3] if msk.dim()==4 else [2])
        if random.random() < 0.5:
            k = random.randint(1,3)
            img = torch.rot90(img, k, dims=(2,3))
            if msk.dim()==4: msk = torch.rot90(msk, k, dims=(2,3))
            else:            msk = torch.rot90(msk.unsqueeze(0), k, dims=(1,2)).squeeze(0)
        return img, msk

    def _maybe_augment_intensity(self, img: torch.Tensor):
        if random.random() < 0.5:
            img = img * (1.0 + (random.random()*0.3 - 0.15))  # contrast ±15%
        if random.random() < 0.5:
            img = img + (random.random()*0.3 - 0.15)          # brightness ±0.15
        return img

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]
        img_np = nib.load(str(img_path)).get_fdata(caching='unchanged').astype(np.float32)
        msk_np = nib.load(str(msk_path)).get_fdata(caching='unchanged')
        img_np = _ensure_3d(img_np); msk_np = _ensure_3d(msk_np)
        img_np = _zscore(img_np)

        img_t = torch.from_numpy(img_np[None, ...].copy()).float()   # [1,D,H,W]
        if self.binary:
            m = (msk_np == self.target_label).astype(np.uint8)
            msk_t = torch.from_numpy(m[None, ...].copy())            # [1,D,H,W] uint8
        else:
            msk_t = torch.from_numpy(msk_np.copy()).long()           # [D,H,W]

        if self.fixed_size is not None:
            img_t, msk_t = _resize3d(img_t, msk_t, self.fixed_size, self.binary)

        if self.augment:
            img_t, msk_t = self._maybe_augment_geom(img_t, msk_t)
            if self.intensity_jitter:
                img_t = self._maybe_augment_intensity(img_t)

        return img_t, msk_t, _basename_niigz(img_path)

class Prostate3DPatchDataset(Dataset):
    """
    Random 3D patches for memory-efficient training.
    """
    def __init__(
        self,
        image_dir: Path = DEFAULT_IMG_DIR,
        mask_dir: Path = DEFAULT_MSK_DIR,
        patch_size: Tuple[int,int,int] = (96,128,128),
        patches_per_volume: int = 4,
        binary: bool = True,
        target_label: int = 1,
        augment: bool = True,
        ensure_foreground: bool = True,
        restrict_ids: Optional[List[str]] = None,
        seed: Optional[int] = 1337,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.pairs = _pair_lists(self.image_dir, self.mask_dir)
        if restrict_ids is not None:
            keep = set(restrict_ids)
            self.pairs = [(i,m) for (i,m) in self.pairs if _basename_niigz(i) in keep]
        if not self.pairs:
            raise RuntimeError("No pairs after filtering by restrict_ids")

        self.patch_size = patch_size
        self.ppv = patches_per_volume
        self.binary = binary
        self.target_label = target_label
        self.augment = augment
        self.ensure_foreground = ensure_foreground
        self._cache = [None] * len(self.pairs)
        if seed is not None:
            random.seed(seed)

    def __len__(self): return len(self.pairs) * self.ppv

    def _load_pair(self, i: int):
        if self._cache[i] is not None:
            return self._cache[i]
        img_path, msk_path = self.pairs[i]
        img_np = nib.load(str(img_path)).get_fdata(caching='unchanged').astype(np.float32)
        msk_np = nib.load(str(msk_path)).get_fdata(caching='unchanged')
        img_np = _ensure_3d(img_np); msk_np = _ensure_3d(msk_np)
        img_np = _zscore(img_np)
        img_t = torch.from_numpy(img_np[None, ...].copy()).float()       # [1,D,H,W]
        if self.binary:
            m = (msk_np == self.target_label).astype(np.uint8)
            m_t = torch.from_numpy(m[None, ...].copy())                  # [1,D,H,W]
        else:
            m_t = torch.from_numpy(msk_np.copy()).long()                 # [D,H,W]
        stem = _basename_niigz(img_path)
        self._cache[i] = (img_t, m_t, stem)
        return self._cache[i]

    def _rand_crop_coords(self, shape: Tuple[int,int,int], size: Tuple[int,int,int]):
        D,H,W = shape; d,h,w = size
        sd = 0 if D<=d else random.randint(0, D-d)
        sh = 0 if H<=h else random.randint(0, H-h)
        sw = 0 if W<=w else random.randint(0, W-w)
        return sd, sh, sw

    def __getitem__(self, idx: int):
        vol_idx = idx // self.ppv
        img_t, msk_t, stem = self._load_pair(vol_idx)
        D,H,W = img_t.shape[1:]
        d,h,w = self.patch_size

        tries = 0
        while True:
            sd, sh, sw = self._rand_crop_coords((D,H,W), (d,h,w))
            img = img_t[:, sd:sd+d, sh:sh+h, sw:sw+w]
            if msk_t.dim()==4:
                msk = msk_t[:, sd:sd+d, sh:sh+h, sw:sw+w]
                fg = (msk.sum() > 0)
            else:
                msk = msk_t[sd:sd+d, sh:sh+h, sw:sw+w]
                fg = (msk.sum() > 0)
            if not self.ensure_foreground or fg or tries > 8:
                break
            tries += 1

        if self.augment:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[1]); msk = torch.flip(msk, dims=[1] if msk.dim()==4 else [0])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2]); msk = torch.flip(msk, dims=[2] if msk.dim()==4 else [1])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[3]); msk = torch.flip(msk, dims=[3] if msk.dim()==4 else [2])

        return img, msk, f"{stem}"

# ---------- splitting utilities ----------
def build_patient_splits(pairs: List[Tuple[Path,Path]], seed: int = 1337,
                         ratios=(0.7, 0.15, 0.15)) -> Dict[str, List[str]]:
    by_patient: Dict[str, List[str]] = {}
    for img, _ in pairs:
        stem = _basename_niigz(img)
        pid = _patient_id_from_stem(stem)
        by_patient.setdefault(pid, []).append(stem)

    patients = sorted(by_patient.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_tr = int(ratios[0]*n)
    n_va = int(ratios[1]*n)
    tr_p = patients[:n_tr]
    va_p = patients[n_tr:n_tr+n_va]
    te_p = patients[n_tr+n_va:]

    def stems(ps): 
        acc = []
        for p in ps: acc.extend(by_patient[p])
        return acc

    return {"train": stems(tr_p), "val": stems(va_p), "test": stems(te_p)}

# ---------- convenience dataloaders ----------
def make_dataloaders_fullvol(
    batch_size: int = 1,
    num_workers: int = 4,
    binary: bool = True,
    target_label: int = 1,
    fixed_size: Optional[Tuple[int,int,int]] = (128,256,256),
    seed: int = 1337,
):
    # discover and split by patient
    all_pairs = _pair_lists(DEFAULT_IMG_DIR, DEFAULT_MSK_DIR)
    splits = build_patient_splits(all_pairs, seed=seed)

    train_ds = Prostate3DVolumeDataset(binary=binary, target_label=target_label,
                                       augment=True, intensity_jitter=True,
                                       fixed_size=fixed_size, restrict_ids=splits["train"], seed=seed)
    val_ds   = Prostate3DVolumeDataset(binary=binary, target_label=target_label,
                                       augment=False, fixed_size=fixed_size, restrict_ids=splits["val"], seed=seed)
    test_ds  = Prostate3DVolumeDataset(binary=binary, target_label=target_label,
                                       augment=False, fixed_size=fixed_size, restrict_ids=splits["test"], seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def make_dataloaders_patches(
    patch_size: Tuple[int,int,int] = (96,128,128),
    patches_per_volume: int = 4,
    batch_size: int = 2,
    num_workers: int = 4,
    binary: bool = True,
    target_label: int = 1,
    seed: int = 1337,
):
    all_pairs = _pair_lists(DEFAULT_IMG_DIR, DEFAULT_MSK_DIR)
    splits = build_patient_splits(all_pairs, seed=seed)

    train_ds = Prostate3DPatchDataset(patch_size=patch_size, patches_per_volume=patches_per_volume,
                                      binary=binary, target_label=target_label,
                                      augment=True, ensure_foreground=True,
                                      restrict_ids=splits["train"], seed=seed)
    val_ds   = Prostate3DVolumeDataset(binary=binary, target_label=target_label,
                                       augment=False, fixed_size=(128,256,256), restrict_ids=splits["val"], seed=seed)
    test_ds  = Prostate3DVolumeDataset(binary=binary, target_label=target_label,
                                       augment=False, fixed_size=(128,256,256), restrict_ids=splits["test"], seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ---------- sanity check ----------
if __name__ == "__main__":
    try:
        tr, va, te = make_dataloaders_patches(batch_size=2, num_workers=0, seed=1337)
        xb, yb, ids = next(iter(tr))
        print("[patch] x:", xb.shape, "y:", yb.shape, "id0:", ids[0])
        print(" NaNs?", xb.isnan().any().item(), "mask uniq (b0):", torch.unique(yb[0]))
    except Exception as e:
        print("[patch] Skipped:", e)

    try:
        trv, vv, tv = make_dataloaders_fullvol(batch_size=1, num_workers=0, seed=1337)
        x, y, ids = next(iter(trv))
        print("[full] x:", x.shape, "y:", y.shape, "id0:", ids[0])
        print(" NaNs?", x.isnan().any().item(), "mask uniq (b0):", torch.unique(y[0] if y.dim()==4 else y))
    except Exception as e:
        print("[full] Skipped:", e)
