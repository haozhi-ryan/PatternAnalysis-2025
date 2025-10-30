# train.py — 3D Improved UNet training (patch-train + full-vol eval)
import argparse, os, csv, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import UNet3D_Improved, bce_dice_loss, dice_coefficient, count_parameters
from dataset import make_dataloaders_patches, make_dataloaders_fullvol

def set_seed(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --- tiny synthetic dataset for quick dry-run ---
class _Random3DSet(Dataset):
    def __init__(self, n=4, shape=(16, 32, 32)):
        self.n = n; self.shape = shape
    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.randn(1, *self.shape)
        y = (torch.rand(1, *self.shape) > 0.7).to(torch.uint8)
        return x, y, f"rand_{idx}"

@torch.no_grad()
def eval_fullvol(model, val_loader, device):
    model.eval()
    dices = []
    for x, y, _ids in val_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        d = dice_coefficient(model(x), y).item()
        dices.append(d)
    return float(np.mean(dices)), dices

def save_ckpt(model, optimizer, scaler, epoch, best_dice, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_dice": best_dice
    }, outdir / "checkpoint_best.pt")

def plot_curves(log, outdir):
    outdir = Path(outdir); outdir.mkdir(exist_ok=True, parents=True)
    epochs = [r["epoch"] for r in log]
    tr = [r["train_loss"] for r in log]
    vd = [r["val_dice"] for r in log]
    plt.figure(); plt.plot(epochs, tr, label="train_loss"); plt.plot(epochs, vd, label="val_dice")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=150); plt.close()

def write_csv(log, outdir):
    out = Path(outdir) / "training_log.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_dice","lr","time_sec"])
        w.writeheader(); w.writerows(log)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[device] {device}")

    # --- loaders ---
    if args.quick:
        tr_ds = _Random3DSet(n=max(2, 2*args.batch_size), shape=tuple(args.patch_size))
        va_ds = _Random3DSet(n=2, shape=tuple(args.eval_size))
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=0)
    else:
        tr_loader, _, _ = make_dataloaders_patches(
            patch_size=tuple(args.patch_size),
            patches_per_volume=args.patches_per_volume,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            binary=True, target_label=args.target_label, seed=args.seed,
        )
        _, va_loader, _ = make_dataloaders_fullvol(
            batch_size=1, num_workers=args.num_workers,
            binary=True, target_label=args.target_label, fixed_size=tuple(args.eval_size), seed=args.seed
        )

    # --- model/opt/sched/amp ---
    net = UNet3D_Improved(in_channels=1, num_classes=1, base_channels=args.base_ch).to(device)
    print(f"[model] params: {count_parameters(net):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda" and args.amp))

    best = -1.0; history = []; outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        t0 = time.time(); net.train(); running = 0.0
        for xb, yb, _ids in tr_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type=="cuda" and args.amp)):
                logits = net(xb); loss = bce_dice_loss(logits, yb, bce_weight=args.bce_weight)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            running += loss.item() * xb.size(0)

        sched.step()
        train_loss = running / len(tr_loader.dataset)
        val_dice, _ = eval_fullvol(net, va_loader, device)

        row = {"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice,
               "lr": sched.get_last_lr()[0], "time_sec": round(time.time()-t0,2)}
        history.append(row)
        print(f"[{epoch:03d}] loss={train_loss:.4f}  val_dice={val_dice:.4f}  lr={row['lr']:.3e}  t={row['time_sec']}s")

        if val_dice > best:
            best = val_dice; save_ckpt(net, opt, scaler, epoch, best, outdir)
        if epoch % max(1, args.save_every) == 0 or epoch == args.epochs:
            write_csv(history, outdir); plot_curves(history, outdir)

    print(f"[done] best val Dice = {best:.4f}  (checkpoint_best.pt)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--patch_size", nargs=3, type=int, default=[96,128,128])
    p.add_argument("--patches_per_volume", type=int, default=4)
    p.add_argument("--eval_size", nargs=3, type=int, default=[128,256,256])
    p.add_argument("--target_label", type=int, default=1)
    p.add_argument("--outdir", type=str, default="./runs_unet3d")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--quick", action="store_true", help="Run fast synthetic dry-run to catch errors")
    args = p.parse_args()
    train(args)
