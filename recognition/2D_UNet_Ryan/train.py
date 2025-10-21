# train.py
import argparse, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import make_dataloaders
from modules import build_model, BCEWithLogitsDice, dice_coef

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss, total_dice, n = 0.0, 0.0, 0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = loss_fn(logits, msks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        opt.step()

        with torch.no_grad():
            bs = imgs.size(0)
            total_loss += float(loss) * bs
            total_dice += float(dice_coef(logits, msks)) * bs
            n += bs
    return total_loss / max(n,1), total_dice / max(n,1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_dice, n = 0.0, 0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        bs = imgs.size(0)
        total_dice += float(dice_coef(logits, msks)) * bs
        n += bs
    return total_dice / max(n,1)

def get_args():
    p = argparse.ArgumentParser(description="Train 2D Improved U-Net on HipMRI slices")
    p.add_argument("--root", type=str, default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    p.add_argument("--target_label", type=int, default=1, help="prostate label id in masks")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--p_drop", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="runs/hipmri_unet")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.ckpt")
    p.add_argument("--overfit_batches", type=int, default=0, help=">0 to overfit this many batches for sanity")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- I/O ---
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=outdir.as_posix())

    # --- Data ---
    train_loader, val_loader, test_loader = make_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers if torch.cuda.is_available() else 0,
        binary=True,
        target_label=args.target_label
    )

    # Overfit mode (replace loaders with trimmed versions)
    if args.overfit_batches > 0:
        from itertools import islice
        small = list(islice(iter(train_loader), args.overfit_batches))
        class SmallLoader:
            def __iter__(self): return iter(small)
            def __len__(self): return len(small)
        train_loader = SmallLoader()
        val_loader = SmallLoader()

    # --- Model / Opt / Loss ---
    model = build_model(
        num_classes=1, in_channels=1,
        base_ch=args.base_ch, depth=args.depth,
        p_drop=args.p_drop, deep_supervision=False
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = BCEWithLogitsDice(dice_weight=0.5)

    # --- Train loop ---
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_dice = evaluate(model, val_loader, device)
        sched.step(val_dice)

        tb.add_scalar("loss/train", tr_loss, epoch)
        tb.add_scalar("dice/train", tr_dice, epoch)
        tb.add_scalar("dice/val", val_dice, epoch)
        tb.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | trainDice {tr_dice:.4f} | valDice {val_dice:.4f} | {dt:.1f}s")

        if val_dice > best_val:
            best_val = val_dice
            torch.save({"model": model.state_dict(), "val_dice": best_val, "args": vars(args)}, args.checkpoint)
            print(f"  ✓ Saved best checkpoint to {args.checkpoint} (valDice={best_val:.4f})")

    tb.close()

    # --- Test (best checkpoint) ---
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint (valDice={ckpt.get('val_dice', -1):.4f})")
    test_dice = evaluate(model, test_loader, device)
    print(f"TEST Dice (prostate): {test_dice:.4f}")

if __name__ == "__main__":
    main()
