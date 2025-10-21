# predict.py
import argparse
from pathlib import Path
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import make_dataloaders
from modules import build_model, dice_coef

@torch.no_grad()
def run_inference(model, loader, device, out_dir: Path, threshold: float = 0.5, save_csv: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    masks_dir = out_dir / "pred_masks"
    overlays_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    rows = []
    dices = []

    for imgs, msks, ids in loader:
        imgs = imgs.to(device)                # [B,1,H,W]
        msks = msks.to(device)                # [B,1,H,W]
        logits = model(imgs)                  # [B,1,H,W]
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).to(msks.dtype)

        # Dice per-slice (binary)
        d = float(dice_coef(logits, msks))
        dices.append(d)

        # Save first in batch (B==1 for test loader as defined)
        name = ids if isinstance(ids, str) else ids[0]
        name = str(name).replace(".nii", "").replace(".gz", "")
        pred_np = preds[0, 0].cpu().numpy().astype(np.uint8)
        gt_np   = msks[0, 0].cpu().numpy().astype(np.uint8)
        img_np  = imgs[0, 0].cpu().numpy()

        # Save raw prediction mask
        np.save(masks_dir / f"{name}_pred.npy", pred_np)

        # Save overlay (image + GT + Pred)
        plt.figure(figsize=(6, 4))
        plt.title(f"{name} | Dice={d:.3f}")
        plt.imshow(img_np, cmap="gray")
        # Show GT and Pred as semi-transparent overlays
        gt_alpha = np.where(gt_np > 0, 0.35, 0.0)
        pred_alpha = np.where(pred_np > 0, 0.35, 0.0)
        plt.imshow(gt_np, alpha=gt_alpha)
        plt.imshow(pred_np, alpha=pred_alpha)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{name}_overlay.png", dpi=150)
        plt.close()

        rows.append({"id": name, "dice": f"{d:.6f}"})

    mean_dice = float(np.mean(dices)) if dices else 0.0
    print(f"Test mean Dice: {mean_dice:.4f}")

    if save_csv:
        with open(out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "dice"])
            w.writeheader()
            w.writerows(rows)
        with open(out_dir / "summary.txt", "w") as f:
            f.write(f"Test mean Dice: {mean_dice:.6f}\n")

    return mean_dice

def get_args():
    p = argparse.ArgumentParser(description="Predict & visualize HipMRI prostate segmentation")
    p.add_argument("--root", type=str, default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.ckpt")
    p.add_argument("--target_label", type=int, default=1, help="prostate label id used during training")
    p.add_argument("--batch_size", type=int, default=1, help="test loader batch size (keep 1 for per-slice saves)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--p_drop", type=float, default=0.0)
    p.add_argument("--outdir", type=str, default="preds")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build only the test loader
    _, _, test_loader = make_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers if torch.cuda.is_available() else 0,
        binary=True,
        target_label=args.target_label
    )

    # Rebuild model and load best checkpoint
    model = build_model(
        num_classes=1, in_channels=1,
        base_ch=args.base_ch, depth=args.depth, p_drop=args.p_drop, deep_supervision=False
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (valDice={ckpt.get('val_dice', -1):.4f})")

    out_dir = Path(args.outdir)
    run_inference(model, test_loader, device, out_dir, threshold=args.threshold, save_csv=True)

if __name__ == "__main__":
    main()
