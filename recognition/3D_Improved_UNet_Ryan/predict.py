# predict.py — 3D UNet3D inference on full volumes (test set)
# - Loads checkpoint, runs inference on test_loader (full volumes)
# - Computes Dice per case and overall mean
# - Optionally saves predicted masks as NIfTI (.nii.gz) with original affine/header

import argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from modules import UNet3D_Improved, dice_coefficient, count_parameters
from dataset import make_dataloaders_fullvol, DEFAULT_IMG_DIR

@torch.no_grad()
def run_inference(
    ckpt_path: Path,
    outdir: Path,
    threshold: float = 0.5,
    eval_size=(128, 256, 256),
    target_label: int = 1,
    save_nii: bool = True,
    device_str: str = None,
):
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)

    # dataloader (full volumes) — uses dataset resizing to eval_size
    _, _, test_loader = make_dataloaders_fullvol(
        batch_size=1,
        num_workers=2,
        binary=True,
        target_label=target_label,
        fixed_size=tuple(eval_size),
        seed=1337,
    )

    # model
    net = UNet3D_Improved(in_channels=1, num_classes=1, base_channels=32).to(device)
    print(f"[model] params: {count_parameters(net):,}")

    # load weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    # results
    rows = []
    dices = []

    for xb, yb, ids in test_loader:
        case_id = ids[0]  # e.g., "B006_Week0_LFOV"
        xb = xb.to(device).float()             # [1,1,D,H,W]
        yb = yb.to(device)                     # [1,1,D,H,W] uint8

        logits = net(xb)                       # [1,1,D,H,W]
        d = dice_coefficient(logits, yb).item()
        dices.append(d)

        # binarize prediction
        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).to(torch.uint8)  # [1,1,D,H,W]

        # save prediction as NIfTI (resampled back to original volume shape if needed)
        if save_nii:
            # find original NIfTI (prefer .nii.gz, fallback to .nii)
            cand1 = DEFAULT_IMG_DIR / f"{case_id}.nii.gz"
            cand2 = DEFAULT_IMG_DIR / f"{case_id}.nii"
            img_path = cand1 if cand1.exists() else cand2
            if not img_path.exists():
                # fallback: save in resized eval space (identity affine)
                print(f"[warn] original image not found for {case_id}; saving in eval grid.")
                to_save = pred[0, 0].cpu().numpy().astype(np.uint8)
                nii = nib.Nifti1Image(to_save, affine=np.eye(4))
                nib.save(nii, outdir / f"{case_id}_pred_resized.nii.gz")
            else:
                img_nifti = nib.load(str(img_path))
                orig_aff = img_nifti.affine
                orig_hdr = img_nifti.header
                orig_shape = img_nifti.shape  # expected (D,H,W) or (H,W,D) depending on dataset

                # ensure we interpret as (D,H,W). Our dataset used (D,H,W) order.
                pred_vol = pred[0, 0].float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                wantDHW = tuple(int(s) for s in orig_shape[:3])
                if pred_vol.shape[-3:] != wantDHW:
                    pred_vol = F.interpolate(pred_vol, size=wantDHW, mode="trilinear", align_corners=False)
                pred_np = (pred_vol[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)

                nii = nib.Nifti1Image(pred_np, affine=orig_aff, header=orig_hdr)
                nib.save(nii, outdir / f"{case_id}_pred.nii.gz")

        vox_pos = int(pred.sum().item())
        rows.append({"id": case_id, "dice": f"{d:.6f}", "voxels_predicted_1": vox_pos})

        print(f"[{case_id}] dice={d:.4f}  voxels(pred=1)={vox_pos}")

    mean_dice = float(np.mean(dices)) if dices else 0.0
    print(f"[summary] test mean Dice = {mean_dice:.4f}")

    # write CSV
    csv_path = outdir / "test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dice", "voxels_predicted_1"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({"id": "__mean__", "dice": f"{mean_dice:.6f}", "voxels_predicted_1": ""})

    return mean_dice

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint_best.pt")
    ap.add_argument("--outdir", type=str, default="./predictions_unet3d")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--eval_size", nargs=3, type=int, default=[128, 256, 256], help="D H W used by dataset for eval")
    ap.add_argument("--target_label", type=int, default=1)
    ap.add_argument("--no_save", action="store_true", help="Do not save NIfTI predictions")
    ap.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or leave empty for auto")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        ckpt_path=Path(args.ckpt),
        outdir=Path(args.outdir),
        threshold=args.threshold,
        eval_size=tuple(args.eval_size),
        target_label=args.target_label,
        save_nii=not args.no_save,
        device_str=args.device,
    )
