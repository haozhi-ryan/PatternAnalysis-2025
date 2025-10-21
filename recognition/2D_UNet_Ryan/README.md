# HipMRI Prostate Segmentation (2D Improved U-Net)

Segmented the HipMRI Study prostate label from processed **2D NIfTI slices** using an **Improved U-Net** (InstanceNorm + LeakyReLU, strided-conv downs, transposed-conv ups) with **BCE+Dice** loss. Achieved **Test Dice = 0.7954** (≥ 0.75 target).

---

## Project Structure
```
report/
├─ dataset.py # NIfTI loaders, pairing case_* with seg_*; z-score; DataLoaders
├─ modules.py # 2D Improved U-Net + Dice metrics/losses; build_model()
├─ train.py # train/val loop, checkpointing, test eval
├─ predict.py # inference on test set, per-slice Dice, overlays + CSV
├─ checkpoints/ # saved model (best.ckpt)
├─ preds/ # outputs from predict.py (created on first run)
└─ README.md
```
---

## Environment
```bash
# login to Rangpur, then:
conda activate torch
```

## Data
```bash
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/
  ├─ keras_slices_train/          # images
  ├─ keras_slices_validate/       # images
  ├─ keras_slices_test/           # images
  ├─ keras_slices_seg_train/      # masks
  ├─ keras_slices_seg_validate/   # masks
  └─ keras_slices_seg_test/       # masks
```
dataset.py automatically pairs image/mask files, handling naming differences (case_* ↔ seg_*).
Set the prostate label ID via --target_label (commonly 1).

## How to Run (from ~/report)
Ensure you’re in the right environment:
(torch) s4696809@login0:~/report$

### Train
Sanity overfit
```bash
python train.py --overfit_batches 2 --epochs 10 --batch_size 4 --num_workers 0 --target_label 1
```
Full training (example)
```bash
python train.py --epochs 40 --batch_size 16 --target_label 1
```
Outputs:
- checkpoints saved to checkpoints/best.ckpt
- per-epoch logs (train/val Dice)

### Predict / Visualize
```bash
python predict.py --target_label 1 --checkpoint checkpoints/best.ckpt --outdir preds
```
Outputs:

preds/summary.txt (mean Dice)

preds/metrics.csv (per-slice Dice)

preds/overlays/*_overlay.png (image + GT + prediction)

preds/pred_masks/*_pred.npy (binary masks)

Results

Validation best Dice: 0.8259

Test mean Dice (prostate): 0.7954

See qualitative overlays in preds/overlays/.

## Training Details

Model: 2D Improved U-Net (InstanceNorm + LeakyReLU, strided-conv down, transposed-conv up)

Loss: BCEWithLogits + Dice (0.5 weight)

Optimizer: AdamW (lr=1e-3, wd=1e-4)

Scheduler: ReduceLROnPlateau (monitor val Dice)

Inputs: single-channel slices (z-score normalized)

Binary training on prostate label (--target_label 1)
