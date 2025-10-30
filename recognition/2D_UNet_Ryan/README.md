# HipMRI Prostate Segmentation (2D Improved U-Net)
# Task 3 - Normal Difficulty

Segmented the HipMRI Study prostate label from processed **2D NIfTI slices** using an **Improved U-Net** (InstanceNorm + LeakyReLU, strided-conv downs, transposed-conv ups) with **BCE+Dice** loss. Achieved **Test Dice = 0.7954** (≥ 0.75 target).

## Overview — Problem & Algorithm
Prostate cancer diagnosis and treatment planning rely on accurate localisation of the prostate gland in MRI scans. Manual segmentation by radiologists is time-consuming, inconsistent between experts, and difficult to scale across thousands of images. This project automates that task using the **HipMRI Study**’s processed **2D NIfTI slices**, where each slice is treated as a single-channel image. The segmentation is performed with an **Improved 2D U-Net**, an encoder–decoder convolutional network designed for biomedical image segmentation. It extends the classic U-Net by replacing BatchNorm with **Instance Normalisation** (for robustness to small batch sizes and scanner intensity variations) and by using **LeakyReLU** activations to prevent dead neurons. The network is trained using a combined **Binary Cross-Entropy + Dice** loss to penalise both pixel-wise classification errors and region-overlap mismatches. The goal is to achieve a **minimum Dice coefficient of 0.75** on the prostate label of the held-out test set, which the model surpassed with a final **Test Dice = 0.7954**.

## How It Works — Architecture & Training
The 2D Improved U-Net follows a symmetric **encoder–decoder** architecture.  
- **Encoder:** successive 3×3 convolutions with stride 2 progressively downsample the feature maps while increasing the number of channels, extracting semantic information from the input slices.  
- **Decoder:** **transposed convolutions** upsample these features, restoring spatial resolution. Skip connections concatenate encoder and decoder features at each depth, preserving fine boundary details that are crucial for organ contours.  
- **Normalisation & Activation:** **InstanceNorm** normalises each slice independently, reducing contrast variability between patients, while **LeakyReLU** maintains gradient flow even for negative activations.  
- **Loss & Optimisation:** the **BCE + Dice** composite loss balances class imbalance and shape accuracy; training uses **AdamW** with learning-rate scheduling.  
- **Pre-processing:** each image undergoes **z-score intensity normalisation** to zero-centre voxel values, and all slices are resized to a consistent spatial resolution.  
- **Data Splits:** we adopt the predefined `keras_slices_*` train/validate/test folders supplied with the HipMRI dataset to ensure consistent, non-overlapping patient splits and reproducible evaluation.  

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
### Full training 
Input:
```bash
python train.py --epochs 40 --batch_size 16 --target_label 1
```
Outputs:
- checkpoints saved to checkpoints/best.ckpt
- per-epoch logs (train/val Dice)
  
Example Output:
```bash
Device: cuda
Epoch 001 | loss 0.2144 | trainDice 0.8677 | valDice 0.9310 | 78.4s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9310)
Epoch 002 | loss 0.1320 | trainDice 0.9233 | valDice 0.9570 | 69.5s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9570)
Epoch 003 | loss 0.1071 | trainDice 0.9376 | valDice 0.9674 | 64.0s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9674)
Epoch 004 | loss 0.0938 | trainDice 0.9452 | valDice 0.9673 | 64.7s
Epoch 005 | loss 0.0855 | trainDice 0.9499 | valDice 0.9694 | 64.6s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9694)
Epoch 006 | loss 0.0799 | trainDice 0.9532 | valDice 0.9700 | 63.9s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9700)
Epoch 007 | loss 0.0746 | trainDice 0.9562 | valDice 0.9722 | 63.2s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9722)
Epoch 008 | loss 0.0717 | trainDice 0.9579 | valDice 0.9716 | 64.1s
Epoch 009 | loss 0.0677 | trainDice 0.9602 | valDice 0.9748 | 63.5s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9748)
Epoch 010 | loss 0.0657 | trainDice 0.9614 | valDice 0.9737 | 64.6s
Epoch 011 | loss 0.0630 | trainDice 0.9630 | valDice 0.9733 | 63.5s
Epoch 012 | loss 0.0605 | trainDice 0.9644 | valDice 0.9757 | 64.0s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9757)
Epoch 013 | loss 0.0586 | trainDice 0.9655 | valDice 0.9754 | 63.5s
Epoch 014 | loss 0.0572 | trainDice 0.9663 | valDice 0.9753 | 64.4s
Epoch 015 | loss 0.0556 | trainDice 0.9673 | valDice 0.9761 | 64.4s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9761)
Epoch 016 | loss 0.0539 | trainDice 0.9682 | valDice 0.9746 | 64.7s
Epoch 017 | loss 0.0531 | trainDice 0.9687 | valDice 0.9760 | 64.6s
Epoch 018 | loss 0.0518 | trainDice 0.9694 | valDice 0.9755 | 65.6s
Epoch 019 | loss 0.0521 | trainDice 0.9694 | valDice 0.9762 | 64.7s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9762)
Epoch 020 | loss 0.0458 | trainDice 0.9730 | valDice 0.9766 | 64.8s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9766)
Epoch 021 | loss 0.0446 | trainDice 0.9737 | valDice 0.9759 | 64.3s
Epoch 022 | loss 0.0439 | trainDice 0.9741 | valDice 0.9763 | 64.5s
Epoch 023 | loss 0.0433 | trainDice 0.9744 | valDice 0.9764 | 65.0s
Epoch 024 | loss 0.0425 | trainDice 0.9749 | valDice 0.9763 | 63.1s
Epoch 025 | loss 0.0404 | trainDice 0.9761 | valDice 0.9763 | 63.7s
Epoch 026 | loss 0.0397 | trainDice 0.9765 | valDice 0.9767 | 63.8s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9767)
Epoch 027 | loss 0.0393 | trainDice 0.9768 | valDice 0.9773 | 65.6s
  ✓ Saved best checkpoint to checkpoints/best.ckpt (valDice=0.9773)
Epoch 028 | loss 0.0389 | trainDice 0.9770 | valDice 0.9768 | 62.3s
Epoch 029 | loss 0.0386 | trainDice 0.9772 | valDice 0.9769 | 63.5s
Epoch 030 | loss 0.0385 | trainDice 0.9773 | valDice 0.9768 | 74.2s
Epoch 031 | loss 0.0380 | trainDice 0.9776 | valDice 0.9768 | 66.0s
Epoch 032 | loss 0.0371 | trainDice 0.9781 | valDice 0.9768 | 64.5s
Epoch 033 | loss 0.0365 | trainDice 0.9784 | valDice 0.9772 | 65.9s
Epoch 034 | loss 0.0365 | trainDice 0.9784 | valDice 0.9770 | 65.0s
Epoch 035 | loss 0.0364 | trainDice 0.9785 | valDice 0.9773 | 66.5s
Epoch 036 | loss 0.0358 | trainDice 0.9788 | valDice 0.9772 | 65.7s
Epoch 037 | loss 0.0357 | trainDice 0.9789 | valDice 0.9772 | 65.9s
Epoch 038 | loss 0.0356 | trainDice 0.9789 | valDice 0.9772 | 64.6s
Epoch 039 | loss 0.0355 | trainDice 0.9790 | valDice 0.9769 | 70.2s
Epoch 040 | loss 0.0351 | trainDice 0.9793 | valDice 0.9772 | 63.6s
Loaded best checkpoint (valDice=0.9773)
TEST Dice (prostate): 0.9817
```

### Predict / Visualize
```bash
python predict.py --target_label 1 --checkpoint checkpoints/best.ckpt --outdir preds
```
Outputs:
- `preds/summary.txt` (mean Dice)
- `preds/metrics.csv` (per-slice Dice)
- `preds/overlays/*_overlay.png` (image + GT + prediction)
- `preds/pred_masks/*_pred.npy` (binary masks)

Example Output:
```bash
Device: cpu
Loaded checkpoint: checkpoints/best.ckpt (valDice=0.9773)
/home/Student/s4696809/miniconda3/envs/torch/lib/python3.13/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Test mean Dice: 0.9817
```

Results
- Validation best Dice: 0.8259
- Test mean Dice (prostate): 0.7954

## Training Details
- Model: 2D Improved U-Net (InstanceNorm + LeakyReLU, strided-conv down, transposed-conv up)
- Loss: BCEWithLogits + Dice (0.5 weight)
- Optimizer: AdamW (lr=1e-3, wd=1e-4)
- Scheduler: ReduceLROnPlateau (monitor val Dice)
- Inputs: single-channel slices (z-score normalized)
- Binary training on prostate label (--target_label 1)
