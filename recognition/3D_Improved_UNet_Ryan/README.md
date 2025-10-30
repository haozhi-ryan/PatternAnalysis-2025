# 3D Improved UNet for Prostate MRI Segmentation

## Overview
This project implements a **3D Improved UNet** architecture to segment prostate regions from MRI volumes in the **HipMRI_Study_open** dataset. The goal is to achieve accurate voxel-level segmentation with a **minimum Dice similarity coefficient of 0.7** across all labels. The model addresses the challenge of learning from limited medical data by leveraging **3D convolutions**, **residual connections**, and **data augmentation** to improve spatial consistency and generalization. This segmentation aids in automating prostate boundary identification, an essential step in diagnosis, treatment planning, and volumetric analysis.

---

## How It Works
The model builds upon the original **3D UNet** by incorporating **residual blocks**, **Group Normalization**, and **LeakyReLU** activations for smoother gradient flow and better stability on smaller batch sizes. The encoder progressively downsamples volumetric features, while the decoder upsamples and fuses them with corresponding high-resolution features via skip connections.  
The training pipeline uses **patch-based sampling** for memory efficiency, while inference reconstructs **full 3D volumes**. Dice and BCE-Dice losses guide optimization, ensuring balance between overlap accuracy and boundary precision.

<div align="center">
  <img src="https://raw.githubusercontent.com/mateuszbuda/brain-segmentation-pytorch/master/docs/unet3d_architecture.png" width="600" alt="UNet3D architecture diagram">
  <p><em>Figure 1. Simplified 3D UNet architecture (illustration adapted from UNet3D).</em></p>
</div>

---

## Dependencies
| Package | Version |
|----------|----------|
| Python | 3.10+ |
| PyTorch | 2.2.0 |
| nibabel | 5.2.0 |
| numpy | 1.26.0 |
| argparse | 1.4.0 |
| tqdm | 4.66.0 |

Install all dependencies with:
```bash
pip install torch nibabel numpy tqdm
```

## Dataset & Pre-processing
The dataset consists of 3D NIfTI (.nii/.nii.gz) prostate MRI volumes (semantic_MRs) and their corresponding label masks (semantic_labels_only).
Each volume undergoes z-score intensity normalization to standardize voxel distributions:
```bash
x = (x - x.mean()) / (x.std() + 1e-8)
```
All data are resized to (128, 256, 256) for consistent training dimensions.
We apply geometric augmentations (random flips and 90° rotations) and mild intensity jitter to enhance generalization.

## Data Splits
Splits are performed by patient ID to avoid data leakage:
- 70% Training
- 15% Validation
- 15% Testing

This ensures independent evaluation across patients, improving clinical reliability.

## Usage
### Training
```bash
python train.py --epochs 1 --batch_size 2 --lr 1e-4 --amp
```
Trains the 3D Improved UNet using patch-based sampling and logs validation Dice after each epoch. Best model checkpoints are automatically saved. 1 epoch is enough to produce a Dice coefficient of 0.81.

### Example - training with 40 epochs
```bash
(torch) s4696809@login0:~/report_hard_difficulty$ cat runner_train.out [device] cuda [model] params: 47,325,761 [001] loss=0.2688 val_dice=0.8134 lr=2.995e-04 t=433.11s [002] loss=0.2186 val_dice=0.8395 lr=2.982e-04 t=419.53s [003] loss=0.1635 val_dice=0.8673 lr=2.959e-04 t=389.42s [004] loss=0.1285 val_dice=0.8899 lr=2.927e-04 t=402.64s [005] loss=0.1174 val_dice=0.9074 lr=2.886e-04 t=417.47s [006] loss=0.1039 val_dice=0.9128 lr=2.837e-04 t=421.88s [007] loss=0.0957 val_dice=0.9227 lr=2.779e-04 t=415.84s [008] loss=0.0833 val_dice=0.9327 lr=2.714e-04 t=417.48s [009] loss=0.0768 val_dice=0.9387 lr=2.641e-04 t=400.64s [010] loss=0.0711 val_dice=0.9431 lr=2.561e-04 t=397.76s [011] loss=0.0676 val_dice=0.9426 lr=2.474e-04 t=398.26s [012] loss=0.0706 val_dice=0.9386 lr=2.382e-04 t=403.24s [013] loss=0.0645 val_dice=0.9398 lr=2.284e-04 t=409.86s [014] loss=0.0602 val_dice=0.9476 lr=2.181e-04 t=404.18s [015] loss=0.0546 val_dice=0.9462 lr=2.074e-04 t=414.47s [016] loss=0.0533 val_dice=0.9512 lr=1.964e-04 t=400.37s [017] loss=0.0520 val_dice=0.9490 lr=1.850e-04 t=363.0s
```

## Inference
```bash
python predict.py --ckpt ./checkpoints/best.pt --outdir ./predictions
```
Runs inference on full 3D test volumes and saves predicted segmentation masks as NIfTI files (.nii.gz), preserving original affine and header metadata.

### Example Input & Output
**Input:** 3D MRI volume (downsampled prostate scan)  
**Output:** Binary segmentation mask highlighting the prostate region  
```bash
[model] params: 47,325,761
[J026_Week0_LFOV] dice=0.9539 voxels(pred=1)=3321383
[J026_Week1_LFOV] dice=0.9492 voxels(pred=1)=2857973
[M030_Week5_LFOV] dice=0.9538 voxels(pred=1)=3024402
[N010_Week7_LFOV] dice=0.9641 voxels(pred=1)=3198632
[O025_Week4_LFOV] dice=0.9633 voxels(pred=1)=2985943
[V027_Week2_LFOV] dice=0.9619 voxels(pred=1)=2728240
[summary] test mean Dice = 0.9581
```
## Results
- The 3D Improved UNet3D achieved highly consistent Dice values (0.93–0.97) across multiple weekly scans.
- The model consistently exceeded the required 0.7 Dice threshold across all test cases, demonstrating reliable segmentation performance.
