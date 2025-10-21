# HipMRI Prostate Segmentation (2D Improved U-Net)

Segmented the HipMRI Study prostate label from processed **2D NIfTI slices** using an **Improved U-Net** (InstanceNorm + LeakyReLU, strided-conv downs, transposed-conv ups) with **BCE+Dice** loss. Achieved **Test Dice = 0.7954** (≥ 0.75 target).

---

## Project Structure
report/
├─ dataset.py # NIfTI loaders, pairing case_* with seg_*; z-score; DataLoaders
├─ modules.py # 2D Improved U-Net + Dice metrics/losses; build_model()
├─ train.py # train/val loop, checkpointing, test eval
├─ predict.py # inference on test set, per-slice Dice, overlays + CSV
├─ checkpoints/ # saved model (best.ckpt)
├─ preds/ # outputs from predict.py (created on first run)
└─ README.md
