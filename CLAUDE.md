# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MPRNet (Multi-Stage Progressive Image Restoration) — CVPR 2021. A multi-stage CNN that progressively restores degraded images across three tasks: **Deblurring**, **Deraining**, and **Denoising**.

Each task lives in its own subdirectory (`Deblurring/`, `Deraining/`, `Denoising/`) with **copy-pasted, task-specific code** — there is no shared library between them. The model architecture is nearly identical across tasks but differs in feature dimensions (`n_feat`, `scale_unetfeats`, `scale_orsnetfeats`).

## Commands

### Setup

```bash
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
cd pytorch-gradual-warmup-lr && python setup.py install && cd ..
```

### Demo (quick inference with pretrained models)

```bash
python demo.py --task Deblurring --input_dir ./samples/input/ --result_dir ./samples/output/
```

`--task` choices: `Deblurring`, `Deraining`, `Denoising`. Downloads pretrained weights from Google Drive (URLs in README) to `<Task>/pretrained_models/model_<task>.pth`.

### Training

From within a task directory (e.g., `Deblurring/`):

```bash
python train.py
```

All configuration is in `<task>/training.yml` (GPU IDs, batch size, epochs, LR, data paths, patch sizes). The `config.py` file loads defaults from `yacs` and overrides from the YAML.

### Testing

```bash
python test.py --input_dir ./Datasets/ --result_dir ./results/ --weights ./pretrained_models/model_deblurring.pth --dataset GoPro
```

Some tasks have additional evaluation scripts: `evaluate_PSNR_SSIM.m` (MATLAB), `evaluate_RealBlur.py`, `evaluate_GOPRO_HIDE.m`.

## Architecture

### Three-Stage Progressive Design

1. **Stage 1** — Input split into 4 non-overlapping patches → U-Net encoder-decoder on each → merge patches back → SAM produces intermediate image and attention features for Stage 2
2. **Stage 2** — Input split into 2 patches (top/bottom) → U-Net encoder-decoder with **CSFF** (Cross Stage Feature Fusion) receiving encoder/decoder features from Stage 1 → SAM produces intermediate image and attention features for Stage 3
3. **Stage 3** — Full-resolution input → **ORSNet** (Original Resolution Subnetwork) fusing features from Stage 2 encoder/decoder → final output via residual connection (`stage3_img + x3_img`)

Output: `[stage3_result, stage2_result, stage1_result]` — a list of three tensors. Only the first (`[0]`) is the final output; the others are intermediate outputs used for deep supervision during training.

### Key Building Blocks (defined in `MPRNet.py`)

- **CALayer** — Channel attention via global average pooling + 2-layer MLP with sigmoid gating
- **CAB** (Channel Attention Block) — Conv → Act → Conv → CALayer → residual
- **SAM** (Supervised Attention Module) — Generates both an intermediate restoration image and attention-modulated features for the next stage
- **ORB** (Original Resolution Block) — Stack of `num_cab` CABs → conv → residual
- **ORSNet** — 3 ORBs with upsample-and-inject connections from encoder/decoder features of Stage 2
- **CSFF** (Cross Stage Feature Fusion) — In Stage 2's encoder, adds 1×1 conv projections of previous stage's encoder and decoder outputs at each U-Net level

### Per-Task Differences

| Hyperparameter | Deblurring | Deraining | Denoising |
|---|---|---|---|
| `n_feat` | 96 | 40 | 80 |
| `scale_unetfeats` | 48 | 20 | 48 |
| `scale_orsnetfeats` | 32 | 16 | 32 |
| EdgeLoss weight | 0.05 | 0.05 | Not used |
| Adam weight_decay | 0 | 0 | 1e-8 |
| MixUp augmentation | No | No | Yes (epoch > 5) |
| Eval frequency | Every N epochs | Every N epochs | Every 1/3 of epoch (certain epochs) |

### Data Pipeline

- **Dataset directory layout**: `<data_dir>/input/` and `<data_dir>/target/` — paired degraded/clean images with matching sorted filenames
- **Training**: Random crop to `TRAIN_PS` (specified in YAML), random flips/rotations (8 augmentations), optional gamma/saturation jitter
- **Validation**: Center crop to `VAL_PS`
- **`dataset_RGB.py`** — Three `Dataset` classes: `DataLoaderTrain`, `DataLoaderVal`, `DataLoaderTest`
- **`data_RGB.py`** — Thin factory: `get_training_data()`, `get_validation_data()`, `get_test_data()`

### Training Details

- **Optimizer**: Adam (betas=0.9, 0.999, eps=1e-8)
- **Scheduler**: 3-epoch linear warmup → CosineAnnealingLR
- **Loss**: CharbonnierLoss (smooth L1) summed across all 3 stage outputs + optionally 0.05 × EdgeLoss (Laplacian pyramid edge loss)
- **Multi-GPU**: `nn.DataParallel`
- **Checkpoints**: `model_best.pth` (best PSNR), `model_latest.pth` (most recent), `model_epoch_<N>.pth` (periodic)
- **Resume**: Set `TRAINING.RESUME: True` in YAML; looks for `_latest.pth`

### Utilities (`utils/`)

- `model_utils.py` — Checkpoint save/load, freeze/unfreeze, handles `module.` prefix stripping from DataParallel
- `image_utils.py` — `torchPSNR()` (tensor PSNR, values 0-1), `numpyPSNR()` (numpy PSNR, values 0-255), `save_img()` (OpenCV RGB→BGR write)
- `dir_utils.py` — `mkdir()`, `mkdirs()`, `get_last_path()` (finds latest checkpoint by natural sort)
- `dataset_utils.py` — `MixUp_AUG` class (used only by Denoising)

## Blind Pixel Restoration (new adaptation)

A fourth task adapting MPRNet for **grayscale blind pixel restoration**. Uses a unified entry point at the project root.

### Commands

```bash
# Training (saves best model to experiments/models/best_model.pt)
python main.py --train --config_path ./experiment.yaml

# Testing (generates metrics CSV, triple-comparison images, per-group blind stats)
python main.py --test --config_path ./experiment.yaml
```

### Training outputs

```
experiments/
├── models/
│   └── best_model.pt          # 只保存 PSNR 最高的一个模型，自动覆盖
└── logs/
    ├── training.txt            # 每轮记录: epoch, loss, time, lr
    └── validation.txt          # 每次验证记录: epoch, PSNR, SSIM, 模型更新提示
```

- 验证频率由 `TRAINING.VAL_AFTER_EVERY` 控制（默认每 20 轮）
- 验证时计算 PSNR 和 SSIM（与测试代码一致的计算方式）
- 当 PSNR 超过历史最佳时自动覆盖保存 `best_model.pt`，并打印更新提示

### Files (root-level, added for this task)

| File | Purpose |
|------|---------|
| `main.py` | Unified entry: `train()` and `test()` functions |
| `config.py` | yacs Config class (copied from Deblurring, extended with blind-pixel defaults) |
| `experiment.yaml` | YAML config with all hyperparameters |
| `blind_pixel/mprnet.py` | Adapted MPRNet for grayscale: `in_c=1, out_c=1`, SAM parameterized by `out_c` |
| `blind_pixel/dataset.py` | DataLoaderTrain/Val/Test scanning multi-group dirs (train_blur/, train_sharp/, etc.) |
| `blind_pixel/losses.py` | CharbonnierLoss, EdgeLoss (channel-aware), optional BlindMaskWeightedLoss |
| `blind_pixel/utils.py` | TestReport, blind-coord CSV loaders, PSNR/SSIM, checkpoint helpers |

### Key differences from original MPRNet tasks

- **模型完全保持原样**：`in_c=3, out_c=3, n_feat=80`（与 Denoising 相同，适配 ~885 张训练图的规模）。SAM 的 `conv2`/`conv3` 保持原始 3 通道设计，所有架构代码未做任何修改
- **数据集图像**：原始即为 RGB 640×512 格式，直接加载无需任何转换，与 MPRNet 的 3 通道输入完全匹配
- **EdgeLoss 保持原样**：`.repeat(3,1,1,1)` 3 通道核，与原版一致
- **数据集结构**：多组目录布局（`train_blur/001/1.png` 等），而非原始的扁平 `input/` / `target/` 结构
- **测试指标**：除 PSNR/SSIM 外，基于 `test_mask/<group>/blind_pixel_coords.csv` 和 `flash_pixel_coords.csv` 计算 `blind_mae`、`blind_rmse`、`blind_psnr`
- **数据根目录**：`TRAINING.DATA_ROOT` 指向 `/home/student_server/Qtt/NAFNet/data_new`

### Dataset layout (expected at `DATA_ROOT`)

```
data_new/
├── train_blur/   (groups 001-007, sequential grayscale PNGs)
├── train_sharp/  (groups 001-007, matching ground truth)
├── train_mask/   (groups 001-007, blind_coords.csv + flash_pixel_coords.csv + blind_pixel_mask.png)
├── val_blur/     (groups 001-002)
├── val_sharp/    (groups 001-002)
├── val_mask/     (groups 001-002)
├── test_blur/    (groups 001-006)
├── test_sharp/   (groups 001-006)
└── test_mask/    (groups 001-006)
```

All images are 640×512 grayscale PNG. Mask CSVs contain `x,y` columns (blind_coords) and `frame_name,x,y` columns (flash_pixel_coords). Only `blind_pixel_mask.png` is not actively used by the default pipeline; blind coords CSV provides the same information more efficiently.

### Test output structure

```
results/
├── test/                     # Restored images (replicates test_blur/ structure)
├── triple_comparison/        # Input|Output|GT concatenated for visual inspection
└── blind_eval/
    ├── test_blind_metrics.csv           # Global per-image metrics
    ├── test_blind_summary_by_seq.csv    # Global per-group summary
    ├── 001/
    │   ├── test_blind_metrics_001.csv   # Group 001 per-image metrics
    │   └── test_blind_summary_001.csv   # Group 001 summary
    ├── 002/
    │   └── ...
    └── ...
```

## License

Academic non-commercial use only. See `LICENSE.md`. Commercial use requires contacting the authors.
