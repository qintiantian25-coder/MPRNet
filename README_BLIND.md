# MPRNet 红外图像盲元修复

基于 MPRNet（Multi-Stage Progressive Image Restoration，CVPR 2021）的红外图像盲元修复任务。使用原始 MPRNet 架构，不做任何模型修改，仅通过数据加载适配盲元数据集。

---

## 目录

- [项目文件说明](#项目文件说明)
- [环境配置](#环境配置)
- [数据集结构](#数据集结构)
- [模型架构](#模型架构)
- [训练流程](#训练流程)
- [测试流程](#测试流程)
- [配置参数说明](#配置参数说明)
- [输出文件说明](#输出文件说明)

---

## 项目文件说明

本项目在原 MPRNet 基础上新增了以下文件，专用于盲元修复任务：

### 核心入口

| 文件 | 用途 |
|------|------|
| `main.py` | 统一入口，包含 `train()` 训练函数和 `test()` 测试函数 |
| `experiment.yaml` | 总配置文件（YAML 格式），所有超参数、路径、训练策略均在此配置 |
| `config.py` | 配置解析模块，基于 yacs 库，负责读取 YAML 配置文件并提供默认值 |

### 盲元修复专用模块（`blind_pixel/` 目录）

| 文件 | 用途 |
|------|------|
| `blind_pixel/mprnet.py` | **MPRNet 模型定义**，这是原始 MPRNet 论文模型的逐字复制（与 `Deblurring/MPRNet.py` 完全一致），未做任何架构修改。默认配置采用 `n_feat=80`（与 Denoising 相同，适配盲元数据集规模） |
| `blind_pixel/dataset.py` | 数据加载模块，包含训练集、验证集、测试集三个 DataLoader，自动扫描多组目录结构 |
| `blind_pixel/losses.py` | 损失函数模块，包含 CharbonnierLoss、EdgeLoss（拉普拉斯金字塔边缘损失）、BlindMaskWeightedLoss（可选的盲元加权损失） |
| `blind_pixel/utils.py` | 工具函数模块，包含 PSNR/SSIM 计算、checkpoint 加载、盲元坐标 CSV 读取、TestReport 测试报告类 |

### 其他

| 文件 | 用途 |
|------|------|
| `demo.py` | 原始 MPRNet 的 Demo 推理脚本（用于 Deblurring/Deraining/Denoising 三个任务） |
| `README.md` | 原始 MPRNet 项目的英文说明 |
| `CLAUDE.md` | 面向 AI 助手（Claude Code）的代码库说明文档 |
| `pytorch-gradual-warmup-lr/` | 学习率预热调度器（GradualWarmupScheduler） |

> **重点**：本任务使用的模型文件是 `blind_pixel/mprnet.py`，它是原始 MPRNet 架构的完整保留，唯一区别是构造函数默认参数 `n_feat=80`（由 `experiment.yaml` 中的 `N_FEAT: 80` 控制），可根据需要调整。

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n pytorch1 python=3.7
conda activate pytorch1
```

### 2. 安装 PyTorch 及依赖

```bash
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

### 3. 安装学习率预热调度器

```bash
cd pytorch-gradual-warmup-lr
python setup.py install
cd ..
```

### 4. 验证安装

```bash
python -c "from blind_pixel.mprnet import MPRNet; print('环境就绪')"
```

---

## 数据集结构

数据集根目录为 `/home/student_server/Qtt/NAFNet/data_new`，组织方式如下：

```
data_new/
├── train_blur/          # 训练集：退化输入（含盲元噪声）
│   ├── 001/             # 数据组 001（270 帧）
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── 002/             # 数据组 002（270 帧）
│   ├── ...
│   └── 007/             # 数据组 007（59 帧）
│
├── train_sharp/         # 训练集：无噪清晰真值（GT）
│   ├── 001/             # 与 train_blur 文件名完全对应
│   ├── ...
│   └── 007/
│
├── train_mask/          # 训练集：盲元标注信息
│   ├── 001/
│   │   ├── blind_pixel_coords.csv    # 静态盲元坐标（列: x, y, original_gray, simulated_gray）
│   │   ├── flash_pixel_coords.csv    # 帧级闪元记录（列: frame_name, x, y, original_gray, simulated_gray, mode）
│   │   └── blind_pixel_mask.png      # 二值盲元掩码图像（与输入同分辨率）
│   ├── ...
│   └── 007/
│
├── val_blur/            # 验证集：退化输入（2 组）
│   ├── 001/
│   └── 002/
│
├── val_sharp/           # 验证集：清晰真值（2 组）
│
├── val_mask/            # 验证集：盲元标注（2 组）
│
├── test_blur/           # 测试集：退化输入（6 组）
│   ├── 001/
│   ├── ...
│   └── 006/
│
├── test_sharp/          # 测试集：清晰真值（6 组）
│
└── test_mask/           # 测试集：盲元标注（6 组）
```

**数据规格**：

- 图像尺寸：640 × 512 像素
- 图像格式：PNG（无损压缩）
- 颜色模式：RGB（三通道，灰度内容 R=G=B）
- 训练集共 7 组约 885 张图像
- 各组内图像按时间序列连续编号，blur 与 sharp 文件名一一对应

---

## 模型架构

本任务使用 **原始 MPRNet 模型**（`blind_pixel/mprnet.py`），未做任何架构修改。

### MPRNet 核心设计

MPRNet 是一个三阶段渐进式图像复原网络：

```
输入 (3, H, W)
    │
    ├─ 浅层特征提取 ──────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
阶段一：4 个互不重叠的图块 → U-Net 编码器-解码器            │
    │   （每个图块为原图的 1/4）                             │
    │   编码器: Encoder (csff=False)                        │
    │   解码器: Decoder                                      │
    │                                                       │
    ├─ SAM → 中间复原图 + 注意力特征 ──────────────────────┤
    │                                                       │
    ▼                                                       │
阶段二：2 个图块（上下各半）→ U-Net 编码器-解码器           │
    │   编码器: Encoder (csff=True) ← CSFF 融合阶段一的特征  │
    │   解码器: Decoder                                      │
    │                                                       │
    ├─ SAM → 中间复原图 + 注意力特征 ──────────────────────┤
    │                                                       │
    ▼                                                       │
阶段三：全分辨率输入 → ORSNet（原分辨率子网络）             │
    │   融合阶段二的编码器/解码器特征                         │
    │   残差连接: 输出 = ORSNet(x) + 原始输入               │
    │                                                       │
    ▼                                                       │
输出: [阶段三结果, 阶段二结果, 阶段一结果]
```

### 关键模块

| 模块 | 说明 |
|------|------|
| **CALayer** | 通道注意力层：全局平均池化 → 2 层 MLP → Sigmoid 门控 |
| **CAB** | 通道注意力块：Conv → Act → Conv → CALayer → 残差连接 |
| **SAM** | 监督注意力模块：生成中间复原图像 + 注意力调制特征（保持原始 3 通道输出设计） |
| **ORB** | 原分辨率块：堆叠 num_cab 个 CAB + 末端卷积 + 残差连接 |
| **ORSNet** | 原分辨率子网络：3 个 ORB，逐级注入来自阶段二的编码器/解码器特征 |
| **CSFF** | 跨阶段特征融合：通过 1×1 卷积将前一阶段的编码器和解码器输出注入当前阶段编码器 |

### 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `in_c` | 3 | 输入通道（RGB） |
| `out_c` | 3 | 输出通道（RGB） |
| `n_feat` | 80 | 特征通道数（与 Denoising 相同，适配 ~885 张训练图规模） |
| `scale_unetfeats` | 48 | U-Net 下采样时的通道增量 |
| `scale_orsnetfeats` | 32 | ORSNet 的通道增量 |
| `num_cab` | 8 | 每个 ORB 中的 CAB 数量 |
| 总参数量 | **15,741,025** | 约 1573 万参数 |

---

## 训练流程

### 训练命令

```bash
python main.py --train --config_path ./experiment.yaml
```

### 训练过程详解

1. **读取配置**：从 `experiment.yaml` 加载所有超参数
2. **初始化模型**：创建 MPRNet 实例（`in_c=3, out_c=3, n_feat=80`），移至 GPU
3. **加载数据**：
   - 训练集：扫描 `train_blur/` 和 `train_sharp/` 下所有数据组的配对图像
   - 验证集：扫描 `val_blur/` 和 `val_sharp/` 下所有数据组的配对图像
4. **数据增强**：每张训练图随机裁剪 256×256 图块 + 8 向翻转/旋转（共 8 种可能）
5. **损失函数**：
   - `CharbonnierLoss`（平滑 L1 损失）
   - `+ 0.05 × EdgeLoss`（拉普拉斯金字塔边缘损失）
   - 两个损失均施加于模型输出的**全部三个阶段**（deep supervision）
6. **优化策略**：
   - 优化器：Adam（`betas=0.9, 0.999, eps=1e-8`）
   - 学习率调度：前 3 轮线性预热 → 余弦退火（`2e-4 → 1e-6`）
7. **验证策略**：
   - 每 **20 轮**（由 `VAL_AFTER_EVERY` 控制）在验证集上评估一次
   - 计算 PSNR 和 SSIM（计算方式与测试阶段完全一致：numpy uint8, data_range=255）
   - 验证图块大小：256×256（中心裁剪）
8. **模型保存**：
   - **只保存一个最佳模型**：`experiments/models/best_model.pt`
   - 当 PSNR 超过历史最佳时**自动覆盖更新**
   - 不保存每个 epoch 的历史模型
9. **日志记录**：
   - 训练日志：`experiments/logs/training.txt`（每轮记录 epoch、loss、耗时、学习率）
   - 验证日志：`experiments/logs/validation.txt`（记录每次验证的 PSNR/SSIM 及模型更新提示）
10. **断点续训**：设置 `RESUME: True` 后，从 `best_model.pt` 恢复训练（含模型权重、优化器状态、epoch 计数）

### 训练过程终端输出示例

```
===> Start Epoch 1 End Epoch 3001
===> Loading datasets
Epoch 1: 100%|████████| 56/56 [00:45<00:00]
[2025-06-03 10:00:45] Epoch [1/3000]  Loss: 1.2345  Time: 45.23s  LR: 0.00020000
[2025-06-03 10:01:30] >>> 模型已更新! Epoch [20]  PSNR: 34.5678 dB  SSIM: 0.9123  (Best at epoch 20)
...
[2025-06-03 15:30:00] >>> 模型已更新! Epoch [500]  PSNR: 38.1234 dB  SSIM: 0.9567  (Best at epoch 500)
```

---

## 测试流程

### 测试命令

```bash
python main.py --test --config_path ./experiment.yaml
```

### 测试过程详解

1. **加载模型**：从 `experiments/models/best_model.pt` 加载训练好的最佳模型
2. **加载数据**：扫描 `test_blur/` 下的 6 个数据组，共若干张测试图像
3. **逐张推理**：
   - 加载测试图像（RGB 格式）
   - 自动 Padding 至 8 的倍数（模型要求）
   - 前向传播，取 `output[0]`（阶段三的最终输出）
   - 去除 Padding 恢复原始尺寸
   - 保存修复后的纯净图像
4. **计算指标**：
   - **全图 PSNR / SSIM**：在整张图像上与 Ground Truth 比较
   - **盲元聚焦指标**：仅在盲元像素位置计算 MAE / RMSE / PSNR，以及输入图像的对应指标，得出模型在盲元区域的修复增益
5. **盲元坐标来源**：
   - 静态盲元：`test_mask/<组名>/blind_pixel_coords.csv`
   - 帧级闪元：`test_mask/<组名>/flash_pixel_coords.csv`
   - 两者合并去重后作为盲元评估像素集

### 测试输出文件

测试结果保存在 `./results/` 目录下：

```
results/
├── test/                              # 修复后的纯净图像
│   ├── 001/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── ...
│
├── triple_comparison/                 # 三连对比图（输入｜输出｜真值）
│   ├── 001/
│   │   ├── triple_1.png
│   │   └── ...
│   └── ...
│
└── blind_eval/                        # 盲元评估 CSV 报告
    ├── test_blind_metrics.csv         # 逐张图像指标明细
    └── test_blind_summary_by_seq.csv  # 按数据组汇总的指标
```

### 指标说明

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **PSNR** | `20 × log10(255 / √MSE)` | 全图峰值信噪比，越高越好 |
| **SSIM** | skimage 结构相似度 | 全图感知质量，越接近 1 越好 |
| **Blind MAE** | 盲元位置 `|out - gt|` 均值 | 盲元区域的平均绝对误差 |
| **Blind RMSE** | 盲元位置 `√(mean((out - gt)²))` | 盲元区域的均方根误差 |
| **Blind PSNR** | `10 × log10(255² / MSE_blind)` | 盲元区域的峰值信噪比 |
| **MAE Gain** | `Input_MAE - Output_MAE` | 盲元 MAE 的绝对改善量 |
| **MAE Gain %** | `MAE_Gain / Input_MAE × 100%` | 盲元 MAE 的相对改善百分比 |

### 测试终端输出示例

```
===> Testing using weights: ./experiments/models/best_model.pt
===> Starting inference on 200 test images...
100%|████████████| 200/200 [00:30<00:00]

============================================================
Full-image PSNR: 38.1234 dB  |  SSIM: 0.9567
============================================================

===> Blind-Pixel Focused Metrics
Blind MAE: 2.345678 | Blind RMSE: 8.901234 | Blind PSNR: 29.123
Input Blind MAE: 12.345678 | Input Blind PSNR: 22.456 | MAE Gain: 10.000000 (81.00%)

Per-image test metrics saved to: ./results/blind_eval/test_blind_metrics.csv
Per-seq summary saved to: ./results/blind_eval/test_blind_summary_by_seq.csv
```

---

## 配置参数说明

`experiment.yaml` 中所有可调参数：

### GPU 设置

```yaml
GPU: [0]              # 使用的 GPU 编号列表，多卡如 [0,1,2,3]
```

### 模型参数

```yaml
MODEL:
  IN_C: 3             # 输入通道数（固定为 3，RGB）
  OUT_C: 3            # 输出通道数（固定为 3，RGB）
  N_FEAT: 80          # 特征通道数（越大模型越大，80=1573万参数，96=2012万参数）
  SCALE_UNETFEATS: 48 # U-Net 下采样通道增量
  SCALE_ORSNETFEATS: 32 # ORSNet 通道增量
  NUM_CAB: 8          # 每个 ORB 中 CAB 的数量
```

### 优化器参数

```yaml
OPTIM:
  BATCH_SIZE: 16      # 训练批次大小
  VAL_BATCH_SIZE: 8   # 验证批次大小
  NUM_EPOCHS: 3000    # 总训练轮数
  WARMUP_EPOCHS: 3    # 学习率预热轮数
  LR_INITIAL: 2e-4    # 初始学习率
  LR_MIN: 1e-6        # 余弦退火最低学习率
```

### 训练参数

```yaml
TRAINING:
  VAL_AFTER_EVERY: 20 # 每 N 轮验证一次
  RESUME: False       # 是否从 best_model.pt 断点续训
  TRAIN_PS: 256       # 训练随机裁剪大小
  VAL_PS: 256         # 验证中心裁剪大小
  DATA_ROOT: '/home/student_server/Qtt/NAFNet/data_new'  # 数据集根目录
  SAVE_DIR: './experiments'  # 模型和日志保存目录
  NUM_WORKERS: 8      # 训练数据加载线程数
  NUM_WORKERS_VAL: 4  # 验证数据加载线程数
  EDGE_LOSS_WEIGHT: 0.05  # EdgeLoss 权重
  USE_MASK_WEIGHT: False   # 是否启用盲元加权损失（实验功能）
  MASK_BLIND_WEIGHT: 5.0   # 盲元加权损失的权重倍数
```

### 测试参数

```yaml
TEST:
  CHECKPOINT: './experiments/models/best_model.pt'  # 测试用的模型路径
  SAVE_DIR: './results'   # 测试结果保存目录
  IMAGE_BORDER: 0         # 评估时裁掉的边界像素数
```

---

## 完整工作流示例

### 首次训练

```bash
# 1. 激活环境
conda activate pytorch1

# 2. 确认配置文件中的路径正确（DATA_ROOT 和 SAVE_DIR）

# 3. 开始训练
python main.py --train --config_path ./experiment.yaml

# 4. 训练完成后，模型保存在 experiments/models/best_model.pt
```

### 断点续训

```bash
# 修改 experiment.yaml 中 TRAINING.RESUME 为 True，然后：
python main.py --train --config_path ./experiment.yaml
```

### 测试评估

```bash
# 确保 experiment.yaml 中 TEST.CHECKPOINT 指向正确的模型
python main.py --test --config_path ./experiment.yaml

# 查看结果
# - 修复图像: results/test/
# - 对比图像: results/triple_comparison/
# - 指标报告: results/blind_eval/test_blind_metrics.csv
```

---

## 文件依赖关系图

```
main.py
  ├── config.py  ←  experiment.yaml（用户配置）
  ├── blind_pixel/mprnet.py       ← MPRNet 模型定义（原始架构，未修改）
  ├── blind_pixel/dataset.py      ← 数据加载（扫描多组目录，RGB 直读）
  ├── blind_pixel/losses.py       ← 损失函数（Charbonnier + EdgeLoss）
  ├── blind_pixel/utils.py        ← 工具函数（PSNR/SSIM/checkpoint/TestReport）
  └── pytorch-gradual-warmup-lr/  ← 学习率预热调度器

训练产生：
  experiments/
    ├── models/best_model.pt      ← 供测试使用的唯一模型文件
    └── logs/
         ├── training.txt
         └── validation.txt

测试产生：
  results/
    ├── test/                     ← 修复图像
    ├── triple_comparison/        ← 对比图像
    └── blind_eval/               ← 指标 CSV

数据集（外部只读）：
  /home/student_server/Qtt/NAFNet/data_new/
    ├── train_blur/ + train_sharp/ + train_mask/
    ├── val_blur/   + val_sharp/   + val_mask/
    └── test_blur/  + test_sharp/  + test_mask/
```
