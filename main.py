#!/usr/bin/env python3
"""
MPRNet Blind Pixel Restoration
Usage:
    python main.py --train --config_path ./experiment.yaml
    python main.py --test  --config_path ./experiment.yaml
"""

import os
import sys
import re
import csv
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Add warmup scheduler to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-gradual-warmup-lr'))

import warnings
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler\.step.*before.*optimizer\.step')

from config import Config
from blind_pixel.mprnet import MPRNet
from blind_pixel.dataset import get_training_data, get_validation_data, get_test_data
from blind_pixel.losses import CharbonnierLoss, EdgeLoss
from blind_pixel.utils import (
    mkdir, mkdirs, torchPSNR,
    calculate_psnr, calculate_ssim,
    load_checkpoint, load_start_epoch, load_optim,
    load_blind_coords, load_flash_map, natural_sort_key,
    TestReport,
)
from warmup_scheduler import GradualWarmupScheduler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path):
    opt = Config(config_path)

    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    torch.backends.cudnn.benchmark = True

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    save_dir = opt.TRAINING.SAVE_DIR
    model_dir = os.path.join(save_dir, 'models')
    log_dir = os.path.join(save_dir, 'logs')
    mkdir(model_dir)
    mkdir(log_dir)

    model_path = os.path.join(model_dir, 'best_model.pt')
    training_log_path = os.path.join(log_dir, 'training.txt')
    validation_log_path = os.path.join(log_dir, 'validation.txt')

    ######### Logging helpers ###########
    def log_training(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(msg)
        with open(training_log_path, 'a') as f:
            f.write(line + '\n')

    def log_validation(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(msg)
        with open(validation_log_path, 'a') as f:
            f.write(line + '\n')

    ######### Model ###########
    model_restoration = MPRNet(
        in_c=opt.MODEL.IN_C,
        out_c=opt.MODEL.OUT_C,
        n_feat=opt.MODEL.N_FEAT,
        scale_unetfeats=opt.MODEL.SCALE_UNETFEATS,
        scale_orsnetfeats=opt.MODEL.SCALE_ORSNETFEATS,
        num_cab=opt.MODEL.NUM_CAB,
        kernel_size=opt.MODEL.KERNEL_SIZE,
        reduction=opt.MODEL.REDUCTION,
        bias=opt.MODEL.BIAS,
    )
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        log_training(f"Using {torch.cuda.device_count()} GPUs!")

    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = opt.OPTIM.WARMUP_EPOCHS
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model_restoration.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        log_training(f"Resumed from {model_path}, epoch {start_epoch}, lr {new_lr:.8f}")

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    edge_loss_weight = opt.TRAINING.EDGE_LOSS_WEIGHT

    ######### DataLoaders ###########
    data_root = opt.TRAINING.DATA_ROOT
    train_dataset = get_training_data(data_root, {'patch_size': opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE,
                              shuffle=True, num_workers=opt.TRAINING.NUM_WORKERS,
                              drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(data_root, {'patch_size': opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.VAL_BATCH_SIZE,
                            shuffle=False, num_workers=opt.TRAINING.NUM_WORKERS_VAL,
                            drop_last=False, pin_memory=True)

    log_training(f'Start Epoch {start_epoch} End Epoch {opt.OPTIM.NUM_EPOCHS}')
    log_training(f'Train images: {len(train_dataset)}, Val images: {len(val_dataset)}')
    log_training(f'Batch size: {opt.OPTIM.BATCH_SIZE}, Patch size: {opt.TRAINING.TRAIN_PS}')
    log_training(f'LR: {new_lr:.8f}, EdgeLoss weight: {edge_loss_weight}')

    # -------- Init validation log --------
    with open(validation_log_path, 'a') as f:
        f.write(f"{'='*60}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training started\n")
        f.write(f"{'='*60}\n")

    best_psnr = 0
    best_epoch = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}'), 0):
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()

            restored = model_restoration(input_)

            loss_char = torch.sum(torch.stack(
                [criterion_char(restored[j], target) for j in range(len(restored))]))
            loss_edge = torch.sum(torch.stack(
                [criterion_edge(restored[j], target) for j in range(len(restored))]))
            loss = loss_char + edge_loss_weight * loss_edge

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start_time
        lr = scheduler.get_lr()[0]

        # -------- Training log --------
        log_training(f'Epoch [{epoch}/{opt.OPTIM.NUM_EPOCHS}]  '
                     f'Loss: {epoch_loss:.4f}  Time: {epoch_time:.2f}s  LR: {lr:.8f}')

        ######### Validation ###########
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_vals = []
            ssim_vals = []

            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                with torch.no_grad():
                    restored = model_restoration(input_)
                restored = restored[0]  # final stage only

                for res, tar in zip(restored, target):
                    # PSNR via torch (same as test: values in [0,1])
                    psnr_vals.append(torchPSNR(res, tar).item())
                    # SSIM via numpy (same as test: uint8, data_range=255)
                    res_np = (res.permute(1,2,0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    tar_np = (tar.permute(1,2,0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    ssim_val = calculate_ssim(res_np, tar_np)
                    ssim_vals.append(ssim_val)

            avg_psnr = np.mean(psnr_vals)
            avg_ssim = np.mean(ssim_vals)

            improved = avg_psnr > best_psnr
            if improved:
                best_psnr = avg_psnr
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_path)
                log_validation(
                    f'>>> 模型已更新! Epoch [{epoch}]  '
                    f'PSNR: {avg_psnr:.4f} dB  SSIM: {avg_ssim:.4f}  (Best at epoch {best_epoch})')
            else:
                log_validation(
                    f'Epoch [{epoch}]  PSNR: {avg_psnr:.4f} dB  SSIM: {avg_ssim:.4f}  '
                    f'(Best: {best_psnr:.4f} dB at epoch {best_epoch})')

            model_restoration.train()

        scheduler.step()

    # -------- Training finished --------
    log_training(f'Training finished. Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}')
    log_training(f'Model saved at: {model_path}')
    with open(validation_log_path, 'a') as f:
        f.write(f"{'='*60}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training finished. Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}\n")
        f.write(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(config_path):
    opt = Config(config_path)

    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    data_root = opt.TRAINING.DATA_ROOT
    save_dir = opt.TEST.SAVE_DIR
    checkpoint_path = opt.TEST.CHECKPOINT
    image_border = opt.TEST.IMAGE_BORDER

    save_triple = os.path.join(save_dir, 'triple_comparison')
    save_pure = os.path.join(save_dir, 'test')
    os.makedirs(save_triple, exist_ok=True)
    os.makedirs(save_pure, exist_ok=True)

    ######### Model ###########
    model = MPRNet(
        in_c=opt.MODEL.IN_C,
        out_c=opt.MODEL.OUT_C,
        n_feat=opt.MODEL.N_FEAT,
        scale_unetfeats=opt.MODEL.SCALE_UNETFEATS,
        scale_orsnetfeats=opt.MODEL.SCALE_ORSNETFEATS,
        num_cab=opt.MODEL.NUM_CAB,
        kernel_size=opt.MODEL.KERNEL_SIZE,
        reduction=opt.MODEL.REDUCTION,
        bias=opt.MODEL.BIAS,
    )
    load_checkpoint(model, checkpoint_path)
    print("===> Testing using weights:", checkpoint_path)
    model.cuda()
    model.eval()

    # Collect GT mapping
    gt_root = os.path.join(data_root, 'test_sharp')
    gt_rel_map = {}
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith('.png'):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, gt_root).replace('\\', '/')
                gt_rel_map[rel] = p

    # Collect input images grouped by group dir
    input_root = os.path.join(data_root, 'test_blur')
    input_rel_map = {}
    grouped_files = defaultdict(list)
    all_files = []

    for root, _, files in os.walk(input_root):
        for f in sorted(files, key=natural_sort_key):
            if f.lower().endswith('.png'):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, input_root).replace('\\', '/')
                input_rel_map[rel] = p
                group_name = rel.split('/')[0] if '/' in rel else ''
                grouped_files[group_name].append((p, rel, f))
                all_files.append((p, rel, f, group_name))

    mask_root = os.path.join(data_root, 'test_mask')
    report = TestReport(crop_border=image_border)

    img_multiple_of = 8
    total_per_image_logs = []
    total_seq_logs = {}
    total_seq_stats = {}

    current_group = None

    print(f"===> Starting inference on {len(all_files)} test images...")

    with torch.no_grad():
        for idx, (inp_path, rel_path, fname, group_name) in enumerate(tqdm(all_files)):
            # Determine group for mask loading
            if current_group is None:
                current_group = group_name
            elif group_name != current_group:
                current_group = group_name

            # Load and preprocess input (RGB, matching model's 3-channel input)
            inp_img = Image.open(inp_path).convert('RGB')
            inp_tensor = TF.to_tensor(inp_img).unsqueeze(0).cuda()

            # Pad to multiple of 8
            h, w = inp_tensor.shape[2], inp_tensor.shape[3]
            H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
            W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - h if h % img_multiple_of != 0 else 0
            padw = W - w if w % img_multiple_of != 0 else 0
            if padh > 0 or padw > 0:
                inp_tensor = F.pad(inp_tensor, (0, padw, 0, padh), 'reflect')

            output = model(inp_tensor)
            output = output[0]  # final stage
            output = torch.clamp(output, 0, 1)
            output = output[:, :, :h, :w]  # unpad
            if inp_tensor.shape[2] != h or inp_tensor.shape[3] != w:
                inp_tensor_display = inp_tensor[:, :, :h, :w]
            else:
                inp_tensor_display = inp_tensor

            # Model outputs 3-channel (R=G=B since grayscale input was replicated).
            # Take first channel for saving as grayscale.
            out_np = (output[0, 0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)

            # Save pure output
            pure_dir = os.path.join(save_pure, os.path.dirname(rel_path)) if os.path.dirname(rel_path) else save_pure
            os.makedirs(pure_dir, exist_ok=True)
            cv2.imwrite(os.path.join(pure_dir, fname), out_np)

            # Load GT and compute metrics
            gt_path = gt_rel_map.get(rel_path)
            if gt_path is None:
                # Fallback: search by filename
                for k, v in gt_rel_map.items():
                    if os.path.basename(k) == fname:
                        gt_path = v
                        break

            if gt_path and os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is None:
                    continue

                gt_tensor = torch.from_numpy(gt_img).float() / 255.0
                gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0).cuda()  # (1,1,H,W)
                # Replicate GT to 3 channels for comparison with model output
                gt_tensor_3ch = gt_tensor.repeat(1, 3, 1, 1)  # (1,3,H,W)

                output_resized = output  # (1,3,H,W)
                inp_display = inp_tensor_display  # (1,3,H,W)

                if output_resized.shape[2:] != gt_tensor.shape[2:]:
                    output_resized = F.interpolate(output_resized, size=(gt_tensor.shape[2], gt_tensor.shape[3]),
                                                   mode='bilinear')
                if inp_display.shape[2:] != gt_tensor.shape[2:]:
                    inp_display = F.interpolate(inp_display, size=(gt_tensor.shape[2], gt_tensor.shape[3]),
                                                mode='bilinear')

                # Save triple comparison: [input | output | gt]
                comparison = torch.cat([inp_display, output_resized, gt_tensor_3ch], dim=3)
                import torchvision
                triple_dir = os.path.join(save_triple, os.path.dirname(rel_path)) if os.path.dirname(rel_path) else save_triple
                os.makedirs(triple_dir, exist_ok=True)
                torchvision.utils.save_image(comparison, os.path.join(triple_dir, f"triple_{fname}"))

                # Full-image PSNR/SSIM
                report.update_metric(gt_img, out_np, rel_path)
                full_psnr = float(report.total_rgb_psnr[-1])
                full_ssim = float(report.total_ssim[-1])

                # Blind pixel metrics
                row = {
                    'image': rel_path,
                    'seq': group_name,
                    'psnr': full_psnr,
                    'ssim': full_ssim,
                    'blind_mae': None,
                    'blind_rmse': None,
                    'blind_psnr': None,
                    'blind_mae_input': None,
                    'blind_mae_gain_abs': None,
                    'blind_mae_gain_pct': None,
                    'blind_count': 0,
                }

                merged_coords = []
                h_img, w_img = gt_img.shape[:2]
                blind_csv = os.path.join(mask_root, group_name, 'blind_pixel_coords.csv')
                flash_csv = os.path.join(mask_root, group_name, 'flash_pixel_coords.csv')

                blind_coords = load_blind_coords(blind_csv)
                flash_map = load_flash_map(flash_csv)

                if blind_coords is not None:
                    x = blind_coords[:, 0]
                    y = blind_coords[:, 1]
                    valid = (x >= 0) & (x < w_img) & (y >= 0) & (y < h_img)
                    if np.any(valid):
                        merged_coords.extend(list(zip(x[valid].tolist(), y[valid].tolist())))

                frame_flash = flash_map.get(fname, []) if flash_map else []
                for (fx, fy) in frame_flash:
                    if 0 <= fx < w_img and 0 <= fy < h_img:
                        merged_coords.append((fx, fy))

                if len(merged_coords) > 0:
                    coords_arr = np.unique(np.array(merged_coords, dtype=np.int32), axis=0)
                    if coords_arr.size > 0:
                        x = coords_arr[:, 0]
                        y = coords_arr[:, 1]
                        gt_vals = gt_img[y, x].astype(np.float64)
                        out_vals = out_np[y, x].astype(np.float64)
                        err = out_vals - gt_vals

                        blind_abs_sum = float(np.abs(err).sum())
                        blind_sq_sum = float((err ** 2).sum())
                        blind_count = int(len(err))

                        total_seq_stats.setdefault(group_name, {
                            'blind_abs_sum': 0.0, 'blind_sq_sum': 0.0,
                            'blind_abs_in_sum': 0.0, 'blind_sq_in_sum': 0.0,
                            'blind_pix_sum': 0,
                        })
                        st = total_seq_stats[group_name]
                        st['blind_abs_sum'] += blind_abs_sum
                        st['blind_sq_sum'] += blind_sq_sum
                        st['blind_pix_sum'] += blind_count

                        # Input image blind metrics
                        in_path = input_rel_map.get(rel_path)
                        in_mae = None
                        if in_path and os.path.exists(in_path):
                            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                            if in_img is not None:
                                if in_img.shape != gt_img.shape:
                                    in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                                in_vals = in_img[y, x].astype(np.float64)
                                in_err = in_vals - gt_vals
                                in_abs_sum = float(np.abs(in_err).sum())
                                in_sq_sum = float((in_err ** 2).sum())
                                st['blind_abs_in_sum'] += in_abs_sum
                                st['blind_sq_in_sum'] += in_sq_sum
                                in_mae = float(np.abs(in_err).mean())

                        row.update({
                            'blind_mae': float(np.abs(err).mean()),
                            'blind_rmse': float(np.sqrt((err ** 2).mean())),
                            'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float((err ** 2).mean()), 1e-12))),
                            'blind_mae_input': in_mae,
                            'blind_count': blind_count,
                        })
                        if in_mae is not None:
                            row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                            row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)

                total_per_image_logs.append(row)
                total_seq_logs.setdefault(group_name, []).append(row)

    report.print_final_result()

    # Save per-image CSV
    save_blind_dir = os.path.join(save_dir, 'blind_eval')
    os.makedirs(save_blind_dir, exist_ok=True)
    keys = [
        'image', 'seq', 'psnr', 'ssim',
        'blind_mae', 'blind_rmse', 'blind_psnr',
        'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count'
    ]

    if len(total_per_image_logs) > 0:
        per_img_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
        with open(per_img_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in total_per_image_logs:
                writer.writerow(row)
        print(f"Per-image test metrics saved to: {per_img_csv}")

        # Save per-group summary CSV
        summary_keys = [
            'seq', 'images', 'blind_count',
            'blind_mae', 'blind_rmse', 'blind_psnr',
            'input_blind_mae', 'input_blind_psnr',
            'blind_mae_gain_abs', 'blind_mae_gain_pct'
        ]
        summary_csv = os.path.join(save_blind_dir, 'test_blind_summary_by_seq.csv')
        with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_keys)
            writer.writeheader()
            for seq_name in sorted(total_seq_logs.keys(), key=natural_sort_key):
                st = total_seq_stats.get(seq_name, {})
                pix = int(st.get('blind_pix_sum', 0))
                row = {
                    'seq': seq_name,
                    'images': len(total_seq_logs.get(seq_name, [])),
                    'blind_count': pix,
                    'blind_mae': None,
                    'blind_rmse': None,
                    'blind_psnr': None,
                    'input_blind_mae': None,
                    'input_blind_psnr': None,
                    'blind_mae_gain_abs': None,
                    'blind_mae_gain_pct': None,
                }
                if pix > 0:
                    mae = st['blind_abs_sum'] / pix
                    mse = st['blind_sq_sum'] / pix
                    row['blind_mae'] = float(mae)
                    row['blind_rmse'] = float(np.sqrt(mse))
                    row['blind_psnr'] = float(10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-12)))
                    if st['blind_abs_in_sum'] > 0:
                        in_mae = st['blind_abs_in_sum'] / pix
                        in_mse = st['blind_sq_in_sum'] / pix
                        row['input_blind_mae'] = float(in_mae)
                        row['input_blind_psnr'] = float(10.0 * np.log10((255.0 * 255.0) / max(in_mse, 1e-12)))
                        row['blind_mae_gain_abs'] = float(in_mae - mae)
                        row['blind_mae_gain_pct'] = float(100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12))
                writer.writerow(row)
        print(f"Per-seq summary saved to: {summary_csv}")

    # Print global blind stats
    total_blind_abs_sum = sum(s['blind_abs_sum'] for s in total_seq_stats.values())
    total_blind_sq_sum = sum(s['blind_sq_sum'] for s in total_seq_stats.values())
    total_blind_pix_sum = sum(s['blind_pix_sum'] for s in total_seq_stats.values())
    total_blind_abs_in_sum = sum(s['blind_abs_in_sum'] for s in total_seq_stats.values())
    total_blind_sq_in_sum = sum(s['blind_sq_in_sum'] for s in total_seq_stats.values())

    if total_blind_pix_sum > 0:
        blind_mae = total_blind_abs_sum / total_blind_pix_sum
        blind_mse = total_blind_sq_sum / total_blind_pix_sum
        blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))
        print(f"\n===> Blind-Pixel Focused Metrics")
        print(f"Blind MAE: {blind_mae:.6f} | Blind RMSE: {np.sqrt(blind_mse):.6f} | Blind PSNR: {blind_psnr:.3f}")

        if total_blind_abs_in_sum > 0:
            blind_mae_in = total_blind_abs_in_sum / total_blind_pix_sum
            blind_mse_in = total_blind_sq_in_sum / total_blind_pix_sum
            blind_psnr_in = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse_in, 1e-12)))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(f"Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | "
                  f"MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPRNet Blind Pixel Restoration')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run testing')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    if args.train:
        train(args.config_path)
    elif args.test:
        test(args.config_path)
    else:
        print("Please specify --train or --test")
        sys.exit(1)
