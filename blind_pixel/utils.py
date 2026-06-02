import os
import re
import csv
import torch
import numpy as np
import cv2
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


# ---------------------------------------------------------------------------
# PSNR / SSIM (torch tensors, values in [0,1])
# ---------------------------------------------------------------------------

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    if rmse == 0:
        return torch.tensor(100.0)
    ps = 20 * torch.log10(1 / rmse)
    return ps


# ---------------------------------------------------------------------------
# PSNR / SSIM (numpy uint8, values in [0,255])
# ---------------------------------------------------------------------------

def calculate_psnr(img1, img2):
    """img1, img2: numpy uint8 arrays, same shape."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """img1, img2: numpy uint8 arrays, HWC or HW format."""
    if img1.ndim == 3 and img1.shape[2] > 1:
        return ssim(img1, img2, data_range=255, channel_axis=2)
    else:
        return ssim(img1, img2, data_range=255)


# ---------------------------------------------------------------------------
# Checkpoint helpers (handles DataParallel module. prefix)
# ---------------------------------------------------------------------------

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except Exception:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_start_epoch(weights):
    checkpoint = torch.load(weights, map_location='cpu')
    return checkpoint.get("epoch", 0)


def load_optim(optimizer, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])


# ---------------------------------------------------------------------------
# Blind pixel coordinate loaders
# ---------------------------------------------------------------------------

def load_blind_coords(csv_path):
    """Read blind pixel coordinates from CSV, return unique (N,2) array or None."""
    if not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if len(coords) == 0:
        return None
    return np.unique(np.array(coords, dtype=np.int32), axis=0)


def load_flash_map(csv_path):
    """
    Read flash pixel coordinates from CSV.
    Returns: dict {frame_name: [(x,y), ...], ...}
    """
    if not os.path.exists(csv_path):
        return {}
    flash_map = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'frame_name' not in reader.fieldnames or \
           'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                fname = os.path.basename(row['frame_name'])
                x = int(float(row['x']))
                y = int(float(row['y']))
            except Exception:
                continue
            flash_map.setdefault(fname, []).append((x, y))
    return flash_map


# ---------------------------------------------------------------------------
# Natural sort key
# ---------------------------------------------------------------------------

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# ---------------------------------------------------------------------------
# TestReport – accumulates PSNR/SSIM across test images
# ---------------------------------------------------------------------------

class TestReport:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border
        self.total_rgb_psnr = []
        self.total_ssim = []

    def update_metric(self, gt_img, out_img, name=None):
        """gt_img, out_img: numpy uint8 arrays."""
        if self.crop_border > 0:
            b = self.crop_border
            gt_img = gt_img[b:-b, b:-b]
            out_img = out_img[b:-b, b:-b]
        psnr_val = calculate_psnr(gt_img, out_img)
        ssim_val = calculate_ssim(gt_img, out_img)
        self.total_rgb_psnr.append(psnr_val)
        self.total_ssim.append(ssim_val)

    def print_final_result(self):
        if len(self.total_rgb_psnr) == 0:
            print("[TestReport] No results recorded.")
            return
        avg_psnr = np.mean(self.total_rgb_psnr)
        avg_ssim = np.mean(self.total_ssim)
        print(f"\n{'='*60}")
        print(f"Full-image PSNR: {avg_psnr:.4f} dB  |  SSIM: {avg_ssim:.4f}")
        print(f"{'='*60}")
