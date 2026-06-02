import os
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def _collect_pairs(data_root, subset):
    """Collect (blur_path, sharp_path) pairs across all groups."""
    blur_root = os.path.join(data_root, f'{subset}_blur')
    sharp_root = os.path.join(data_root, f'{subset}_sharp')

    pairs = []
    if not os.path.exists(blur_root) or not os.path.exists(sharp_root):
        return pairs

    for group_dir in sorted(os.listdir(blur_root)):
        blur_group = os.path.join(blur_root, group_dir)
        sharp_group = os.path.join(sharp_root, group_dir)
        if not os.path.isdir(blur_group) or not os.path.isdir(sharp_group):
            continue
        for fname in sorted(os.listdir(blur_group)):
            if is_image_file(fname):
                blur_path = os.path.join(blur_group, fname)
                sharp_path = os.path.join(sharp_group, fname)
                if os.path.exists(sharp_path):
                    pairs.append((blur_path, sharp_path))
    return pairs


def _load_image(path):
    """Load image as RGB (dataset images are already RGB, .convert('RGB') as safety)."""
    return Image.open(path).convert('RGB')


class DataLoaderTrain(Dataset):
    def __init__(self, data_root, img_options=None):
        super(DataLoaderTrain, self).__init__()
        self.img_options = img_options or {}
        self.ps = self.img_options.get('patch_size', 256)
        self.pairs = _collect_pairs(data_root, 'train')
        self.sizex = len(self.pairs)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path, tar_path = self.pairs[index_]

        inp_img = _load_image(inp_path)
        tar_img = _load_image(tar_path)

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations (8-way)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, data_root, img_options=None):
        super(DataLoaderVal, self).__init__()
        self.img_options = img_options or {}
        self.ps = self.img_options.get('patch_size', 256)
        self.pairs = _collect_pairs(data_root, 'val')
        self.sizex = len(self.pairs)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path, tar_path = self.pairs[index_]

        inp_img = _load_image(inp_path)
        tar_img = _load_image(tar_path)

        if ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, data_root):
        super(DataLoaderTest, self).__init__()
        self.files = []
        blur_root = os.path.join(data_root, 'test_blur')
        if os.path.exists(blur_root):
            for group_dir in sorted(os.listdir(blur_root)):
                group_path = os.path.join(blur_root, group_dir)
                if not os.path.isdir(group_path):
                    continue
                for fname in sorted(os.listdir(group_path)):
                    if is_image_file(fname):
                        full_path = os.path.join(group_path, fname)
                        rel_path = os.path.join(group_dir, fname)
                        self.files.append((full_path, rel_path, group_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        inp_path, rel_path, group_name = self.files[index]
        inp = _load_image(inp_path)
        inp = TF.to_tensor(inp)
        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
        return inp, filename, rel_path


def get_training_data(data_root, img_options):
    return DataLoaderTrain(data_root, img_options)


def get_validation_data(data_root, img_options):
    return DataLoaderVal(data_root, img_options)


def get_test_data(data_root):
    return DataLoaderTest(data_root)
