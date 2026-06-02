import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    """Laplacian pyramid edge loss. Uses 3-channel kernel matching original MPRNet."""

    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class BlindMaskWeightedLoss(nn.Module):
    """
    Mask-weighted Charbonnier loss that gives higher weight to blind pixel regions.
    mask: binary tensor (B,1,H,W), 1 = blind pixel, 0 = normal pixel.
    """

    def __init__(self, blind_weight=5.0, eps=1e-3):
        super(BlindMaskWeightedLoss, self).__init__()
        self.blind_weight = blind_weight
        self.eps = eps

    def forward(self, x, y, mask):
        diff = x - y
        loss_per_pixel = torch.sqrt((diff * diff) + (self.eps * self.eps))
        weight = 1.0 + mask * (self.blind_weight - 1.0)
        return (loss_per_pixel * weight).mean()
