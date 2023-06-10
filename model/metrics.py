import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Callable, Sequence, Union, List, Dict


def check_shape_dtype(
    output: torch.Tensor,
    target: torch.Tensor
):
    if output.dtype != target.dtype:
        raise TypeError(
            f"Got output: {output.dtype} and target: {target.dtype}."
        )
    if output.shape != target.shape:
        raise ValueError(
            f"Got output: {output.shape} and target: {target.shape}."
        )


class PSNR(nn.Module):
    def __init__(
        self,
        val_range: float = 1.,
    ):
        super().__init__()
        self.val_range = val_range

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        check_shape_dtype(output, target)
        dim = tuple(range(0, output.ndim))
        mse_error = torch.pow(output - target, 2).mean(dim=dim)
        psnr = torch.sum(10.0 * torch.log10(self.val_range**2 / (mse_error + 1e-10)))

        return psnr



class SSIM(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        val_range: float = 1.,
        n_im_channel: int = 3
    ):
        super().__init__()
        self.val_range = val_range
        self.n_im_channel = n_im_channel
        self.window_size = window_size

    def create_gaussian(
        self, 
        window_size: int,
        sigma: float
    ):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(
        self, 
        window_size: int,
        channel: int = 3
    ):
        _1D_window = self.create_gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        check_shape_dtype(output, target)
        device = torch.device(output.device)
        window = self.create_window(self.window_size, channel=self.n_im_channel).to(device)

        L = self.val_range
        mu1 = F.conv2d(output, window, padding=0, groups=self.n_im_channel)
        mu2 = F.conv2d(target, window, padding=0, groups=self.n_im_channel)

        sigma1_sq = F.conv2d(output * output, window, padding=0, groups=self.n_im_channel) - mu1.pow(2)
        sigma2_sq = F.conv2d(target * target, window, padding=0, groups=self.n_im_channel) - mu2.pow(2)
        sigma12 = F.conv2d(output * target, window, padding=0, groups=self.n_im_channel) - mu1*mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.mean()


class Dice(nn.Module):
    """The Dice score.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        pred = output.argmax(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice score.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (pred * target).sum(reduced_dims)
        union = pred.sum(reduced_dims) + target.sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return score.mean(dim=0)
