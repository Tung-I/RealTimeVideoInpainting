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


class MeshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred_verts, verts):
        return self.loss_fn(pred_verts, verts)

class ScreenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred_tex, gt_tex, texstd):
        # print(((pred_tex - gt_tex)**2).mean())
        return self.loss_fn(pred_tex, gt_tex) * (255**2) / (texstd.mean()**2)

class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward():
        return 





class MS_SSIM(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        val_range: float = 1.,
        n_im_channel: int = 3
    ):
        super().__init__()
        self.val_range = val_range
        self.n_im_channel = n_im_channel
        self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        self.window_size = window_size

    def create_gaussian(
        self, 
        window_size: int,
        sigma: float
    ):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

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
        if self.weights.device != output.device:
            self.weights = self.weights.to(device)

        cs_list = []
        ssim_list = []
        for weight in self.weights:
            window = self.create_window(self.window_size, channel=self.n_im_channel).to(device)

            L = self.val_range
            mu1 = F.conv2d(output, window, padding=0, groups=self.n_im_channel)
            mu2 = F.conv2d(target, window, padding=0, groups=self.n_im_channel)

            sigma1_sq = F.conv2d(output * output, window, padding=0, groups=self.n_im_channel) - mu1.pow(2)
            sigma2_sq = F.conv2d(target * target, window, padding=0, groups=self.n_im_channel) - mu2.pow(2)
            sigma12 = F.conv2d(output * target, window, padding=0, groups=self.n_im_channel) - mu1*mu2

            C1 = (0.01 * L) ** 2
            C2 = (0.03 * L) ** 2

            cs = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))

            cs_list.append(torch.relu(cs.mean()))
            ssim_list.append(torch.relu(ssim.mean()))

            output = F.avg_pool2d(output, kernel_size=2, padding=0)
            target = F.avg_pool2d(target, kernel_size=2, padding=0)
        
        cs = (torch.stack(cs_list) + 1.) / 2.
        ssim = (torch.stack(ssim_list) + 1.) / 2.

        ms_ssim = torch.prod((cs**self.weights)[:-1]) * (ssim**self.weights)[-1]
        
        return 1. - ms_ssim 


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, D_real, D_fake):
        true_label = torch.tensor(1.).expand_as(D_real).type_as(D_real)
        false_label = torch.tensor(0.).expand_as(D_fake).type_as(D_fake)
        loss = 0.5 * (self.loss(D_real, true_label) + self.loss(D_fake, false_label))
        return loss


class ConsisLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, D_pred, D_real):
        return self.loss(D_pred, D_real)


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        kld_weight = 0.00025 # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(output, target)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        # return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        return loss


class DiceLoss(nn.Module):
    """The Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()

