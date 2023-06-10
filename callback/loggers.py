import torch
import random
import copy
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from callback.base_logger import BaseLogger


class ImageLogger(BaseLogger):
    """The 2D image logger
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def _add_images(
        self,
        epoch: int,
        train_batch: dict,
        train_output: torch.Tensor,
        valid_batch: dict,
        valid_output: torch.Tensor
    ):

        train_img = make_grid(train_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_label = make_grid(train_batch['label'].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_pred = make_grid(train_output.float(), nrow=1, normalize=True, scale_each=True, pad_value=1)

        valid_img = make_grid(valid_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_label = make_grid(valid_batch['label'].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_pred = make_grid(valid_output.float(), nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat((train_img, train_label, train_pred), dim=-1)
        valid_grid = torch.cat((valid_img, valid_label, valid_pred), dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)








