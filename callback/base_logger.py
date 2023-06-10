import torch
import random
import copy
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """The base class for all loggers.
    Args:
        log_dir: The saved directory.
        dummy_input: The dummy input for plotting the network architecture.
    """
    def __init__(
        self,
        log_dir: str,
        dummy_input: torch.Tensor
    ):
        self.writer = SummaryWriter(log_dir)

    def write(
        self,
        epoch: int,
        train_log: dict,
        train_batch: dict,
        train_outputs: torch.Tensor,
        valid_log: dict,
        valid_batch: dict,
        valid_outputs: torch.Tensor
    ):
        self._add_scalars(epoch, train_log, valid_log)
        self._add_images(epoch, train_batch, train_outputs, valid_batch, valid_outputs)

    def close(self):
        self.writer.close()

    def _add_scalars(
        self,
        epoch: int,
        train_log: dict,
        valid_log: dict
    ):
        for key in train_log:
            self.writer.add_scalars(key, {'train': train_log[key], 'valid': valid_log[key]}, epoch)

    def _add_images(
        self,
        epoch: int,
        train_batch: dict,
        train_outputs: torch.Tensor,
        valid_batch: dict,
        valid_outputs: torch.Tensor
    ):
        raise NotImplementedError