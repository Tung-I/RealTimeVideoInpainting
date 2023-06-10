import torch
import logging
import random
import copy
import numpy as np
from pathlib import Path
from typing import Callable, Sequence, Union, List, Dict
from tqdm import tqdm

import callback


class BaseTrainer:
    def __init__(
        self,
        device: torch.device,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        loss_fns: Sequence[torch.nn.Module],
        loss_weights: Sequence[float],
        metric_fns: Sequence[torch.nn.Module],
        optimizer: torch.optim,
        lr_scheduler: torch.optim,
        logger: callback.BaseLogger, 
        monitor: callback.Monitor,
        num_epochs: int
    ):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net.to(device)
        self.loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]
        self.optimizer = optimizer

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
            raise NotImplementedError('Do not support torch.optim.lr_scheduler.CyclicLR scheduler yet.')
        self.lr_scheduler = lr_scheduler

        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.epoch = 1
        self.np_random_seeds = None


    def _init_log(self):
        log = {}
        log['Loss'] = 0
        for loss_fn in self.loss_fns:
            log[loss_fn.__class__.__name__] = 0
        for metric_fn in self.metric_fns:
            log[metric_fn.__class__.__name__] = 0
        return log


    def _update_log(
        self,
        log: dict,
        batch_size: int,
        loss: torch.Tensor,
        losses: Sequence[torch.Tensor],
        metrics: Sequence[torch.Tensor]
    ):
        log['Loss'] += loss.item() * batch_size
        for loss_fn, loss in zip(self.loss_fns, losses):
            log[loss_fn.__class__.__name__] += loss.item() * batch_size
        for metric_fn, metric in zip(self.metric_fns, metrics):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size


    def save(self, path: Path):
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor': self.monitor,
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'np_random_seeds': self.np_random_seeds
        }, path)


    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.monitor = checkpoint['monitor']
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        self.np_random_seeds = checkpoint['np_random_seeds']


    def train(self):
        raise NotImplementedError


    def _run_epoch(
        self,
        mode: str
    ):
        raise NotImplementedError


    def _get_inputs_targets(
        self,
        batch: dict
    ):
        raise NotImplementedError


    def _compute_losses(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        raise NotImplementedError


    def _compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        raise NotImplementedError

