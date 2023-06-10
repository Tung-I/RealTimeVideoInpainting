import torch
import logging
from tqdm import tqdm
import random
import copy
import numpy as np
from pathlib import Path
from typing import Callable, Sequence, Union, List, Dict

import callback


class BasePredictor:
    def __init__(
        self,
        device: torch.device,
        test_dataloader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        metric_fns: Sequence[torch.nn.Module],
    ):
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]


    def _init_log(self):
        log = {}
        for metric_fn in self.metric_fns:
            log[metric_fn.__class__.__name__] = 0
        return log


    def _update_log(
        self,
        log: dict,
        batch_size: int,
        metrics: Sequence[torch.Tensor]
    ):
        for metric_fn, metric in zip(self.metric_fns, metrics):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size


    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])

    def predict(self):
        raise NotImplementedError


    def _get_inputs_targets(
        self,
        batch: dict
    ):
        raise NotImplementedError


    def _compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        raise NotImplementedError
