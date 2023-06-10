import torch
import logging
from tqdm import tqdm
import random
import copy
import numpy as np
from typing import Callable, Sequence, Union, List, Dict
from runner.trainer import BaseTrainer


class ImageTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def train(self):
        # Make the experiment reproducible
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k=self.num_epochs)

        # For each epoch
        while self.epoch <= self.num_epochs:
            # Set the np random seed
            np.random.seed(self.np_random_seeds[self.epoch - 1])
            # Do training and validation.
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')
            valid_log, valid_batch, valid_outputs = self._run_epoch('validation')
            logging.info(f'Valid log: {valid_log}.')

            # Adjust the learning rate.
            if self.lr_scheduler is None:
                pass
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                self.lr_scheduler.step(valid_log['Loss'])
            else:
                self.lr_scheduler.step()

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_batch, train_outputs,
                              valid_log, valid_batch, valid_outputs)

            # Save the regular checkpoint.
            saved_path = self.monitor.is_saved(self.epoch)
            if saved_path:
                logging.info(f'Save the checkpoint to {saved_path}.')
                self.save(saved_path)

            # Save the best checkpoint.
            saved_path = self.monitor.is_best(valid_log)
            if saved_path:
                logging.info(f'Save the best checkpoint to {saved_path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                self.save(saved_path)
            else:
                logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_improved_count}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

            # Early stop.
            if self.monitor.is_early_stopped():
                logging.info('Early stopped.')
                break

            self.epoch +=1

        self.logger.close()


    def _run_epoch(self, mode: str):
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        
        # For each batch
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets = self._get_inputs_targets(batch)

            if mode == 'training':
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    losses = self._compute_losses(outputs, targets)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
            metrics =  self._compute_metrics(outputs, targets)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, batch, outputs

    
    def _compute_losses(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        losses = [loss(output, target) for loss in self.loss_fns]
        return losses


    def _compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics


    def _get_inputs_targets(
        self,
        batch: dict
    ):
        return batch['image'], batch['label']

    
    def _allocate_data(
        self,
        batch: Union[dict, Sequence, torch.Tensor]
    ):
        if isinstance(batch, dict):
            return dict((key, self._allocate_data(data)) for key, data in batch.items())
        elif isinstance(batch, list):
            return list(self._allocate_data(data) for data in batch)
        elif isinstance(batch, tuple):
            return tuple(self._allocate_data(data) for data in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)