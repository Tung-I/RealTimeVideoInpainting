import argparse
import logging
import os
import sys
import torch
import random
import yaml
from box import Box
from pathlib import Path

import dataloader
import model
import runner
import callback


def main(args):
    # Load the config
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)
    logging.info(f'Save the config to "{config.main.saved_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Make experiment results deterministic.
    random.seed(config.main.random_seed)
    torch.manual_seed(random.getstate()[1][1])
    torch.cuda.manual_seed_all(random.getstate()[1][1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Cuda
    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable.")
    device = torch.device(config.trainer.kwargs.device)

    # Create the dataloaders
    logging.info('Create the training and validation datasets.')
    base_dir = Path(config.dataset.kwargs.base_dir)
    config.dataset.kwargs.update(base_dir=base_dir, type='train', device=device)
    train_dataset = _get_instance(dataloader.dataset, config.dataset)
    config.dataset.kwargs.update(base_dir=base_dir, type='valid', device=device)
    valid_dataset = _get_instance(dataloader.dataset, config.dataset)

    logging.info('Create the training and validation dataloaders.')
    train_batch_size, valid_batch_size = config.dataloader.kwargs.pop('train_batch_size'), config.dataloader.kwargs.pop('valid_batch_size')
    config.dataloader.kwargs.update(collate_fn=None, batch_size=train_batch_size)
    train_dataloader = _get_instance(dataloader, config.dataloader, train_dataset)
    config.dataloader.kwargs.update(batch_size=valid_batch_size)
    valid_dataloader = _get_instance(dataloader, config.dataloader, valid_dataset)

    # Create the model
    logging.info('Create the network architecture.')
    net = _get_instance(model.net, config.net)

    # Create the loss and metric functions
    logging.info('Create the loss functions and the corresponding weights.')
    loss_fns, loss_weights = [], []
    defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
    for config_loss in config.losses:
        if config_loss.name in defaulted_loss_fns:
            loss_fn = _get_instance(torch.nn, config_loss)
        else:
            loss_fn = _get_instance(model.losses, config_loss)
        loss_fns.append(loss_fn)
        loss_weights.append(config_loss.weight)

    logging.info('Create the metric functions.')
    metric_fns = []
    for config_metric in config.metrics:
        if config_metric.name in defaulted_loss_fns:
            metric_fn = _get_instance(torch.nn, config_metric)
        else:
            metric_fn = _get_instance(model.metrics, config_metric)
        metric_fns.append(metric_fn)

    # Create the optimizer and lr scheduler
    logging.info('Create the optimizer.')
    optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())

    logging.info('Create the learning rate scheduler.')
    lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer) if config.get('lr_scheduler') else None

    # Create the tensorboard logger and monitor
    logging.info('Create the logger.')
    config.logger.kwargs.update(log_dir=saved_dir / 'log', dummy_input=torch.randn(tuple(config.logger.kwargs.dummy_input)))
    logger = _get_instance(callback.loggers, config.logger)

    logging.info('Create the monitor.')
    config.monitor.kwargs.update(checkpoints_dir=saved_dir / 'checkpoints')
    monitor = _get_instance(callback.monitor, config.monitor)

    # Create the trainer
    logging.info('Create the trainer.')
    kwargs = {'device': device,
                'train_dataloader': train_dataloader,
                'valid_dataloader': valid_dataloader,
                'net': net,
                'loss_fns': loss_fns,
                'loss_weights': loss_weights,
                'metric_fns': metric_fns,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'logger': logger,
                'monitor': monitor}
    config.trainer.kwargs.update(kwargs)
    trainer = _get_instance(runner.trainer, config.trainer)

    loaded_path = config.main.get('loaded_path')
    if loaded_path:
        logging.info(f'Load the previous checkpoint from "{loaded_path}".')
        trainer.load(Path(loaded_path))
        logging.info('Resume training.')
    else:
        logging.info('Start training.')
    trainer.train()
    logging.info('End training.')


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.
    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The script for training and testing.")
    parser.add_argument('-c', '--config_path', type=Path, help='The path of the config file.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    main(args)