import torch
import logging
from tqdm import tqdm
import random
import copy
import numpy as np
import sys
import time

from runner.predictor import BasePredictor


class DAVAEPredictor(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pred_frames = []
        self.gt_frames = []
        self.base_frames = []
        self.infer_speed = None
    
    def predict(self):
        dataloader = self.test_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc='test')

        log = self._init_log()
        count = 0
        total_time = 0

        ##################
        # flag = time.time()
        ##################
        for batch in trange:
            ##################
            flag = time.time()
            ##################
            batch = self._allocate_data(batch)

            verts, view, frontview, targetview = self._get_inputs_targets(batch)

            with torch.no_grad():
                pred_tex, pred_verts, kl = self.net(frontview, verts, view)
                pred_tex = (pred_tex * batch['texstd'] + batch['texmean']) / 255.0
                targetview = (targetview * batch['texstd'] + batch['texmean']) / 255.0
                frontview = (frontview * batch['texstd'] + batch['texmean']) / 255.0
            ##################
            total_time  += (time.time() - flag)
            ##################
            metrics =  self._compute_metrics(pred_tex, targetview)

            batch_size = self.test_dataloader.batch_size
            self._update_log(log, batch_size, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

            # test_batch = {'image': frontview, 'label': targetview}
            # test_outputs = pred_tex
            logging.info(f'test log: {log}.')

            pred_frame = pred_tex * 255.
            pred_frame -= pred_frame.min()
            pred_frame = pred_frame.squeeze().permute(1, 2, 0).cpu().numpy()
            self.pred_frames.append(pred_frame.astype(np.uint8))

            gt_frame = targetview * 255.
            gt_frame -= gt_frame.min()
            gt_frame = gt_frame.squeeze().permute(1, 2, 0).cpu().numpy()
            self.gt_frames.append(gt_frame.astype(np.uint8))

            base_frame = frontview * 255.
            base_frame -= base_frame.min()
            base_frame = base_frame.squeeze().permute(1, 2, 0).cpu().numpy()
            self.base_frames.append(base_frame.astype(np.uint8))
        
            if count > 720-1:
                break


        for key in log:
            log[key] /= count

        ####################################
        # total_time  = time.time() - flag
        ###################################
        # self.infer_speed = total_time / count
        self.infer_speed = total_time / (count+1)


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
        # return batch['verts'], batch['view'], batch['frontview'], batch['targetview'], batch['mask']
        return batch['verts'], batch['view'], batch['frontview'], batch['targetview']

    
    def _allocate_data(
        self,
        batch: dict
    ):
        batch['verts'] = batch['verts'].to(self.device)
        batch['view'] = batch['view'].to(self.device)
        batch['frontview'] = batch['frontview'].to(self.device)
        batch['targetview'] = batch['targetview'].to(self.device)
        batch['texmean'] = batch['texmean'].to(self.device)
        batch['texstd'] = batch['texstd'].to(self.device)
        # batch['mask'] = batch['mask'].to(self.device)
        return batch
