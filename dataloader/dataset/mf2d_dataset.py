import csv
import glob
import torch
import cv2
import numpy as np
import copy
import random
from pathlib import Path

from dataloader.dataset.base_dataset import BaseDataset
from dataloader.dataset.transform import Normalize, ToTensor

entities_train = ["m--20180227--0000--6795937--GHS", "m--20180406--0000--8870559--GHS", "m--20180418--0000--2183941--GHS",
                "m--20180426--0000--002643814--GHS", "m--20180510--0000--5372021--GHS", "m--20180927--0000--7889059--GHS",
                "m--20181017--0000--002914589--GHS"]
entities_valid = ["m--20180226--0000--6674443--GHS"]


class MF2DDataset(BaseDataset):
    """
    Args:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normalize = Normalize()
        self.to_tensor = ToTensor()
        self.im_paths = []
        # self.gt_paths = []
        self.im_gt_pairs = []
        if self.type == 'train':
            self.entities = entities_train
        else:
            self.entities = entities_valid

        for ent in self.entities:
            imdir = self.base_dir / Path('warped') / Path(ent) 
            self.im_paths.extend(list(imdir.glob('*.jpg')))
            imdir = self.base_dir / Path('warped_with_noise') / Path(ent) 
            self.im_paths.extend(list(imdir.glob('*.jpg')))
            
        for im_path in self.im_paths:
            ent = im_path.parts[-2]
            imdir = self.base_dir / Path('warped_gt') / ent 
            str_split = str(im_path.parts[-1]).split('_')
            file_name = '{}_{}_{}'.format(str_split[0], str_split[1], str_split[2])
            file_name = file_name + '.jpg' if '.jpg' not in file_name else file_name
            # print(imdir / Path(file_name))
            # print(im_path)
            # raise Exception(' ')
            self.im_gt_pairs.append((im_path, imdir/Path(file_name)))

        # if self.type == 'valid':
        #     random.seed(0)
        #     self.im_paths = random.sample(self.im_paths, 500)

        # self.im_gt_pairs = random.sample(self.im_gt_pairs, 48)


    def __len__(self):
        return len(self.im_gt_pairs)

    def __getitem__(self, index):
        im_path, gt_path = self.im_gt_pairs[index]
        # print(im_path)
        # print(gt_path)
        im = cv2.imread(str(im_path))[..., ::-1]
        gt = cv2.imread(str(gt_path))[..., ::-1]

        if im.shape != (self.im_size[0], self.im_size[1], 3): 
            im = cv2.resize(im, (self.im_size[1], self.im_size[0]))
            gt = cv2.resize(gt, (self.im_size[1], self.im_size[0]))

        im, gt = self.normalize(im, gt)
        im, gt = self.to_tensor(im, gt, dtypes=[torch.float, torch.float])

        return {"image": im, "label": gt}