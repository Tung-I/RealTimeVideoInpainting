import torch
import cv2
import numpy as np
import copy
import os 
import random
from pathlib import Path
from PIL import Image
from typing import Callable, Sequence, Union, List, Dict
from torchvision import transforms


from dataloader.dataset.base_dataset import BaseDataset


def read_frame_from_video(video_path, im_size=(240, 240), is_color=True):
    """
    Return:
        frames: list of np arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame if is_color else frame[:, :, 0]
        frame = cv2.resize(frame, im_size)
        frames.append(frame.astype(np.uint8)[..., ::-1])
    cap.release()
    return frames

class TileDataset(BaseDataset):
    """Multiface Inpainting Dataset
    Args:
        im_size: The [H, W] of the input image
    """
    def __init__(
        self,
        tile_id,
        im_size=(240, 240), 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tile_id = tile_id
        self.im_size = im_size
        self.video_paths = []
        self.input_data = []
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        input_video_path = os.path.join(self.base_dir, f'input_{self.tile_id}.mp4')
        gt_video_path = os.path.join(self.base_dir, f'gt_{self.tile_id}.mp4')
        input_frames = read_frame_from_video(input_video_path)
        gt_frames = read_frame_from_video(gt_video_path)

        frame_num = len(gt_frames)
        print(frame_num)
        for n in range(frame_num):
            self.input_data.append((input_frames[n], gt_frames[n]))

        if self.type == 'valid':
            random.seed(0)
            self.input_data = random.sample(self.input_data, 32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        im, gt = self.input_data[index]

        im = self.transform(im.copy())
        gt = self.transform(gt.copy())

        return {"image": im, "label": gt}
