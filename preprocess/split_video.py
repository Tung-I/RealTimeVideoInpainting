import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *


viewport = 'l15'

input_frames, n_frame = read_video(f'./videos/{viewport}/input.mp4')
gt_frames, n_frame = read_video(f'./videos/{viewport}/gt.mp4')

input_tiles = {}
for j in range(0, 960, 240):
    for i in range(0, 960, 240):
        input_tiles[f'{j}_{i}'] = []
for frame in input_frames:
    for j in range(0, 960, 240):
        for i in range(0, 960, 240):
            input_tiles[f'{j}_{i}'].append(frame[j:j+240, i:i+240])

gt_tiles = {}
for j in range(0, 960, 240):
    for i in range(0, 960, 240):
         gt_tiles[f'{j}_{i}'] = []
for frame in gt_frames:
    for j in range(0, 960, 240):
        for i in range(0, 960, 240):
            gt_tiles[f'{j}_{i}'].append(frame[j:j+240, i:i+240])

fps = 24
resolution = 240
mkdir_if_missing(f'./videos/{viewport}/tiles')
for k in tqdm(input_tiles.keys()):
    save_path = f'./videos/{viewport}/tiles/input_{k}.mp4'
    save_video(save_path, input_tiles[k], fps, resolution=(resolution, resolution))
    save_path = f'./videos/{viewport}/tiles/gt_{k}.mp4'
    save_video(save_path, gt_tiles[k], fps, resolution=(resolution, resolution))