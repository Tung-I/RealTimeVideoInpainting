import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def save_video(save_path, frames, fps=24, resolution=(960, 960)):
    """
    Args:
        frames: list of np arrays
    """
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
    for frame in frames:
        frame = frame.astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.release()

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

def main():

    # empty_tile_list = ['0_0', '0_720', '240_0', '480_0', '480_720', '720_0']

    boarder_color = (20, 152, 20)
    boarder_size = 4
    tile_folder = '/home/tungi/RealTimeVideoInpainting/preprocess/videos/l15/predicted_tiles'

    merged_frames = []
    merged_frames_with_boarder = []

    n_frame = 64
    for n in range(n_frame):
        merged_frames.append(np.zeros((960, 960, 3)))
        merged_frames_with_boarder.append(np.zeros((960, 960, 3)))
    for j in range(0, 960, 240):
        for i in range(0, 960, 240):

            # if f'{j}_{i}' in empty_tile_list:
            #     vpath = os.path.join(tile_folder, f'prediction_240_720.mp4')
            # else:
            #     vpath = os.path.join(tile_folder, f'prediction_{j}_{i}.mp4')

            vpath = os.path.join(tile_folder, f'prediction_{j}_{i}.mp4')

            tile_frames = read_frame_from_video(vpath)
            for n, tile_frame in enumerate(tile_frames):
                merged_frames[n][j:j+240, i:i+240] = tile_frame
                tile_frame_with_boarder = np.stack((np.full((240, 240), boarder_color[0]), np.full((240, 240), boarder_color[1]), np.full((240, 240), boarder_color[2])), axis=2)
                tile_frame_with_boarder[boarder_size:240-boarder_size, boarder_size:240-boarder_size] = tile_frame[boarder_size:240-boarder_size, boarder_size:240-boarder_size]
                merged_frames_with_boarder[n][j:j+240, i:i+240] = tile_frame_with_boarder

    save_path = os.path.join(tile_folder, 'merged_result.mp4')
    save_video(save_path, merged_frames)
    save_path = os.path.join(tile_folder, 'merged_result_with_boarder.mp4')
    save_video(save_path, merged_frames_with_boarder)

if __name__ == '__main__':
    main()