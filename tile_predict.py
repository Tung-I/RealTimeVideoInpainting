import torch
import cv2
import numpy as np
import copy
import os 
import random
import time
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Callable, Sequence, Union, List, Dict
from torchvision import transforms

import model

checkpoint_root_folder = '/home/tungi/RealTimeVideoInpainting/checkpoints/l15'


def mkdir_if_missing(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)

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

def save_video(save_path, frames, fps=24, resolution=(240, 240)):
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

def frame2tensor(frame, device=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tensor = transform(frame.copy())
    tensor = tensor.unsqueeze(0) 
    return tensor.to(device) if device is not None else tensor

def tensor2frame(tensor):
    invTrans = transforms.Compose([
        transforms.Normalize([0., 0., 0.], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [ 1., 1., 1. ]),
    ])
    tensor = invTrans(tensor)

    tensor = tensor.squeeze().permute(1, 2, 0).contiguous()
    frame = tensor.cpu().detach().numpy()
    frame = frame - frame.min()
    frame = frame / frame.max()
    frame = frame * 255
    return frame.astype(np.uint8)

def main():

    empty_tile_list = ['0_0', '0_720', '240_0', '480_0', '480_720', '720_0']


    #### Collect checkpoints
    checkpoint_paths = {}
    for j in range(0, 960, 240):
        for i in range(0, 960, 240):
            folder = os.path.join(checkpoint_root_folder, f'{j}_{i}')
            if os.path.exists(folder):
                checkpoint_paths[f'{j}_{i}'] = os.path.join(folder, 'checkpoints/model_best.pth')
    #### Collect pre-trained models
    device = torch.device('cuda:0')
    models = {}
    print('Loading pre-trained models...')
    print(checkpoint_paths)
    for k in tqdm(checkpoint_paths.keys()):
        # Initialize networks
        cls = getattr(model.net, 'miniUNet')
        kwargs = {
            'in_channels': 3,
            'out_channels': 3,
            'num_features': [16, 32, 64]
        }
        net = cls(**kwargs)
        # Load checkpoints
        # empty_tile_list = ['0_0', '0_720', '240_0', '480_0', '480_720', '720_0']
        # if k in empty_tile_list:
        #     checkpoint = torch.load(checkpoint_paths['240_720'], map_location=device)
        # else:
        #     checkpoint = torch.load(checkpoint_paths[k], map_location=device)
        checkpoint = torch.load(checkpoint_paths[k], map_location=device)
        net.load_state_dict(checkpoint['net'])
        net = net.to(device)
        models[k] = net.eval()

    #### Collect input tiles
    input_dir = '/home/tungi/RealTimeVideoInpainting/preprocess/videos/l15/tiles'
    inputs = {}
    print('Collect inputs...')
    for k in checkpoint_paths.keys():
        path = os.path.join(input_dir, f'input_{k}.mp4')
        tile_frames = read_frame_from_video(path)
        tile_tensors = []
        for tf in tile_frames:
            tile_tensors.append(frame2tensor(tf, device=device))
        inputs[k] = tile_tensors

    #### Create cuda streams for asynchronization
    cuda_streams = {}
    for k in checkpoint_paths.keys():
        cuda_streams[k] = torch.cuda.Stream()

    # Run prediction and save videos
    fps = 24
    resolution = (240, 240)
    output_dir = '/home/tungi/RealTimeVideoInpainting/preprocess/videos/l15/predicted_tiles'
    mkdir_if_missing(output_dir)

    # Calculate latency
    timestamp = time.time()

    # Within each stream
    for k in cuda_streams.keys(): 
        with torch.cuda.stream(cuda_streams[k]):

            predicted_tile_frames = [] 
            # For each frame
            for tile_tensor in inputs[k]:
                prediction = models[k](tile_tensor)
                frame = tensor2frame(prediction)

                #######################################
                bg_color = (225, 237, 245)
                if k in empty_tile_list:
                    frame =np.stack((np.full((240, 240), bg_color [0]), np.full((240, 240), bg_color [1]), np.full((240, 240), bg_color [2])), axis=2)
                #######################################

                predicted_tile_frames.append(frame)

            # Save the video
            save_video_path = os.path.join(output_dir, f'prediction_{k}.mp4')
            save_video(save_video_path, predicted_tile_frames, fps=fps, resolution=resolution)
    
    total_inf_latency = time.time() - timestamp
    inf_time_per_frame = total_inf_latency / len(inputs[k])
    print(f'Average inference time per frame: total latency {total_inf_latency} / N of frames {len(inputs[k])} = {inf_time_per_frame}')


if __name__ == '__main__':
    main()