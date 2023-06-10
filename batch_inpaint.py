# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import time

from net.DSTT import InpaintGenerator


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img

input_size = 240
output_size = 60

parser = argparse.ArgumentParser(description="DSTT")
parser.add_argument("-v", "--video", type=str, required=False)
parser.add_argument("-m", "--mask",   type=str, required=False)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model", type=str, default='DSTT')
parser.add_argument("--width", type=int, default=input_size)
parser.add_argument("--height", type=int, default=input_size)
parser.add_argument("--outw", type=int, default=input_size)
parser.add_argument("--outh", type=int, default=input_size)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=20)
parser.add_argument("--use_mp4", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index


def process_mask(mask_frames):
    masks = []
    for m in mask_frames: 
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks

#  read frames from video 
def read_frame_from_videos(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((input_size ,input_size)))
        success, image = vidcap.read()
        count += 1    
    return frames 


def main_worker():
    total_infer_time = 0.

    # set up models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintGenerator(output_size=(output_size, output_size)).to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckpt))
    model.eval()

    for row in range(0, 960, 240):
        for col in range(0, 960, 240):
            # prepare datset, encode all frames into deep space 
            frames = read_frame_from_videos(f'./preprocess/videos/tiles/input_{row}_{col}.mp4')
            video_length = len(frames)
            imgs = _to_tensors(frames).unsqueeze(0)*2-1
            frames = [np.array(f).astype(np.uint8) for f in frames]

            # masks = read_mask(args.mask)
            masks = read_frame_from_videos(f'./preprocess/videos/tiles/mask_{row}_{col}.mp4')
            masks = process_mask(masks)
            binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
            masks = _to_tensors(masks).unsqueeze(0)
            imgs, masks = imgs.to(device), masks.to(device)
            comp_frames = [None]*video_length
            print('loading videos and masks from: {}'.format(args.video))

            # completing holes by spatial-temporal transformers
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
                ref_ids = get_ref_index(f, neighbor_ids, video_length)
                print(f, len(neighbor_ids), len(ref_ids))
                len_temp = len(neighbor_ids) + len(ref_ids)
                selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
                selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
                with torch.no_grad():

                    timestamp = time.time()

                    masked_imgs = selected_imgs*(1-selected_masks)
                    pred_img = model(masked_imgs)
                    pred_img = (pred_img + 1) / 2
                    pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
                    for i in range(len(neighbor_ids)):
                        idx = neighbor_ids[i]
                        img = np.array(pred_img[i]).astype(
                            np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                        if comp_frames[idx] is None:
                            comp_frames[idx] = img
                        else:
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32)*0.5 + img.astype(np.float32)*0.5

                    infer_time = time.time() - timestamp
                    total_infer_time += infer_time

            save_path = f'./preprocess/videos/processed_tiles/result_{row}_{col}.mp4'
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
            for f in range(video_length):
                comp = np.array(comp_frames[f]).astype(
                    np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
                if w != args.outw:
                    comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
                writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
            writer.release()
            print('Finish in {}'.format(save_path))
    # Print total inference time
    print(f"Totoal inference time: {total_infer_time}")


if __name__ == '__main__':
    main_worker()