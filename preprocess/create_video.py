import os
import cv2
import numpy as np
import torch
import imageio
from tqdm import tqdm

from utils import *


if __name__ == '__main__':

    mkdir_if_missing('./videos')
    mesh_folder = '/home/tungi/dataset/mface/m--20180227--0000--6795937--GHS/tracked_mesh/E061_Lips_Puffed/'
    tex_folder = '/home/tungi/dataset/mface/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E061_Lips_Puffed/average/'

    # Create R and T of the source and target views
    R_source, T_source = look_at_view_transform(400, 0, 0)
    R_target, T_target = look_at_view_transform(400, 0, 15)

    # Collect the frame numbers
    frame_numbers = []
    for p in Path(tex_folder).glob('*.png'):
        n = os.path.split(str(p))[-1].split('.')[0]
        frame_numbers.append(n)
    frame_numbers.sort()

    # Create the mesh of each frame
    resolution = 1024
    meshes = []
    for n in tqdm(frame_numbers):
        obj, verts, tex = load_mf_obj(mesh_folder, tex_folder, n, tex_resolution=resolution)
        mesh = create_p3d_mesh(obj, verts, tex)
        meshes.append(mesh)

    # Create the simulated rgbd video
    z_far = 800
    nview_frames = []
    gt_frames = []
    mask_frames = []

    for mesh in tqdm(meshes):
    
        # novel view with holes
        rgbd_source = mesh2rgbd(mesh, R_source, T_source, resolution=resolution, zfar=z_far, return_cam=False)
        cloud = rgbd2cloud(rgbd_source, R_source, T_source, zfar=z_far)
        rgb_target = cloud2rgb(cloud, R_target, T_target)
        frame_to_save = normalize_image_value(rgb_target[0], convert_to='255')
        nview_frames.append(frame_to_save)
        # gt 
        rgbd_gt = mesh2rgbd(mesh, R_target, T_target, resolution=resolution, zfar=z_far, return_cam=False)
        rgb_gt = rgbd_gt[0][..., :3]
        frame_to_save = normalize_image_value(rgb_gt, convert_to='255')
        gt_frames.append(frame_to_save)
        # mask
        mask = np.isclose(rgb_target[0], rgb_gt, atol=0.1) # each channel could have different results
        mask = np.logical_or(mask[..., 2], np.logical_or(mask[..., 0], mask[..., 1])) 
        mask = mask.astype(np.uint8)
        mask = normalize_image_value(1 - mask, convert_to='255')
        mask_frames.append(mask)

    fps = 20
    save_path = './videos/mask.mp4'
    save_video(save_path, mask_frames, fps=fps, resolution=(resolution, resolution))
    save_path = './videos/input.mp4'
    save_video(save_path, nview_frames, fps=fps, resolution=(resolution, resolution))
    save_path = './videos/gt.mp4'
    save_video(save_path, gt_frames, fps=fps, resolution=(resolution, resolution))
