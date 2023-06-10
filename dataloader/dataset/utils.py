import os
import cv2
import math
import numpy as np
import imageio
import torch
import random
import pytorch3d as p3d
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    NormWeightedCompositor,
    PointLights,
    MeshRenderer
)
import time
from pathlib import Path
from collections import OrderedDict


def mkdir_if_missing(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)

def check_dir(path):
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                print("directory does not exist: {}".format(p))
    else:         
        if not os.path.exists(path):
            print("directory does not exist: {}".format(path))
            
def check_file(path):
    if isinstance(path, list):
        for p in path:
            if not os.path.isfile(p):
                print("file does not exist: {}".format(p))
    else:         
        if not os.path.isfile(path):
            print("file does not exist: {}".format(path))

def load_obj(obj_path):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(obj_path, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    
    return obj       


def create_mesh(obj, verts, tex, device=None):
    faces_uvs = obj['uv_ids']
    verts_uvs = obj['uvs']
    faces = obj['vert_ids']

    faces_uvs = torch.LongTensor(faces_uvs).unsqueeze(0)
    verts_uvs = torch.FloatTensor(verts_uvs).unsqueeze(0)
    tex = torch.FloatTensor(tex).unsqueeze(0)
    faces = torch.LongTensor(faces).unsqueeze(0)
    
    verts_mean = np.mean(verts, axis=0)
    verts -= verts_mean
    verts = torch.FloatTensor(verts).unsqueeze(0)
    
    if device is not None:
        tex = tex.to(device)
        verts = verts.to(device)
        faces_uvs = faces_uvs.to(device)
        faces = faces.to(device)
        verts_uvs = verts_uvs.to(device)

    texture = TexturesUV(tex, faces_uvs, verts_uvs)
    mesh = Meshes(verts, faces, texture)


    return mesh

        
def parse_view_circle(phi=15, dist=300, n_split=180):
    """
    Args:
        phi: the range of elevation and azimuth in degree
        dist: the distance between the object and the centered viewport
        
    Returns:
        R_traj: Tensor of shape [n_split, 3, 3]
        T_traj: Tensor of shape [n_split, 3]
    """
    def norm_vec(x):
        return x / np.linalg.norm(x)
    # the change of degree in the viewing circle
    theta = np.linspace(0, 2*np.pi, n_split)
    # diameter of the viewing circle
    diameter = dist * np.tan(phi * np.pi / 180)
    # (x, y, z) of the view positon
    x = diameter * np.cos(theta)
    y = diameter * np.sin(theta)
    z = np.ones(n_split) * dist

    R_traj = torch.zeros(n_split, 3, 3)
    for i in range(n_split):
        z_axis = norm_vec(np.array((-x[i], y[i], -z[i])))
        up = np.array([0, 1., 0])
        x_axis = norm_vec(np.cross(up, z_axis))
        y_axis = norm_vec(np.cross(z_axis, x_axis))
        _R = np.hstack((x_axis[:, None], y_axis[:, None], z_axis[:, None]))
        R_traj[i] = torch.FloatTensor(_R)
        
    _, T = look_at_view_transform(dist, 0, 0)
    T_traj = T.expand(n_split, 3)
    return R_traj, T_traj


def parse_mesh_files(base_dir, n_sample=None):
    meshdir = os.path.join(base_dir, 'tracked_mesh/EXP_ROM07_Facial_Expressions')
    texdir = os.path.join(base_dir, 'unwrapped_uv_1024/EXP_ROM07_Facial_Expressions/average')
    check_dir([meshdir, texdir])
    
    frame_nums = []
    for objpath in Path(meshdir).glob('*.obj'):
        n = os.path.split(str(objpath))[-1].split('.')[0]
        frame_nums.append(n)
        
    if n_sample is not None:
        random.seed(0)
        frame_nums = random.sample(frame_nums, n_sample)
    frame_nums.sort()
    
    file_paths = []
    for n in frame_nums:
        binpath = os.path.join(meshdir, '{}.bin'.format(n))
        objpath = os.path.join(meshdir, '{}.obj'.format(n))
        texpath = os.path.join(texdir, '{}.png'.format(n))
        check_file([binpath, objpath, texpath])
        file_paths.append({
            'obj': objpath,
            'bin': binpath,
            'tex': texpath
        })
    return file_paths
