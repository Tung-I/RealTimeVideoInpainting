import os
import cv2
import math
import numpy as np
import imageio
import torch
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
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import OrderedDict

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    raise Exception('cuda unavailable')  


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_if_missing(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)

        
def save_video(save_path, frames, fps=30, resolution=(1024, 1024)):
    """
    Args:
        frames: list of np arrays
    """
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
    for frame in frames:
        frame = frame.astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.release()


def read_video(video_path, is_color=True):
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
        frames.append(frame.astype(np.float32))
    cap.release()
    return frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def save_camera(camera_base, camera_novel_list, save_path):
    """
    Save Camera objects as a zipped archive of npy files        
    """
    def _get_camera_matrices(camera, R_list, T_list):
        R_list.append(camera.R.cpu().numpy())
        T_list.append(camera.T.cpu().numpy())
        return

    R_list = []
    T_list = []
    _get_camera_matrices(camera_base, R_list, T_list)
    for camera_novel in camera_novel_list:
        _get_camera_matrices(camera_novel, R_list, T_list)
    R_list = np.concatenate(R_list, axis=0)
    T_list = np.concatenate(T_list, axis=0)
    np.savez(save_path, R=R_list, T=T_list)
    return

def read_camera(camera_path):
    """
    Load Camera objects from a zipped archive of npy files  
    """
    file = np.load(camera_path)
    R = file["R"]
    T = file["T"]
    R_base = torch.from_numpy(R[0:1, :, :])
    T_base = torch.from_numpy(T[0:1, :])
    R_novel = torch.from_numpy(R[1:, :, :])
    T_novel = torch.from_numpy(T[1:, :])
    return R_base, T_base, R_novel, T_novel



def load_mf_obj(mesh_folder, tex_folder, n, tex_resolution=1024):
    """
    Load the multiface metadata required for mesh construction of a particular frame
    
    Args:
        mesh_folder: Str giving the path to the folder that contains tracked mesh
        tex_folder: Str giving the path to the folder that contains average uwrapped texture
        n: Str giving the frame number 
        
    Returns:
        obj: Dict giving the index of each vertex associated with faces and textures
        verts: Array giving the position of each vertex
        tex: Array of shape [H, W, 3] giving the texture image
    """
    tex_path = os.path.join(tex_folder, f'{n}.png')
    obj_path = os.path.join(mesh_folder, f'{n}.obj')
    bin_path = os.path.join(mesh_folder, f'{n}.bin')
    
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
    verts = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 3))
    tex = cv2.imread(tex_path).astype(np.float32)[..., ::-1] / 255.
    tex = cv2.resize(tex, (tex_resolution, tex_resolution))
    return obj, verts, tex

    
def create_p3d_mesh(obj, verts, tex):
    """    
    Args:
        obj: Dict giving the index of each vertex associated with faces and textures
        verts: Array giving the position of each vertex
        tex: Array of shape [H, W, 3] giving the texture image
        
    Returns:
        pytorch3d.structures.mesh object
    """
    faces_uvs = obj['uv_ids']
    verts_uvs = obj['uvs']
    faces = obj['vert_ids']

    faces_uvs = torch.LongTensor(faces_uvs).unsqueeze(0).to(device)
    verts_uvs = torch.FloatTensor(verts_uvs).unsqueeze(0).to(device)
    tex = torch.FloatTensor(tex).unsqueeze(0).to(device)
    faces = torch.LongTensor(faces).unsqueeze(0).to(device)
    
    verts_mean = np.mean(verts, axis=0)
    verts -= verts_mean
    verts = torch.FloatTensor(verts).unsqueeze(0).to(device)

    texture = TexturesUV(tex, faces_uvs, verts_uvs)
    mesh = Meshes(verts, faces, texture)
    return mesh

    
def mesh2rgbd(mesh, R, T, resolution=1024, zfar=800, return_cam=False):
    """
    Args:
        mesh: pytorch3d.structures.mesh object
        R: Tensor of shape [N, 3, 3] giving the rotation matrices of N views
        T: Tensor of shape [N, 3] giving the translation matrices of N views
        
    Returns:
        rgbd: Array of shape [N, H, W, 4] giving a batch of rgbd observed from each source view
    """
    if R.size(0) != T.size(0):
        raise ValueError(f"The numbers of views in R and T are inconsistent: {R.size(0)}, {T.size(0)}")
    n_view = R.size(dim=0)
    rgbd = np.zeros((n_view, resolution, resolution, 4))
    
    raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1
        )
    lights = AmbientLights(device=device)
    cams = []
    for i in range(n_view):
        camera = FoVPerspectiveCameras(zfar=zfar, R=R[i:i+1], T=T[i:i+1], device=device)
        cams.append(camera)
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings).to(device),
            shader=SoftPhongShader(cameras=camera, lights=lights).to(device)
        )

        color, fragment = renderer(mesh)
        depth = fragment.zbuf
        color = color[0, ..., :3].cpu().numpy()
        depth = depth[0, ..., :3].cpu().numpy()[..., 0]
        rgbd[i] = np.append(color, depth[..., None], axis=-1)
    if return_cam:
        return rgbd, cams
    else:
        return rgbd

    
def rgbd2cloud(rgbd, R, T, zfar=800):
    """
    Args:
        rgbd: Array of shape [1, H, W, 4]
        R: Tensor of shape [1, 3, 3]
        T: Tensor of shape [1, 3]
        
    Returns:
        cloud: Pytorch3D Pointclouds object
    """
    color = rgbd[0, ..., :3]
    depth = rgbd[0, ..., 3]
    camera = FoVPerspectiveCameras(zfar=zfar, R=R, T=T, device=device)
    resolution = depth.shape[0]
    
    mask = (depth >= 0)
    y_mat = np.arange(resolution)[:, None].repeat(resolution, axis=1).astype(np.float32)
    y_mat = - y_mat * 2 / (resolution - 1) + 1
    x_mat = y_mat.T
    xy_depth = np.stack([x_mat, y_mat, depth], axis=2)
    xy_depth = xy_depth[mask, :]
    feats = color[mask, :]

    xy_depth = torch.FloatTensor(xy_depth).unsqueeze(0).to(device)
    feats = torch.FloatTensor(feats).unsqueeze(0).to(device)
    points = camera.unproject_points(xy_depth, scaled_depth_input=False, world_coordinates=True, in_ndc=True)
    cloud = Pointclouds(points=points, features=feats)
    return cloud
    
    
def cloud2rgb(cloud, R, T, resolution=1024, zfar=800, bg_color=(1., 1., 1.)):
    """
    Args:
        cloud: Pytorch3D Pointclouds object
        R: Tensor of shape [N, 3, 3] 
        T: Tensor of shape [N, 3]
        
    Returns:
        color: Array of shape [N, H, W, 3] giving a batch of rendered images
    """
    if R.size(0) != T.size(0):
        raise ValueError(f"The numbers of views in R and T are inconsistent: {R.size(0)}, {T.size(0)}")
    n_view = R.size(0)
    color = np.zeros((n_view, resolution, resolution, 3))
    
    raster_settings = PointsRasterizationSettings(
        image_size=resolution,
        radius=0.003,
        points_per_pixel=10
    )
    compositor=AlphaCompositor(background_color=bg_color)
    
    for i in range(n_view):
        camera = FoVPerspectiveCameras(zfar=zfar, R=R[i:i+1], T=T[i:i+1], device=device)
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=compositor
        ).to(device)
        color[i] = renderer(cloud)[0, ..., :3].cpu().numpy()
    return color


def render_path_circle(phi=15, dist=300, n_split=120):
    """
    Args:
        phi: elevation and azimuth in degree
        dist: distance between the object and the center of the circle
        
    Returns:
        R_traj: Tensor of shape [n_split, 3, 3] giving the rotation matrices along the rendering trajectory
    """
    def norm_vec(x):
        return x / np.linalg.norm(x)
    
    theta = np.linspace(0, 2*np.pi, n_split)
    diameter = dist * np.tan(phi * np.pi / 180)
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
    return R_traj

    
def parse_frame(data_root, entity, interval=1):
    """
    Args:
        interval: Integer giving the interval between each sampled frame
        
    Returns:
        List of str indicating the sampled frames
    """
    frame_list = []
    _dir = os.path.join(data_root, entity, 'tracked_mesh/EXP_ROM07_Facial_Expressions')
    for p in Path(_dir).glob('*.obj'):
        n = os.path.split(str(p))[-1].split('.')[0]
        frame_list.append(n)
    frame_list.sort()

    if interval == 1:
        sampled_frames = frame_list
    elif interval >= len(frame_list):
        sampled_frames = [frame_list[0]]
    else:
        sampled_frames = []
        for i in range(0, len(frame_list), interval):
            sampled_frames.append(frame_list[i])
    return sampled_frames

  
def normalize_depth_value(im, z_far=800, bg_val=None, mask=None, convert_to='01'):
    """
    Args:
        bg_val: Float giving the image value of background
        mask: np array giving the positions of background pixels
        
    """
    # depth = val / max_depth_val, bg = 0
    if convert_to == '01':
        mask = (im==bg_val)
        im /= z_far
        im[mask] = 0
        return im, mask
    # depth = [0, max_depth_val], bg = -1
    else:
        if im.min() < 0. or im.max() > 1.:
            raise ValueError("The depth values should be within [0, 1]")
        im *= z_far
        im[mask] = -1
        return im
    
def normalize_image_value(im, convert_to='01'):
    """
    Args:
        im: np array where values range from [0., 1.] or [0, 255]
        convert_to: Str giving the type of normalization
    """
    if convert_to=='255':
        if im.min() < 0. or im.max() > 1.:
            raise ValueError()
        return (im * 255).astype(np.int32)
    elif convert_to=='01':
        if im.min() < 0 or im.max() > 255:
            raise ValueError()
        return im.astype(np.float32) / 255.
    else:
        raise ValueError(f"Unknow argument: {convert_to}")
    
if __name__ == '__main__':
    pass