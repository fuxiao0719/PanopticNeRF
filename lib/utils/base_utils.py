import pickle
import os
import numpy as np
import torch

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def transform_query_points_to_input_views(pts, batch):
    """transform query points into the camera spaces of the input views
    """
    K = batch['intrinsics']
    RT = batch['ext_pose'][None]
    _, n_views, _, _=RT.shape
    K = K.reshape(1,1,3,3).repeat(1,n_views,1,1)
    R = RT[:, :, :3, :3]
    T = RT[:, :, :3, 3]

    n_batch, n_pixel, n_samples = pts.shape[:3]
    # n_views = RT.shape[1]

    pts = pts.view(n_batch, -1, 3)
    pts = torch.matmul(pts, R.permute(0, 1, 3, 2).float())
    xyz_per_view = pts + T[:, :, None]

    pts = torch.matmul(xyz_per_view, K.permute(0, 1, 3, 2))
    xy_per_view = pts[..., :2] / pts[..., 2:]

    xyz_per_view = xyz_per_view.view(n_batch, n_views, n_pixel, n_samples,-1)
    xy_per_view = xy_per_view.view(n_batch, n_views, n_pixel, n_samples, -1)

    return xyz_per_view, xy_per_view

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

