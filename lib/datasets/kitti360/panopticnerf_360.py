import imp
from tkinter import N
import numpy as np
import os
from glob import glob
from lib.utils.data_utils import *
from lib.config import cfg, args
import imageio
from multiprocessing import Pool
from tools.kitti360scripts.helpers.annotation import Annotation3D
from tools.kitti360scripts.helpers.labels import labels, name2label
import cv2
import copy
import torch
import math

class Dataset:
    def __init__(self, cam2world_root, img_root, bbx_root, data_root, sequence, pseudo_root, split):
        super(Dataset, self).__init__()
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.data_root = data_root
        self.spiral_frame = cfg.intersection_spiral_frame
        self.image_ids = np.array([self.spiral_frame])
        self.bbx_intersection_root = os.path.join(data_root, 'bbx_intersection', self.sequence, 'spiral_'+ str(self.spiral_frame))

        # load cam2world poses
        self.cam2world_dict_02 = {}
        self.cam2world_dict_03 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        calib_dir = os.path.join(data_root, 'calibration')
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        Tr = loadCalibrationCameraToPose(fileCameraToPose)
        T2 = Tr['image_02'] # left fisheye
        T3 = Tr['image_03'] # right fisheye
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_02[frame] = np.matmul(pose, T2)
            self.cam2world_dict_03[frame] = np.matmul(pose, T3)
        self.translation = np.array(cfg.center_pose)

        # load fisheye grids
        left_fisheye_grid = np.load(os.path.join(self.data_root,'fisheye/grid_fisheye_02.npy'))
        left_fisheye_grid = left_fisheye_grid.reshape(1400, 1400, 4)[::4,::4,:].reshape(-1, 4)
        self.left_fisheye_grid = left_fisheye_grid
        mask_left = np.load(os.path.join(self.data_root,'fisheye/mask_left_fisheye.npy'))[::4,::4]
        valid = (left_fisheye_grid[:, 3] < 0.5) & (mask_left.reshape(-1) < 0.5)
        self.left_valid = left_fisheye_grid[valid, :3]
        left_inds = torch.arange(left_fisheye_grid.shape[0])
        self.left_inds_valid = left_inds[valid]
        
        right_fisheye_grid = np.load(os.path.join(self.data_root,'fisheye/grid_fisheye_03.npy'))
        right_fisheye_grid = right_fisheye_grid.reshape(1400, 1400, 4)[::4,::4,:].reshape(-1, 4)
        self.right_fisheye_grid = left_fisheye_grid
        mask_right = np.load(os.path.join(self.data_root,'fisheye/mask_right_fisheye.npy'))[::4,::4]
        valid = (right_fisheye_grid[:, 3] < 0.5) & (mask_right.reshape(-1) < 0.5)
        self.right_valid = right_fisheye_grid[valid, :3]
        right_inds = torch.arange(right_fisheye_grid.shape[0])
        self.right_inds_valid = right_inds[valid]

        fisheye_pose = self.cam2world_dict_02[self.spiral_frame]
        fisheye_pose[:3, 3] = fisheye_pose[:3, 3] - self.translation
        fisheye_rays_valid = build_fisheye_rays(self.left_valid, fisheye_pose)
        fisheye_rays = np.zeros((350*350,6))
        fisheye_rays[self.left_inds_valid] = fisheye_rays_valid
        fisheye_rays = fisheye_rays.reshape((350, 350, 6))
        self.dir_02 = fisheye_rays[350//2,350//2][3:]

        fisheye_pose = self.cam2world_dict_03[self.spiral_frame]
        fisheye_pose[:3, 3] = fisheye_pose[:3, 3] - self.translation
        fisheye_rays_valid = build_fisheye_rays(self.right_valid, fisheye_pose)
        fisheye_rays = np.zeros((350*350,6))
        fisheye_rays[self.right_inds_valid] = fisheye_rays_valid
        fisheye_rays = fisheye_rays.reshape((350, 350, 6))
        self.dir_03 = fisheye_rays[350//2,350//2][3:]

        self.translation = np.array(cfg.center_pose)
        self.camera = np.load(os.path.join(self.bbx_intersection_root, 'camera_360.npz'))
        self.pose_t = self.camera['arr_0']
        self.pose_t = self.pose_t - self.translation
        self.H = int(self.camera['arr_1'])
        self.W = int(self.camera['arr_2'])

        self.intersection =  np.load(os.path.join(self.bbx_intersection_root, '360.npz'))

        # load annotation3D
        self.annotation3D = Annotation3D(bbx_root, sequence)
        self.bbx_static = {}
        self.bbx_static_annotationId = []
        self.bbx_static_center = []
        for annotationId in self.annotation3D.objects.keys():
            if len(self.annotation3D.objects[annotationId].keys()) == 1:
                if -1 in self.annotation3D.objects[annotationId].keys():
                    self.bbx_static[annotationId] = self.annotation3D.objects[annotationId][-1]
                    self.bbx_static_annotationId.append(annotationId)
        self.bbx_static_annotationId = np.array(self.bbx_static_annotationId)

        # load metas
        self.build_metas()
    
    def build_metas(self):
        input_tuples = []
        for idx, frameId in enumerate(self.image_ids):
            intersection = self.intersection
            intersection_depths = intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
            intersection_annotations = intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
            intersection = np.concatenate((intersection_depths, intersection_annotations), axis=2)

            H, W = self.H , self.W
            _d = 1.
            _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
            _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T
            _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude
            _phi = 2*math.pi*(0.5 - (_y)/W ) # longtitude
            axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
            axis1 = np.sin(_theta).reshape(H, W, 1)
            axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)
            original_coord = np.concatenate((axis0, axis1, axis2), axis=2)
            coord = original_coord * _d
            coord = coord.reshape(-1, 3)

            x_theta = cfg.x_theta
            y_theta = cfg.y_theta
            z_theta = cfg.z_theta
        
            x_theta = x_theta * np.pi / 180
            rot_x = np.array(
                [
                    [1., 0. ,0.],
                    [0., np.cos(x_theta), np.sin(x_theta)],
                    [0., -np.sin(x_theta), np.cos(x_theta)]
                ]
            )
            y_theta = y_theta * np.pi / 180
            rot_y = np.array(
                [
                    [np.cos(y_theta), 0., -np.sin(y_theta)],
                    [0., 1., 0.],
                    [np.sin(y_theta), 0., np.cos(y_theta)]
                ]
            )
            z_theta = z_theta * np.pi / 180
            rot_z = np.array(
                [
                    [np.cos(z_theta), np.sin(z_theta),0.],
                    [-np.sin(z_theta), np.cos(z_theta), 0.],
                    [0., 0., 1.]
                ]
            )
            coord = coord@rot_z@rot_y@rot_x

            rays = np.zeros((H*W,6),dtype=np.float32)
            rays[:,:3] = self.pose_t
            rays[:,3:6] = coord

            angle_02 = np.arccos(np.clip(np.dot(coord, self.dir_02), -1., 1.))
            angle_03 = np.arccos(np.clip(np.dot(coord, self.dir_03), -1., 1.))
            cam_idx = angle_03 / (angle_02 +  angle_03)
            input_tuples.append((rays, cam_idx, frameId, intersection, 0, idx))

        print('load meta_360 done')

        self.metas = input_tuples
    

    def __getitem__(self, index):
        rays, cam_idx, frameId, intersection, stereo_num, idx = self.metas[index]
        
        instance2id, id2instance, semantic2id, id2semantic = convert_id_instance(intersection)

        ret = {
            'rays': rays.astype(np.float32),
            'cam_idx': cam_idx.astype(np.float32),
            'intersection': intersection,
            'meta': {
                'sequence': '{}'.format(self.sequence)[0],
                'tar_idx': frameId,
                'h': self.H,
                'w': self.W
            },
            'stereo_num': stereo_num,
            'idx': idx,
            'instance2id': instance2id,
            'id2instance': id2instance,
            'semantic2id': semantic2id,
            'id2semantic': id2semantic,
        }

        return ret

    def __len__(self):
        return len(self.metas)
