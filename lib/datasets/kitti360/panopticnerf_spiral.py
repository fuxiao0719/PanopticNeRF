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

class Dataset:
    def __init__(self, cam2world_root, img_root, bbx_root, data_root, sequence, pseudo_root,  split):
        super(Dataset, self).__init__()
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.start = cfg.start
        self.spiral_frame = cfg.intersection_spiral_frame
        self.spiral_frame_num = cfg.intersection_frames
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.dir_02 = np.array(cfg.dir_02)
        self.dir_03 = np.array(cfg.dir_03)

        # load intrinsics
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)
        self.K_00[:2] = self.K_00[:2] * cfg.ratio
        self.K_01[:2] = self.K_01[:2] * cfg.ratio
        self.intrinsic_00 = self.K_00[:, :-1]
        self.intrinsic_01 = self.K_01[:, :-1]

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

        # load cam2world poses
        self.cam2world_dict_00 = {}
        self.cam2world_dict_01 = {}
        self.cam2world_dict_02 = {}
        self.cam2world_dict_03 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        Tr = loadCalibrationCameraToPose(fileCameraToPose)
        T1 = Tr['image_01']
        T2 = Tr['image_02'] # left fisheye
        T3 = Tr['image_03'] # right fisheye
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, T1), np.linalg.inv(self.R_rect))
            self.cam2world_dict_02[frame] = np.matmul(pose, T2)
            self.cam2world_dict_03[frame] = np.matmul(pose, T3)
        self.translation = np.array(cfg.center_pose)

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

        if cfg.use_stereo == True:
            self.cam2world_dict = self.cam2world_dict_01
            self.intrinsic = self.intrinsic_01
            self.stereo_num = 1
        else:
            self.cam2world_dict = self.cam2world_dict_00
            self.intrinsic = self.intrinsic_00
            self.stereo_num = 0
        
        # generate spiral poses
        self.translation = np.array(cfg.center_pose)
        up = np.array([0,0,-1])
        c2w = self.cam2world_dict[self.spiral_frame] #c2w
        N_views = self.spiral_frame_num
        render_poses = render_path_spiral_360(c2w=c2w, up=up, N=N_views)
        self.render_poses = np.array(render_poses).astype(np.float32)

        # load intersections
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.bbx_intersection_root = os.path.join(data_root, 'bbx_intersection', self.sequence, 'spiral_'+str(self.spiral_frame))
        self.intersections_dict = {}
        for idx in range(self.spiral_frame_num):
            frame_name = '%010d' % self.spiral_frame
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            if cfg.use_stereo == True:
                intersection_file = os.path.join(self.bbx_intersection_root, str(idx) + '_01.npz')
            else:
                intersection_file = os.path.join(self.bbx_intersection_root, str(idx) + '.npz')
            if not os.path.isfile(intersection_file):
                raise RuntimeError('%s does not exist!' % intersection_file)
            self.intersections_dict[idx] = intersection_file

        # load metas
        self.build_metas(self.intersections_dict)

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect

    def build_metas(self, intersection_dict):
        input_tuples = []
        for idx in range(self.spiral_frame_num):
            pose = self.render_poses[idx]
            pose[:3, 3] = pose[:3, 3] - self.translation
            intersection_path = intersection_dict[idx]
            intersection = np.load(intersection_path)
            intersection_depths = intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
            intersection_annotations = intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
            intersection = np.concatenate((intersection_depths, intersection_annotations), axis=2)
            rays = build_rays(self.intrinsic, pose, self.H, self.W)
            angle_02 = np.arccos(np.clip(np.dot(rays[:,3:6], self.dir_02), -1., 1.))
            angle_03 = np.arccos(np.clip(np.dot(rays[:,3:6], self.dir_03), -1., 1.))
            cam_idx = angle_03 / (angle_02 +  angle_03)
            input_tuples.append((rays, cam_idx, self.spiral_frame, intersection, self.stereo_num, idx))
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
