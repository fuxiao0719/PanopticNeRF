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
    def __init__(self, cam2world_root, img_root, bbx_root, data_root, sequence, pseudo_root, split):
        super(Dataset, self).__init__()
        
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.start = cfg.start
        self.data_root = data_root
        self.pseudo_root = pseudo_root
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)

        # load image_ids
        train_ids = np.arange(self.start, self.start + cfg.train_frames)
        test_ids = np.array(cfg.val_list)

        # self.image_ids = train_ids
        if split == 'train':
            self.image_ids = train_ids
        elif split == 'val':
            if cfg.use_post_processing == True:
                self.image_ids = train_ids
            else:
                self.image_ids = test_ids
        
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

        # load images
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.images_list_00 = {}
        self.images_list_01 = {}
        self.images_list_02 = {}
        self.images_list_03 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            image_file_00 = os.path.join(img_root, 'image_00/data_rect/%s.png' % frame_name)
            image_file_01 = os.path.join(img_root, 'image_01/data_rect/%s.png' % frame_name)
            image_file_02 = os.path.join(img_root, 'image_02/data_rgb/%s.png' % frame_name)
            image_file_03 = os.path.join(img_root, 'image_03/data_rgb/%s.png' % frame_name)
            if not os.path.isfile(image_file_00):
                raise RuntimeError('%s does not exist!' % image_file_00)
            self.images_list_00[idx] = image_file_00
            self.images_list_01[idx] = image_file_01
            self.images_list_02[idx] = image_file_02
            self.images_list_03[idx] = image_file_03
        
        # load intersections
        self.bbx_intersection_root = os.path.join(data_root, 'bbx_intersection')
        self.intersections_dict_00 = {}
        self.intersections_dict_01 = {}
        self.intersections_dict_02 = {}
        self.intersections_dict_03 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            intersection_file_00 = os.path.join(self.bbx_intersection_root,self.sequence,str(idx) + '.npz')
            intersection_file_01 = os.path.join(self.bbx_intersection_root, self.sequence,str(idx) + '_01.npz')
            intersection_file_02 = os.path.join(self.bbx_intersection_root, self.sequence,str(idx) + '_02.npz')
            intersection_file_03 = os.path.join(self.bbx_intersection_root, self.sequence,str(idx) + '_03.npz')
            if not os.path.isfile(intersection_file_00):
                raise RuntimeError('%s does not exist!' % intersection_file_00)
            self.intersections_dict_00[idx] = intersection_file_00
            self.intersections_dict_01[idx] = intersection_file_01
            self.intersections_dict_02[idx] = intersection_file_02
            self.intersections_dict_03[idx] = intersection_file_03

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
    
    def build_metas(self):
        input_tuples = []
        for idx, frameId in enumerate(self.image_ids):
            idx = frameId - self.start
            pose = self.cam2world_dict_00[frameId]
            pose[:3, 3] = pose[:3, 3] - self.translation
            image_path = self.images_list_00[frameId]
            intersection_path = self.intersections_dict_00[frameId]
            intersection = np.load(intersection_path)
            intersection_depths = intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
            intersection_annotations = intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
            intersection = np.concatenate((intersection_depths, intersection_annotations), axis=2)           
            image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])
            rays_rgb = image.reshape(-1, 3)
            fisheye_image_path = self.images_list_02[frameId]
            fisheye_intersection_path = self.intersections_dict_02[frameId]
            fisheye_intersection = np.load(fisheye_intersection_path)
            fisheye_intersection_depths = fisheye_intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
            fisheye_intersection_annotations = fisheye_intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
            fisheye_intersection = np.concatenate((fisheye_intersection_depths, fisheye_intersection_annotations), axis=2)
            fisheye_intersection = fisheye_intersection[self.left_inds_valid]
            fisheye_image = (np.array(imageio.imread(fisheye_image_path)) / 255.).astype(np.float32)
            fisheye_image = cv2.resize(fisheye_image, (350, 350), interpolation=cv2.INTER_AREA)
            fisheye_rays_rgb = fisheye_image.reshape(-1, 3)[self.left_inds_valid]
            fisheye_pose = self.cam2world_dict_02[frameId]
            fisheye_pose[:3, 3] = fisheye_pose[:3, 3] - self.translation
            fisheye_rays = build_fisheye_rays(self.left_valid, fisheye_pose)
            pseudo_label = cv2.imread(os.path.join(self.pseudo_root, self.sequence[-9:-5]+'_{:010}.png'.format(frameId)), cv2.IMREAD_GRAYSCALE)
            pseudo_label = cv2.resize(pseudo_label, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            fisheye_pseudo_label = np.load(os.path.join(self.data_root, 'tao_fisheye/image_02', self.sequence[-9:-5]+'_{:010}.npy'.format(frameId)))
            fisheye_pseudo_label = fisheye_pseudo_label.reshape(-1)[self.left_inds_valid]
            depth = np.loadtxt("datasets/KITTI-360/sgm/{}/depth_{:010}_0.txt".format(self.sequence, frameId))
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            input_tuples.append((pose, rays, rays_rgb, fisheye_rays, fisheye_rays_rgb, self.left_inds_valid, frameId, intersection, fisheye_intersection, pseudo_label, fisheye_pseudo_label, self.intrinsic_00, 0, idx, depth))
        print('load meta_00 done')
        
        if cfg.use_stereo == True:
            for idx, frameId in enumerate(self.image_ids):
                idx = frameId - self.start
                pose = self.cam2world_dict_01[frameId]
                pose[:3, 3] = pose[:3, 3] - self.translation
                image_path = self.images_list_01[frameId]
                intersection_path = self.intersections_dict_01[frameId]
                intersection = np.load(intersection_path)
                intersection_depths = intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
                intersection_annotations = intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
                intersection = np.concatenate((intersection_depths, intersection_annotations), axis=2)
                image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                rays = build_rays(self.intrinsic_01, pose, image.shape[0], image.shape[1])
                rays_rgb = image.reshape(-1, 3)
                fisheye_image_path = self.images_list_03[frameId]
                fisheye_intersection_path = self.intersections_dict_03[frameId]
                fisheye_intersection = np.load(fisheye_intersection_path)
                fisheye_intersection_depths = fisheye_intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
                fisheye_intersection_annotations = fisheye_intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
                fisheye_intersection = np.concatenate((fisheye_intersection_depths, fisheye_intersection_annotations), axis=2)
                fisheye_intersection = fisheye_intersection[self.right_inds_valid]
                fisheye_image = (np.array(imageio.imread(fisheye_image_path)) / 255.).astype(np.float32)
                fisheye_image = cv2.resize(fisheye_image, (350, 350), interpolation=cv2.INTER_AREA)
                fisheye_rays_rgb = fisheye_image.reshape(-1, 3)[self.right_inds_valid]
                fisheye_pose = self.cam2world_dict_03[frameId]
                fisheye_pose[:3, 3] = fisheye_pose[:3, 3] - self.translation
                fisheye_rays = build_fisheye_rays(self.right_valid, fisheye_pose)
                pseudo_label = np.zeros((self.W, self.H))
                fisheye_pseudo_label = np.load(os.path.join(self.data_root, 'tao_fisheye/image_03', self.sequence[-9:-5]+'_{:010}.npy'.format(frameId)))
                fisheye_pseudo_label = fisheye_pseudo_label.reshape(-1)[self.right_inds_valid]
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                depth = -1 * np.ones_like(image)
                depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                input_tuples.append((pose, rays, rays_rgb, fisheye_rays, fisheye_rays_rgb, self.right_inds_valid, frameId, intersection, fisheye_intersection, pseudo_label, fisheye_pseudo_label, self.intrinsic_01, 1, idx, depth))
            print('load meta_01 done')

        self.metas = input_tuples

    def __getitem__(self, index):
        pose, rays, rays_rgb, fisheye_rays, fisheye_rays_rgb, inds_valid, frameId, intersection, fisheye_intersection, pseudo_label, fisheye_pseudo_label, intrinsics, stereo_num, idx, depth = self.metas[index]

        instance2id, id2instance, semantic2id, id2semantic = convert_id_instance(intersection)
        fisheye_instance2id, fisheye_id2instance, fisheye_semantic2id, fisheye_id2semantic = convert_id_instance(fisheye_intersection)
        
        if self.split == 'train':

            # perspective
            rand_ids = np.random.permutation(len(rays))
            rays = rays[rand_ids[:cfg.N_rays]]
            rays_rgb = rays_rgb[rand_ids[:cfg.N_rays]]
            intersection = intersection[rand_ids[:cfg.N_rays]]
            pseudo_label = pseudo_label.reshape(-1)[rand_ids[:cfg.N_rays]]
            depth = depth.reshape(-1)[rand_ids[:cfg.N_rays]]
            
            # fisheye (sky samp.:forground samp.=1:4)
            fe_sky_idxs = np.where(fisheye_pseudo_label==23)[0]
            fe_foreground_idxs = np.where(fisheye_pseudo_label!=23)[0]
            rand_ids_fe_sky = np.random.permutation(len(fe_sky_idxs))
            rand_ids_fe_foreground = np.random.permutation(len(fe_foreground_idxs))
            fe_sky_num = int(0.2*cfg.N_fe_rays)
            fe_foreground_num = cfg.N_fe_rays - fe_sky_num
            rand_ids_fe = np.concatenate((fe_sky_idxs[rand_ids_fe_sky[:fe_sky_num]], fe_foreground_idxs[rand_ids_fe_foreground[:fe_foreground_num]]))
            fisheye_rays = fisheye_rays[rand_ids_fe]
            fisheye_rays_rgb = fisheye_rays_rgb[rand_ids_fe]
            fisheye_intersection = fisheye_intersection[rand_ids_fe]
            fisheye_pseudo_label = fisheye_pseudo_label.reshape(-1)[rand_ids_fe]

        ret = {
            'rays': rays.astype(np.float32),
            'rays_rgb': rays_rgb.astype(np.float32),
            'fisheye_rays': fisheye_rays.astype(np.float32),
            'fisheye_rays_rgb': fisheye_rays_rgb.astype(np.float32),
            'fisheye_inds_valid': inds_valid,
            'intersection': intersection,
            'fisheye_intersection': fisheye_intersection,
            'intrinsics': intrinsics.astype(np.float32),
            'pseudo_label': pseudo_label,
            'fisheye_pseudo_label': fisheye_pseudo_label,
            'pose' : pose,
            'meta': {
                'sequence': '{}'.format(self.sequence)[0],
                'tar_idx': frameId,
                'h': self.H,
                'w': self.W
            },
            'stereo_num': stereo_num,
            'idx': idx,
            'depth': depth.astype(np.float32),
            'instance2id': instance2id,
            'id2instance': id2instance,
            'semantic2id': semantic2id,
            'id2semantic': id2semantic,
            'fisheye_instance2id': fisheye_instance2id,
            'fisheye_id2instance': fisheye_id2instance,
            'fisheye_semantic2id': fisheye_semantic2id,
            'fisheye_id2semantic': fisheye_id2semantic
        }

        if self.split == 'train' and cfg.use_instance_loss == True:
            linear_regression_pseudo_label = np.load(os.path.join(cfg.result_dir, self.sequence[0], 'img{:4d}_linear_regression_0{}.npy'.format(frameId, stereo_num)))
            linear_regression_pseudo_label = linear_regression_pseudo_label[rand_ids[:cfg.N_rays]]
            ret['linear_regression_pseudo_label'] = linear_regression_pseudo_label

        return ret

    def __len__(self):
        return len(self.metas)
