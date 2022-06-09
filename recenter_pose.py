import numpy as np
import os
from lib.config import cfg
from glob import glob
from numpy.lib.twodim_base import mask_indices
import imageio
from multiprocessing import Pool
import copy
import torch
from tools.kitti360scripts.helpers.annotation import Annotation3D

# AABB
def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    # bounds = bounds + np.array([-0.01, 0.01], dtype=np.float32)[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-3] = 1e-3
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-7
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    if mask_at_box.sum() > 0:
        return True
    else:
        return False

def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat

def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, 'r')
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr

class Dataset:
    def __init__(self, cam2world_root, bbx_root, data_root, sequence, split, frame_num, frame_start, visible_id_root):
        super(Dataset, self).__init__()
        self.visible_id_root = visible_id_root
        self.data_root = data_root
        self.split = split
        self.ratio = 0.5
        self.sequence = sequence
        self.cam2world_dict_00 = {}
        self.cam2world_dict_01 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        self.intrinsics = self.K_00
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, self.camToPose), np.linalg.inv(self.R_rect))
        self.intrinsics[:2] = self.intrinsics[:2] * self.ratio
        self.instance_path = os.path.join(data_root, 'data_2d_semantics', 'train', sequence, 'image_00/instance')
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.annotation3D = Annotation3D(bbx_root, sequence)
        self.start = frame_start
        train_ids = np.arange(self.start, self.start + frame_num)
        test_ids = np.arange(self.start, self.start + frame_num)
        if split == 'train':
            self.image_ids = train_ids
        elif split == 'val':
            self.image_ids = test_ids
        elif split == 'test':
            self.image_ids = train_ids
        self.metas = {}
        for idx in self.image_ids:
            self.metas[idx-self.start] = self.func((idx))
        self.bbx_static = {}
        self.bbx_static_annotationId = []
        self.bbx_static_center = []
        for annotationId in self.annotation3D.objects.keys():
            if len(self.annotation3D.objects[annotationId].keys()) == 1:
                if -1 in self.annotation3D.objects[annotationId].keys():
                    self.bbx_static[annotationId] = self.annotation3D.objects[annotationId][-1]
                    self.bbx_static_annotationId.append(annotationId)
        self.bbx_static_annotationId = np.array(self.bbx_static_annotationId)

    def func(self, input_tuple):
        idx = input_tuple
        H, W = int(self.height*self.ratio), int(self.width*self.ratio)
        filename = os.path.join(self.visible_id_root, self.sequence, '{:010d}.txt'.format(idx))
        with open(filename, "r") as f:
            data = f.read().splitlines()
            annotationId = np.array(list(map(int, data)))
        return (np.unique(annotationId), idx)

    def generate_npy(self, idx, bbx_npy_root):
        annotationId_list = []
        for i in range(idx, idx + 1):
            filename = os.path.join(self.visible_id, '{:010d}.txt'.format(idx))
            with open(filename, "r") as f:  
                data = f.read().splitlines() 
                annotationId = np.array(list(map(int, data)))  
            annotationId_list.append(np.unique(annotationId))
        annotationId_list = np.concatenate(annotationId_list)
        np.save(bbx_npy_root, np.unique(annotationId_list))

    def __getitem__(self, index):
        index = self.start+index
        pose_00 = self.cam2world_dict_00[index][:3, 3]
        pose_01 = self.cam2world_dict_01[index][:3, 3]
        annotationId_list, idx = self.metas[index - self.start]
        bbx_npy_root = os.path.join(self.data_root, 'bbx', self.sequence)
        if os.path.exists(bbx_npy_root) == False:
            os.system('mkdir -p {}'.format(bbx_npy_root))
        fileroot = os.path.join(bbx_npy_root, '{:010d}.npy'.format(idx))
        if os.path.exists(fileroot) == True:
            os.system('rm {}'.format(fileroot))
        self.generate_npy(idx, fileroot)
        annotationId_list = np.load(fileroot)
        bbx_intersection_root = os.path.join(self.data_root, 'bbx_intersection', self.sequence)
        if os.path.exists(bbx_intersection_root) == False:
            os.system('mkdir -p {}'.format(bbx_intersection_root))
        xyz_list = []
        for annotationId in annotationId_list:
            if annotationId in self.bbx_static.keys():
                xyz = self.bbx_static[annotationId].vertices
                xyz_list.append(xyz)
        return pose_00, pose_01, np.concatenate(xyz_list)

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

    def __len__(self):
        return len(self.metas)

if __name__ == "__main__":
    frame_start = cfg.recenter_start_frame
    frame_num = cfg.recenter_frames
    data_root = 'datasets/KITTI-360'
    gt_static_frames_root = os.path.join(data_root, 'static_frames.txt')
    bbx_root = os.path.join(data_root, 'data_3d_bboxes')
    visible_id_root = os.path.join(data_root, 'visible_id')
    split = 'train'
    sequence = '0000'
    sequence = os.path.join('2013_05_28_drive_' + sequence + '_sync')
    cam2world_root = os.path.join(data_root, 'data_poses', sequence, 'cam0_to_world.txt')
    print('{0} : {1}'.format(sequence, frame_start))
    mesh_intersection = Dataset(cam2world_root, bbx_root, data_root, sequence, split, frame_num, frame_start, visible_id_root)
    train_loader = torch.utils.data.DataLoader(mesh_intersection, batch_size=1, shuffle=False, num_workers=0)
    xyz_list = []
    pose = []
    for i, data in enumerate(train_loader):
        pose_00, pose_01, xyz = data
        pose.append(pose_00)
        pose.append(pose_01)
        xyz_list.append(xyz)
    pose = np.concatenate(pose)
    xyz_list = np.concatenate(xyz_list, 1)
    max_xyz = np.array(xyz_list.max(1))
    min_xyz = np.array(xyz_list.min(1))
    aabb = np.stack((max_xyz, min_xyz))
    aabb_vert= []
    n = 3
    gray_map = np.empty((1<<n, n))
    for i in range(0,1<<n):
        gray = i^(i>>1)
        gray_bin = np.binary_repr(gray, width=n)
        gray_map[i,:] = list(map(int, gray_bin))
    dist = []
    for offset in gray_map:
        vertex = np.empty((3,))
        for i in range(3):
            vertex[i] = aabb[int(offset[i]), 0, i]
        dist.append(np.linalg.norm((vertex - pose), axis=1))
    dist = np.stack(dist)
    dist = dist.max(0)
    dist_min = dist.min(0)
    dist_idx = dist.argmin(0)
    start_pose = pose[dist_idx]
    print("center_pose:",start_pose)
    print("dist_idx:",dist_idx)
    print("dist_min:",dist_min)
