import numpy as np
import os
import copy
import trimesh
import time
import torch
import cv2
from lib.config import cfg
from glob import glob
from tools.kitti360scripts.helpers.annotation import Annotation2D, Annotation2DInstance, Annotation3D
from lib.utils.data_utils import readVariable, loadCalibrationCameraToPose, build_rays

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
    if mask_at_box.sum()>0:
        return True
    else:
        return False

class Dataset:
    def __init__(self, cam2world_root, img_root, bbx_root, data_root, sequence, split, frame_num, frame_start, use_stereo):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.image_root = img_root
        self.split = split
        self.ratio = .5
        self.use_stereo = use_stereo
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
        if self.use_stereo:
            self.intrinsics = self.K_01
        else:
            self.intrinsics = self.K_00
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, self.camToPose), np.linalg.inv(self.R_rect))
        if self.use_stereo:
           self.cam2world_dict = self.cam2world_dict_01
        else:
           self.cam2world_dict = self.cam2world_dict_00
        self.intrinsics[:2] = self.intrinsics[:2] * self.ratio
    
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
            self.image_ids = test_ids
        self.metas = {}
        for idx in self.image_ids:
            pose = self.cam2world_dict[idx].astype(np.float32)
            pose[:3, 3] -= self.cam2world_dict[self.start][:3, 3]
            filename = '000000'+str(idx)+'.png'
            self.metas[idx-self.start] = self.func((pose, filename, idx))
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
        ext_pose, filename, idx = input_tuple
        H, W = int(self.height*self.ratio), int(self.width*self.ratio)
        rays = build_rays(self.intrinsics, ext_pose, H, W)
        filename, _ = os.path.splitext(filename)
        filename = os.path.join(visible_id_root, self.sequence, filename + '.txt')
        with open(filename, "r") as f:
            data = f.read().splitlines()
            annotationId = np.array(list(map(int, data)))
        return (rays.astype(np.float32), W, H, np.unique(annotationId), idx)
    
    def generate_npy(self, idx, bbx_npy_root):
        annotationId_list = []
        filename, _ = os.path.splitext('000000'+str(idx)+'.png')
        filename = os.path.join(self.visible_id, filename + '.txt')
        with open(filename, "r") as f:  
            data = f.read().splitlines() 
            annotationId = np.array(list(map(int, data)))  
        annotationId_list.append(np.unique(annotationId))
        annotationId_list = np.concatenate(annotationId_list)
        np.save(bbx_npy_root, np.unique(annotationId_list))

    def __getitem__(self, index):
        initial_time = time.time()
        rays, W, H, annotationId_list, idx = self.metas[index]
        bbx_npy_root = os.path.join(self.data_root, 'bbx', self.sequence)
        if os.path.exists(bbx_npy_root) == False:
            os.system('mkdir -p {}'.format(bbx_npy_root))
        image_path = os.path.join(img_root, '000000'+str(idx)+'.png')
        filename = os.path.basename(image_path)[:-4]
        fileroot = os.path.join(bbx_npy_root, filename + '.npy')
        if os.path.exists(fileroot) == True:
            os.system('rm {}'.format(fileroot))
        self.generate_npy(idx, fileroot)
        annotationId_list = np.load(fileroot)
        bbx = []
        bbx_intersection_root = os.path.join(data_root, 'bbx_intersection', self.sequence)
        if os.path.exists(bbx_intersection_root) == False:
            os.system('mkdir -p {}'.format(bbx_intersection_root))
        if self.use_stereo:
            if os.path.exists(os.path.join(bbx_intersection_root,str(index+self.start)+'_01.npz')) == True:
                return 0
        else:
            if os.path.exists(os.path.join(bbx_intersection_root,str(index+self.start)+'.npz')) == True:
                return 0
        for annotationId in annotationId_list:
            if annotationId in self.bbx_static.keys():
                temp = copy.deepcopy(self.bbx_static[annotationId])
                xyz = self.bbx_static[annotationId].vertices - self.cam2world_dict[self.start][:3, 3]
                max_xyz = np.max(xyz, axis = 0)
                min_xyz = np.min(xyz, axis = 0)
                bounds = np.stack([min_xyz, max_xyz], axis=0)
                temp.vertices = xyz
                bbx.append((temp, xyz.shape[0], bounds))
        bbox_max = 10
        intersection = np.full((H, W, bbox_max, 3), -1., dtype=np.float32)
        obj_num = 0
        index_ray_all = np.array([])
        depth_all = np.array([])
        obj_all = np.array([])
        rays_num = H * W
        rays = rays[:rays_num]
        len_bbox = len(bbx)
        for obj, bbx_vertex, bounds in bbx:
            start_time = time.time()
            obj_num += 1
            mesh_tri = trimesh.Trimesh(vertices = obj.vertices, faces = obj.faces)
            if bbx_vertex == 8 and not get_near_far(bounds, rays[..., 0:3], rays[..., 3:6]):
                continue
            else:
                ray_origins = rays[..., 0:3]
                ray_directions = rays[..., 3:6]
                locations, index_rays, index_tris = mesh_tri.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)
            print('frame {0}: obj{1}/{2} costs {3} s'.format(index+self.start, obj_num, len_bbox, time.time()-start_time))  
            if len(locations) == 0:
                continue
            else:
                index_ray_all = np.append(index_ray_all, index_rays)
                depth = np.linalg.norm(locations-rays[index_rays,:3], axis=1)
                depth_all = np.append(depth_all, depth)
                obj = np.array([obj.annotationId]).repeat(len(index_rays))
                obj_all = np.append(obj_all, obj)
        
        # bbx_sort
        start_time = time.time()
        index_sort_all = np.argsort(index_ray_all,kind='mergesort')
        index_ray_all = index_ray_all[index_sort_all].astype(int)
        depth_all = depth_all[index_sort_all]
        obj_all = obj_all[index_sort_all]
        even_index = np.array([2*i for i in range(len(index_ray_all)//2)])
        odd_index = np.array([2*i+1 for i in range(len(index_ray_all)//2)])
        index_ray = index_ray_all[even_index]
        obj = obj_all[even_index]
        depth_in = depth_all[even_index]
        depth_out = depth_all[odd_index]
        batch = np.dstack((obj,depth_in,depth_out)) 
        ray_unique = np.unique(index_ray)
        index_ray_first = []
        for i in range(len(index_ray)):
            if i == 0 or index_ray[i] != index_ray[i-1]:
                index_ray_first.append(i)
        index_ray_first = np.array(index_ray_first)
        index_ray_first = np.append(index_ray_first, len(index_ray))
        for i in range(len(ray_unique)):
            idx = index_ray_first[i]
            idx_next = index_ray_first[i+1]
            temp = batch[0][idx:idx_next]
            intersection[ray_unique[i]//W][ray_unique[i]%W][:(idx_next-idx)] = temp[np.argsort(temp[:,1])][:bbox_max]
        intersection = intersection.reshape(-1,3)
        temp = copy.deepcopy(intersection)
        intersection[...,1] = np.min(temp[...,1:3], axis=1)
        intersection[...,2] = np.max(temp[...,1:3], axis=1)
        intersection = intersection.reshape(H, W, bbox_max, 3)
        final_depths = intersection[...,1:3].astype(np.float16)
        final_annotations = np.full((H, W, bbox_max, 2), -1., dtype=np.int16)
        final_annotations[...,0] = intersection[...,0].astype(np.int16)
        for i in range(H):
            for j in range(W):
                for k in range(bbox_max):
                    if final_annotations[i][j][k][0] != -1.:
                        final_annotations[i][j][k][1] = self.bbx_static[int(final_annotations[i][j][k][0])].semanticId
        if self.use_stereo:
            np.savez(os.path.join(bbx_intersection_root,str(index+self.start)+'_01.npz'), final_depths, final_annotations)
        else:
            np.savez(os.path.join(bbx_intersection_root,str(index+self.start)+'.npz'), final_depths, final_annotations)
        print("frame {0} is done, costs {1} s".format(index+self.start, time.time() - initial_time))
        return 0

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
    frame_start = cfg.intersection_start_frame
    frame_num = cfg.intersection_frames
    data_root = 'datasets/KITTI-360'
    use_stereo = cfg.use_stereo
    split = 'train'
    sequence = '0000'
    gt_static_frames_root = os.path.join(data_root, 'gt_static_frames.txt')
    bbx_root = os.path.join(data_root, 'data_3d_bboxes')
    visible_id_root = os.path.join(data_root, 'visible_id')
    sequence = os.path.join('2013_05_28_drive_' + sequence + '_sync')
    if use_stereo:
        img_root = os.path.join(data_root, sequence, 'image_01/data_rect')
    else:
        img_root = os.path.join(data_root, sequence, 'image_00/data_rect')
    cam2world_root = os.path.join(data_root, 'data_poses', sequence, 'cam0_to_world.txt')
    print('{0} : {1}'.format(sequence, int(frame_start)))
    mesh_intersection = Dataset(cam2world_root, img_root, bbx_root, data_root, sequence, split, frame_num, frame_start, use_stereo)
    train_loader = torch.utils.data.DataLoader(mesh_intersection, batch_size=1, shuffle=False, num_workers=32)
    for i, data in enumerate(train_loader):
        print('{0} / {1} is done.'.format(i+frame_start, len(train_loader)+frame_start-1))