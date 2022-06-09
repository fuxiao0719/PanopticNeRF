import numpy as np
import open3d as o3d
import glob
import os
from scipy.spatial import KDTree
from lib.config import cfg, args
import cv2

intrinsics = np.array([[552.554261, 0, 682.049453],
                       [0., 552.554261, 238.769549], 
                       [0., 0., 1.]])

def depth_to_point_cloud(depth):
    H,W = depth.shape
    X,Y = np.meshgrid(range(W), range(H))
    XYZ = np.concatenate((X[...,None], Y[...,None], np.ones_like(X[...,None])), axis=2)
    XYZ = XYZ.reshape(-1, 3)
    point_cloud = (np.linalg.inv(intrinsics) @ XYZ.T).T
    #point_cloud = point_cloud / np.linalg.norm(point_cloud, axis=-1, keepdims=True)
    point_cloud = point_cloud * depth.flatten().reshape(-1,1)
    point_cloud = point_cloud[depth.reshape(-1)>0]
    #import ipdb;ipdb.set_trace()
    uv = XYZ[depth.reshape(-1)>0, :2]
    return point_cloud, uv

def cam_to_world(points, cam2world):
    points_world = (cam2world @ np.concatenate((points, np.ones_like(points[:,0:1])), axis=1).T).T
    points_world = points_world[:,:3]
    return points_world

def parse_cam2world(cam2world_root):
    cam2world_dict = {}
    for line in open(cam2world_root, 'r').readlines():
        value = list(map(float, line.strip().split(" ")))
        cam2world_dict[value[0]] = np.array(value[1:]).reshape(4, 4)
    return cam2world_dict

def eval_consistency(frames, cam2world_dict, lidar_dir, semantic_dir, seq):
    # load lidar depth
    lidars = []
    pixel_locs = []
    for i,f in enumerate(frames):
        lidar_file = glob.glob(os.path.join(lidar_dir, '2013_05_28_drive_%04d_*'%seq, '%010d_0.npy'%f))
        # assert (len(lidar_file)==1)
        lidar = np.load(lidar_file[0])
        lidar[lidar>100]=0
        lidar_cam, uv = depth_to_point_cloud(lidar)
        lidar_world = cam_to_world(lidar_cam, cam2world_dict[f])
        lidars.append(lidar_world)
        pixel_locs.append(uv)
    
    # load semantics of the lidar points
    lidar_semantics = []
    for i,f in enumerate(frames):
        semantic_file = glob.glob(os.path.join(semantic_dir, 'panopticnerf_test', '2', 'img%4d_pred_semantic.npy'%f))[0]
        semantic = np.load(semantic_file)[..., 0]
        semantic = cv2.resize(semantic, (1408,376), interpolation = cv2.INTER_NEAREST)
        lidar_semantics.append(semantic[pixel_locs[i][:,1], pixel_locs[i][:,0]])

    # find nearest neighbor
    tree = KDTree(lidars[0])
    distances, indices = tree.query(lidars[1])
    semantic_nn = lidar_semantics[0][indices]
    semantic = lidar_semantics[1]
    
    mask_matched = distances<thres
    mask_consistent = np.logical_and(mask_matched, semantic==semantic_nn)
    print('frame {0}-{1}:{2}/{3}'.format(frames[0], frames[1], mask_consistent.sum(), mask_matched.sum()))
    return mask_matched.sum(), mask_consistent.sum()

if __name__=='__main__':
    num_matched_all = 0
    num_consistent_all = 0
    print('eval consistency')
    thres = cfg.consistency_thres
    lidar_dir = 'datasets/KITTI-360/lidar_depth'
    semantic_dir = 'data/result/panopticnerf/'
    seq = 0
    frames_seq0_all = cfg.val_list
    kitti360Path = 'datasets/KITTI-360'
    cam2world_root = os.path.join(kitti360Path, 'data_poses', '2013_05_28_drive_%04d_sync' % seq, 'cam0_to_world.txt') 
    cam2world_dict = parse_cam2world(cam2world_root)
    for i in range(len(frames_seq0_all)-1):
        seq = 0
        frames = []
        if frames_seq0_all[i+1] - frames_seq0_all[i] > 10:
            continue
        else:
            frames.append(frames_seq0_all[i])
            frames.append(frames_seq0_all[i+1])
        # evaluate
        num_matched, num_consistent = eval_consistency(frames, cam2world_dict, lidar_dir, semantic_dir, seq)
        num_matched_all += num_matched
        num_consistent_all += num_consistent
    print('MC : %.02f  %d/%d' % (num_consistent_all/num_matched_all*100, num_consistent_all, num_matched_all))
