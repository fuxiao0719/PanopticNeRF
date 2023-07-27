import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
import numpy as np
import torch.nn.functional as F
import torch
import cv2
import tqdm
import copy
from lib.config import cfg
import os
from tools.kitti360scripts.helpers.labels import id2label, labels
import torch.nn as nn
from torch.functional import norm
import scipy.stats as st

_color_map_errors = np.array([
    [149,  54, 49],     #0: log2(x) = -infinity
    [180, 117, 69],     #0.0625: log2(x) = -4
    [209, 173, 116],    #0.125: log2(x) = -3
    [233, 217, 171],    #0.25: log2(x) = -2
    [248, 243, 224],    #0.5: log2(x) = -1
    [144, 224, 254],    #1.0: log2(x) = 0
    [97, 174,  253],    #2.0: log2(x) = 1
    [67, 109,  244],    #4.0: log2(x) = 2
    [39,  48,  215],    #8.0: log2(x) = 3
    [38,   0,  165],    #16.0: log2(x) = 4
    [38,   0,  165]    #inf: log2(x) = inf
]).astype(float)

def color_error_image(errors, scale=1, mask=None, BGR=True):
    """
    Color an input error map.
    
    Arguments:
        errors -- HxW numpy array of errors
        [scale=1] -- scaling the error map (color change at unit error)
        [mask=None] -- zero-pixels are masked white in the result
        [BGR=True] -- toggle between BGR and RGB
    Returns:
        colored_errors -- HxWx3 numpy array visualizing the errors
    """
    
    errors_flat = errors.flatten()
    errors_color_indices = np.clip(np.log2(errors_flat / scale + 1e-5) + 5, 0, 9)
    i0 = np.floor(errors_color_indices).astype(int)
    f1 = errors_color_indices - i0.astype(float)
    colored_errors_flat = _color_map_errors[i0, :] * (1-f1).reshape(-1,1) + _color_map_errors[i0+1, :] * f1.reshape(-1,1)

    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 255

    if not BGR:
        colored_errors_flat = colored_errors_flat[:,[2,1,0]]

    return colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.int)

_color_map_depths = np.array([
    [0, 0, 0],          # 0.000
    [255, 0, 0],        # 0.114
    [0, 0, 255],        # 0.299
    [255, 0, 255],      # 0.413
    [0, 255, 0],        # 0.587
    [0, 255, 255],      # 0.701
    [255, 255,  0],     # 0.886
    [255, 255,  255],   # 1.000
    [255, 255,  255],   # 1.000
]).astype(float)

_color_map_bincenters = np.array([
    0.0,
    0.114,
    0.299,
    0.413,
    0.587,
    0.701,
    0.886,
    1.000,
    2.000, # doesn't make a difference, just strictly higher than 1
])

def color_depth_map(depths, scale=None):
    """
    Color an input depth map.
    
    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)
    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """
    
    if scale is None:
        scale = depths.max()

    values = np.clip(depths.flatten() / scale, 0, 1)

    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1,-1)) * np.arange(0,9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1-alphas).reshape(-1,1) + _color_map_depths[lower_bin + 1] * alphas.reshape(-1,1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)

def assigncolor(globalids, gttype='semantic'):
    if not isinstance(globalids, (np.ndarray, np.generic)):
        globalids = np.array(globalids)[None]
    color = np.zeros((globalids.size, 3))
    # semanticid = globalids
    for uid in np.unique(globalids):
        # semanticid, instanceid = global2local(uid)
        if gttype == 'semantic':
            try:
                color[globalids == uid] = id2label[uid].color
            except:
                color[globalids == uid] = (0, 0, 0)  # stuff objects in instance mode
                print("warning! unkown category!")
        else:
            color[globalids == uid] = (96, 96, 96)  # stuff objects in instance mode
    color = color.astype(np.float) / 255.0
    return color

def draw_sample(mask, result_dir):
    h = mask.shape[0]
    w = mask.shape[1]
    draw_temp = np.zeros((h, w, 3), dtype=np.float32)
    draw_temp[mask.detach().cpu()] = 1.
    cv2.imwrite('{}/draw_temp.png'.format(result_dir), (draw_temp.reshape(h, w, 3) * 255).astype(np.uint8))

class Visualizer:
    def __init__(self, ):
        self.color_crit = lambda x, y: ((x - y)**2).mean()
        self.mse2psnr = lambda x: -10. * np.log(x) / np.log(torch.tensor([10.]))
        self.psnr = []

    def visualize(self, output, batch):
        b = len(batch['rays'])
        for b in range(b):
            img_id = int(batch["meta"]["tar_idx"].item())
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            if batch['stereo_num'].item() == 0:
                suffix = '00'
            else:
                suffix = '01'
            result_dir = os.path.join(cfg.result_dir, batch['meta']['sequence'][0])
            os.system("mkdir -p {}".format(result_dir))

            # rendering mode
            render_boundary_mode = False
            if cfg.result_dir[-2:] == 'ft':
                render_boundary_mode = True

            # instance finetuning
            instance_finetuning = False
            if (render_boundary_mode == False) and (cfg.use_post_processing == False) and (cfg.merge_instance == True):
                result_dir_ft = os.path.join(cfg.result_dir+'_ft', batch['meta']['sequence'][0])
                instance_finetuning = True
            
            # rgb
            pred_img = torch.clamp(output['rgb_0'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
            gt_img = batch['rays_rgb'][b].reshape(h, w, 3).detach().cpu().numpy()

            # fisheye mask
            if batch['stereo_num'].item() == 0:
                fisheye_pedestal = cv2.imread('datasets/KITTI-360/fisheye/mask_left.png')
                fisheye_pedestal = cv2.resize(fisheye_pedestal, (350, 350), interpolation=cv2.INTER_AREA)
                mask_fisheye = np.load('datasets/KITTI-360/fisheye/mask_left_fisheye.npy')[::4,::4]
            else:
                fisheye_pedestal = cv2.imread('datasets/KITTI-360/fisheye/mask_right.png')
                fisheye_pedestal = cv2.resize(fisheye_pedestal, (350, 350), interpolation=cv2.INTER_AREA)
                mask_fisheye = np.load('datasets/KITTI-360/fisheye/mask_right_fisheye.npy')[::4,::4]

            # fisheye rgb
            fisheye_pred_img = torch.zeros((350**2,3), dtype=torch.float32)
            fisheye_pred_img[batch['fisheye_inds_valid']] = output['fisheye_rgb_0'].detach().cpu()
            fisheye_pred_img = (fisheye_pred_img.numpy().reshape(350,350,3)[..., ::-1]*255).astype(np.uint8)
            fisheye_pred_img[mask_fisheye] = fisheye_pedestal[mask_fisheye]
            fisheye_gt_img = torch.zeros((350**2,3), dtype=torch.float32)
            fisheye_gt_img[batch['fisheye_inds_valid']] = batch['fisheye_rays_rgb'].detach().cpu()
            fisheye_gt_img = (fisheye_gt_img.numpy().reshape(350,350,3)[..., ::-1]*255).astype(np.uint8)
            fisheye_gt_img[mask_fisheye] = fisheye_pedestal[mask_fisheye]
            fisheye_img_vis = np.concatenate((fisheye_pred_img, fisheye_gt_img), axis=1)
            
            # depth
            pred_depth = output['depth_0'][b].reshape(h, w).detach().cpu().numpy()
            pred_depth_rgb = copy.deepcopy(pred_depth)
            pred_depth_rgb[pred_depth_rgb == 0] = 100000
            pred_depth_rgb = color_depth_map(pred_depth_rgb, 100)

            # fisheye depth
            fisheye_pred_depth = np.zeros((350**2), dtype=np.float32)
            fisheye_pred_depth_valid = output['fisheye_depth_0'][b].detach().cpu().numpy()
            fisheye_pred_depth[batch['fisheye_inds_valid'].detach().cpu().numpy()] = fisheye_pred_depth_valid
            fisheye_pred_depth_rgb = copy.deepcopy(fisheye_pred_depth)
            fisheye_pred_depth_rgb = color_depth_map(fisheye_pred_depth_rgb.reshape(350, 350), 100)
            fisheye_pred_depth_rgb[mask_fisheye] = fisheye_pedestal[mask_fisheye]

            # normal
            pred_normal = (output['normal_map_0'][0].detach().cpu().numpy().reshape(h, w, 3) + 1 ) / 2.
            pred_normal[np.isnan(pred_normal)] = 0.

            # semantic
            _, pred_semantic = output['semantic_map_0'][b].max(1)
            _, fix_semantic = output['fix_semantic_map_0'][b].max(1)
            mask_fix = (output['semantic_bbox_gt'].reshape(1, -1, cfg.samples_all, 50).sum(dim=-1).sum(dim=-1) == cfg.samples_all).reshape(-1)
            pred_semantic[mask_fix] = fix_semantic[mask_fix]
            if instance_finetuning == True:
                pred_semantic_ft = torch.from_numpy(np.load('{}/img{:04d}_pred_semantic_{}.npy'.format(result_dir_ft, img_id, suffix))).cuda().reshape(-1)
                mask_build_semantic = (pred_semantic_ft == 11) & (pred_semantic == 11)
            pred_semantic_rgb = assigncolor(pred_semantic.reshape(h, w).detach().cpu().numpy().reshape(-1), 'semantic')
            mix_pred_semantic = cv2.addWeighted((gt_img[..., ::-1] * 255).astype(np.uint8), 0.5, (pred_semantic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            
            # fisheye semantic
            _, fisheye_fix_semantic_valid = output['fisheye_fix_semantic_map_0'][b].max(1)
            fisheye_pred_semantic = torch.zeros((350**2), dtype=torch.long)
            _, fisheye_pred_semantic_valid = output['fisheye_semantic_map_0'][b].max(1)
            _, fisheye_fix_semantic_valid = output['fisheye_fix_semantic_map_0'][b].max(1)
            fisheye_mask_fix = (output['fisheye_semantic_bbox_gt'].reshape(1, -1, cfg.samples_all, 50).sum(dim=-1).sum(dim=-1) == cfg.samples_all).reshape(-1)
            fix_label_fine = [9]
            for label in fix_label_fine:
                fisheye_mask_fix = fisheye_mask_fix | (fisheye_fix_semantic_valid == label).to(fisheye_mask_fix)
            fisheye_pred_semantic_valid[fisheye_mask_fix] = fisheye_fix_semantic_valid[fisheye_mask_fix]
            if cfg.use_pseudo_fusion == True:
                fisheye_pseudo_semantic = batch['fisheye_pseudo_label'][0].to(fisheye_pred_semantic_valid)
                mask_fisheye_pseudo_fusion = (fisheye_pseudo_semantic != 23) & (fisheye_pred_semantic_valid == 23)
                fisheye_pred_semantic_valid[mask_fisheye_pseudo_fusion] = fisheye_pseudo_semantic[mask_fisheye_pseudo_fusion]
            fisheye_pred_semantic[batch['fisheye_inds_valid'][0]] = fisheye_pred_semantic_valid.detach().cpu()
            fisheye_pred_semantic = fisheye_pred_semantic.reshape(350, 350).numpy()
            if instance_finetuning == True:
                fisheye_pred_semantic_ft = np.load('{}/img{:04d}_fisheye_pred_semantic_{}.npy'.format(result_dir_ft, img_id, suffix)).reshape(-1)
                mask_build_fisheye_semantic = (fisheye_pred_semantic_ft == 11) & (fisheye_pred_semantic.reshape(-1) == 11)
            fisheye_pred_semantic_rgb = assigncolor(fisheye_pred_semantic.reshape(-1), 'semantic')
            mix_fisheye_pred_semantic_rgb = cv2.addWeighted(fisheye_gt_img, 0.5, (fisheye_pred_semantic_rgb.reshape(350, 350, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            mix_fisheye_pred_semantic_rgb[mask_fisheye] = fisheye_pedestal[mask_fisheye]
            
            # instance_regression
            eval_instance_list = [11, 26, 29, 30]
            instance_map_pre =  output['instance_map_0'][b]              
            instance_map_post = torch.zeros_like(instance_map_pre)
            for (semantic_id, id_list) in batch['semantic2id'].items():                                
                semantic_mask = (pred_semantic==semantic_id)
                for id_ in id_list:
                    instance_map_post[semantic_mask, id_] = instance_map_pre[semantic_mask, id_]
            _, instance_temp = instance_map_post.max(1)

            if instance_finetuning == True:
                instance_temp_ft = torch.from_numpy(np.load('{}/img{:04d}_instance_temp_{}.npy'.format(result_dir_ft, img_id, suffix))).to(instance_temp)
                instance_temp[mask_build_semantic] = instance_temp_ft[mask_build_semantic]
            else:
                instance_temp_ft = copy.deepcopy(instance_temp.detach().cpu().numpy())

            if cfg.use_post_processing == True :
                pred_semantic_redis = pred_semantic.reshape(h, w)
                instance_temp_redis = copy.deepcopy(instance_temp.reshape(h, w))
                mask_build = pred_semantic_redis == 11
                build_inst_ids = torch.unique(instance_temp_redis[mask_build]).detach().cpu().tolist()
                build_insts = {}
                build_insts_outstretch = {}
                for inst_id in build_inst_ids:
                    inst_id_temp = mask_build & (instance_temp_redis == inst_id)
                    if inst_id_temp.sum() < 50:
                        continue
                    build_insts[inst_id] = inst_id_temp
                    inst_id_temp_outstretch = torch.zeros((inst_id_temp.shape[0], inst_id_temp.shape[1]+1)).to(inst_id_temp).to(torch.bool)
                    inst_id_temp_outstretch[:,1:] = inst_id_temp
                    inst_id_temp_outstretch_left = inst_id_temp_outstretch[:,:-1]
                    mask_inst_id_temp = inst_id_temp == True
                    inst_id_temp_outstretch_left[mask_inst_id_temp] = inst_id_temp[mask_inst_id_temp]
                    inst_id_temp_outstretch = inst_id_temp_outstretch_left
                    build_insts_outstretch[inst_id] = inst_id_temp_outstretch
                
                query_inst_ids = list(build_insts_outstretch.keys())
                key_inst_ids = copy.deepcopy(query_inst_ids)
                near_inst_pair = []
                for query_inst_id in query_inst_ids:
                    for key_inst_id in key_inst_ids:
                        if key_inst_id == query_inst_id or (key_inst_id, query_inst_id) in near_inst_pair:
                            continue
                        if (build_insts_outstretch[query_inst_id] & build_insts_outstretch[key_inst_id]).sum() > 0:
                            near_inst_pair.append((query_inst_id, key_inst_id))

                for inst_pair in tqdm.tqdm(near_inst_pair):
                    inst_i, inst_j = inst_pair
                    sect_points = build_insts_outstretch[inst_i] & build_insts_outstretch[inst_j]
                    y_points, x_points = torch.where(sect_points==True)

                    cut_len_head = int(len(y_points) * 0.4)
                    cut_len_tail = int(len(y_points) * 0.1)
                    cut_y_points = y_points[cut_len_head:][cut_len_tail:]
                    cut_x_points = x_points[cut_len_head:][cut_len_tail:]

                    slope = st.linregress(cut_y_points.cpu().numpy(), cut_x_points.cpu().numpy())[0]
                    margin = 30
                    y_min = torch.clamp(torch.min(y_points) - margin, min=0)
                    y_max = torch.clamp(torch.max(y_points) + margin, max=h)
                    x_min = torch.clamp(torch.min(x_points) - margin, min=0)
                    x_max = torch.clamp(torch.max(x_points) + margin, max=w)

                    merge_i_j = torch.zeros((h,w)).to(torch.bool)
                    merge_i_j[y_min:y_max, x_min:x_max] = True
                    merge_i_j = merge_i_j.to(build_insts[inst_i]) & (build_insts[inst_i] | build_insts[inst_j])
                    merge_i_j_y_points , merge_i_j_x_points = torch.where(merge_i_j==True)
                    merge_i_j_x_points = torch.unique(merge_i_j_x_points)
                    merge_i_j_y_points = torch.unique(merge_i_j_y_points)

                    if slope < 0 :
                        b_min = (merge_i_j_x_points[0] - (slope * merge_i_j_y_points[0])).to(torch.int16)
                        b_max = (merge_i_j_x_points[-1] - (slope * merge_i_j_y_points[-1])).to(torch.int16)
                    else:
                        b_min = (merge_i_j_x_points[0] - (slope * merge_i_j_y_points[-1])).to(torch.int16)
                        b_max = (merge_i_j_x_points[-1] - (slope * merge_i_j_y_points[0])).to(torch.int16)

                    for b_ in range(b_min, b_max):
                        y_list = []
                        x_list = []
                        for y_ in range(h):
                            x_ = slope * y_ + b_
                            x_ = x_.astype(np.int16)
                            if x_ >= w:
                                continue
                            try:
                                if merge_i_j[y_, x_] == True:
                                    y_list.append(y_)
                                    x_list.append(x_)
                            except:
                                import ipdb; ipdb.set_trace()
                        if len(x_list) == 0:
                            continue
                        instance_temp_redis_b = instance_temp_redis[torch.Tensor(y_list).long(), torch.Tensor(x_list).long()]
                        inst_ids, count_inst_ids = torch.unique(instance_temp_redis_b, return_counts=True)
                        if len(inst_ids) == 1:
                            continue
                        max_count_idx = torch.argmax(count_inst_ids)
                        instance_temp_redis_b_temp = torch.ones_like(instance_temp_redis_b) * inst_ids[max_count_idx]
                        instance_temp_redis[torch.Tensor(y_list).long(), torch.Tensor(x_list).long()] = instance_temp_redis_b_temp
            
                instance_temp = instance_temp_redis
                pred_instance = instance_temp_redis.reshape(-1)
                linear_fitting_npy = np.concatenate((pred_semantic.detach().cpu().numpy()[...,None], pred_instance.detach().cpu().numpy()[...,None]), axis=1)
                np.save('{}/img{:04d}_linear_regression_{}.npy'.format(result_dir, img_id, suffix), linear_fitting_npy)

                return

            pred_instance = np.copy(instance_temp.cpu().numpy().reshape(h,w))
            pred_instance += 50
            instance_temp = instance_temp.detach().cpu().numpy().reshape(h,w)
            instance_list = np.unique(instance_temp)
            for inst in instance_list:
                instance_temp[instance_temp == inst] = batch['instance2id'][inst].detach().cpu().numpy()  
            id2semantic = batch['id2semantic']                                                  
            id2semantic = {(k + 50): v.item() for (k, v) in id2semantic.items()}
            instance_temp_rgb = np.zeros((h, w, 3))
            instance_temp = instance_temp / (100000) * 256**3
            instance_temp_rgb[..., 0] = instance_temp % 256
            instance_temp_rgb[..., 1] = instance_temp / 256 % 256
            instance_temp_rgb[..., 2] = instance_temp / (256 * 256)
            pred_semantic = pred_semantic.reshape(h, w).detach().cpu().numpy()
            panoptic_mask = np.ones_like(pred_semantic) > 1
            color = pred_semantic_rgb.reshape(h, w, 3)
            pred_panoptic_rgb = np.zeros_like(color)
            for id_ in eval_instance_list:
                panoptic_mask = panoptic_mask | (pred_semantic == id_)
            panoptic_mask = ~panoptic_mask
            pred_panoptic_rgb[panoptic_mask] = color[panoptic_mask]
            instance_rgb = instance_temp_rgb.reshape(h, w, 3)[..., ::-1] * 255
            pred_panoptic_rgb[~panoptic_mask] = instance_rgb[~panoptic_mask]

            mix_pred_panoptic_rgb = cv2.addWeighted((gt_img[..., ::-1] * 255).astype(np.uint8), 0.5, (pred_panoptic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            panoptic_id_map = np.zeros((h, w))
            panoptic_id_map[panoptic_mask] = pred_semantic[panoptic_mask]
            panoptic_id_map[~panoptic_mask] = pred_instance[~panoptic_mask]

            # fisheye instance
            fisheye_instance_map_pre = output['fisheye_instance_map_0'][b]                                    
            fisheye_instance_map_post = torch.zeros_like(fisheye_instance_map_pre)
            for (semantic_id, id_list) in batch['fisheye_semantic2id'].items():                                
                fisheye_semantic_mask = (fisheye_pred_semantic_valid==semantic_id)
                for id_ in id_list:
                    fisheye_instance_map_post[fisheye_semantic_mask, id_] = fisheye_instance_map_pre[fisheye_semantic_mask, id_]
            _, fisheye_instance_temp = fisheye_instance_map_post.max(1)

            if instance_finetuning == True:
                fisheye_instance_temp_ft = torch.from_numpy(np.load('{}/img{:04d}_fisheye_instance_temp_{}.npy'.format(result_dir_ft, img_id, suffix))).to(fisheye_instance_temp)
                mask_build_fisheye_semantic = mask_build_fisheye_semantic[batch['fisheye_inds_valid'][0].cpu()]
                fisheye_instance_temp[mask_build_fisheye_semantic] = fisheye_instance_temp_ft[mask_build_fisheye_semantic]
            else:
                fisheye_instance_temp_ft = copy.deepcopy(fisheye_instance_temp.detach().cpu().numpy())

            fisheye_pred_instance = np.copy(fisheye_instance_temp.cpu().numpy())
            fisheye_pred_instance += 50
            fisheye_instance_temp = fisheye_instance_temp.detach().cpu().numpy()
            instance_list = np.unique(fisheye_instance_temp)
            for inst in instance_list:
                fisheye_instance_temp[fisheye_instance_temp == inst] = batch['fisheye_instance2id'][inst].detach().cpu().numpy()  
            fisheye_id2semantic = batch['fisheye_id2semantic']                                                  
            fisheye_id2semantic = {(k + 50): v.item() for (k, v) in fisheye_id2semantic.items()}
            fisheye_instance_temp_rgb = np.zeros((fisheye_pred_semantic_valid.shape[0], 3))
            fisheye_instance_temp = fisheye_instance_temp / (100000) * 256**3
            fisheye_instance_temp_rgb[..., 0] = fisheye_instance_temp % 256
            fisheye_instance_temp_rgb[..., 1] = fisheye_instance_temp / 256 % 256
            fisheye_instance_temp_rgb[..., 2] = fisheye_instance_temp / (256 * 256)
            fisheye_pred_semantic_valid = fisheye_pred_semantic_valid.detach().cpu().numpy()
            fisheye_panoptic_mask = np.ones_like(fisheye_pred_semantic_valid) > 1
            fisheye_rgb = fisheye_pred_semantic_rgb.reshape(-1,3)[batch['fisheye_inds_valid'].cpu()][0]
            fisheye_pred_panoptic_rgb = np.zeros_like(fisheye_rgb)
            for id_ in eval_instance_list:
                fisheye_panoptic_mask = fisheye_panoptic_mask | (fisheye_pred_semantic_valid == id_)
            fisheye_panoptic_mask = ~fisheye_panoptic_mask
            fisheye_pred_panoptic_rgb[fisheye_panoptic_mask] = fisheye_rgb[fisheye_panoptic_mask]
            fisheye_instance_rgb = fisheye_instance_temp_rgb[..., ::-1] * 255
            fisheye_pred_panoptic_rgb[~fisheye_panoptic_mask] = fisheye_instance_rgb[~fisheye_panoptic_mask]
            fisheye_panoptic_rgb_mask = fisheye_pred_panoptic_rgb
            fisheye_pred_panoptic_rgb = np.zeros((350**2, 3), dtype=np.float64)
            fisheye_pred_panoptic_rgb[batch['fisheye_inds_valid'][0].cpu()] = fisheye_panoptic_rgb_mask

            mix_fisheye_pred_panoptic_rgb = cv2.addWeighted((fisheye_gt_img).astype(np.uint8), 0.5, (fisheye_pred_panoptic_rgb.reshape(350, 350, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            mix_fisheye_pred_semantic_rgb[mask_fisheye] = fisheye_pedestal[mask_fisheye]
            mix_fisheye = np.concatenate((mix_fisheye_pred_semantic_rgb, mix_fisheye_pred_panoptic_rgb), axis=1)
            fisheye_panoptic_id_map = np.zeros_like(fisheye_pred_semantic_valid)
            fisheye_panoptic_id_map[fisheye_panoptic_mask] = fisheye_pred_semantic_valid[fisheye_panoptic_mask]
            fisheye_panoptic_id_map[~fisheye_panoptic_mask] = fisheye_pred_instance[~fisheye_panoptic_mask]

            # save output
            if render_boundary_mode == False:
                cv2.imwrite('{}/img{:04d}_pred_img_{}.png'.format(result_dir, img_id, suffix), (pred_img[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_fisheye_pred_img_{}.png'.format(result_dir, img_id, suffix), fisheye_img_vis)
                cv2.imwrite('{}/img{:04d}_pred_depth_{}.png'.format(result_dir, img_id, suffix), pred_depth_rgb)
                cv2.imwrite('{}/img{:04d}_pred_normal_{}.png'.format(result_dir, img_id, suffix), (pred_normal[...,::-1]*255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_fisheye_pred_depth_{}.png'.format(result_dir, img_id, suffix), fisheye_pred_depth_rgb)
                cv2.imwrite('{}/img{:04d}_mix_pred_semantic_{}.png'.format(result_dir, img_id, suffix), mix_pred_semantic.astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_mix_pred_panoptic_{}.png'.format(result_dir, img_id, suffix), mix_pred_panoptic_rgb.astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_fisheye_pred_panoptic_{}.png'.format(result_dir, img_id, suffix), (fisheye_pred_panoptic_rgb.reshape(350, 350, 3)[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_mix_fisheye_pred_semantic_panoptic_{}.png'.format(result_dir, img_id, suffix), mix_fisheye)
                np.save('{}/img{:04d}_pred_depth_{}.npy'.format(result_dir, img_id, suffix), pred_depth)
                np.save('{}/img{:04d}_fisheye_pred_depth_{}.npy'.format(result_dir, img_id, suffix), fisheye_pred_depth)
                np.save('{}/img{:04d}_pred_semantic_{}.npy'.format(result_dir, img_id, suffix), pred_semantic.reshape(h, w, -1))
                np.save('{}/img{:04d}_fisheye_pred_semantic_{}.npy'.format(result_dir, img_id, suffix), fisheye_pred_semantic.reshape(350, 350, -1))
                np.save('{}/img{:04d}_id2semantic_{}.npy'.format(result_dir, img_id, suffix), id2semantic)
                np.save('{}/img{:04d}_panoptic_id_map_{}.npy'.format(result_dir, img_id, suffix), panoptic_id_map)
                np.save('{}/img{:04d}_fisheye_id2semantic_{}.npy'.format(result_dir, img_id, suffix), fisheye_id2semantic)
                np.save('{}/img{:04d}_fisheye_panoptic_id_map_{}.npy'.format(result_dir, img_id, suffix), fisheye_panoptic_id_map)
            else:              
                np.save('{}/img{:04d}_pred_semantic_{}.npy'.format(result_dir, img_id, suffix), pred_semantic.reshape(h, w, -1))
                np.save('{}/img{:04d}_instance_temp_{}.npy'.format(result_dir, img_id, suffix), instance_temp_ft)
                np.save('{}/img{:04d}_fisheye_pred_semantic_{}.npy'.format(result_dir, img_id, suffix), fisheye_pred_semantic.reshape(350, 350, -1))
                np.save('{}/img{:04d}_fisheye_instance_temp_{}.npy'.format(result_dir, img_id, suffix), fisheye_instance_temp_ft)
            