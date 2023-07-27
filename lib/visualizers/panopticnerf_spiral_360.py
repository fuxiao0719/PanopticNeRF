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
            render_boundary_mode = False
            if cfg.result_dir[-2:] == 'ft':
                render_boundary_mode = True
            result_dir = os.path.join(cfg.result_dir, batch['meta']['sequence'][0])
            suffix = '360'
            os.system("mkdir -p {}".format(result_dir))

            # instance finetuning
            instance_finetuning = False
            if (render_boundary_mode == False) and (cfg.use_post_processing == False) and (cfg.merge_instance == True):
                result_dir_ft = os.path.join(cfg.result_dir+'_ft', batch['meta']['sequence'][0])
                instance_finetuning = True

            # rgb
            pred_img = torch.clamp(output['rgb_0'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()

            # depth
            pred_depth = output['depth_0'][b].reshape(h, w).detach().cpu().numpy()
            pred_depth_rgb = copy.deepcopy(pred_depth)
            pred_depth_rgb[pred_depth_rgb == 0] = 100000
            pred_depth_rgb = color_depth_map(pred_depth_rgb, 100)

            # semantic
            _, pred_semantic = output['semantic_map_0'][b].max(1)
            _, fix_semantic = output['fix_semantic_map_0'][b].max(1)
            mask_fix = (output['semantic_bbox_gt'].reshape(1, -1, cfg.samples_all, 50).sum(dim=-1).sum(dim=-1) == cfg.samples_all).reshape(-1)
            pred_semantic[mask_fix] = fix_semantic[mask_fix]

            if instance_finetuning == True:
                pred_semantic_w_boundary = torch.from_numpy(np.load('{}/img{:04d}_pred_semantic_{}.npy'.format(result_dir_ft, img_id, suffix))).cuda()
                mask_build_semantic = (pred_semantic_w_boundary == 11) & (pred_semantic == 11)
            pred_semantic_rgb = assigncolor(pred_semantic.reshape(h, w).detach().cpu().numpy().reshape(-1), 'semantic')
            mix_pred_semantic = cv2.addWeighted((pred_img[..., ::-1] * 255).astype(np.uint8), 0.5, (pred_semantic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            
            # instance
            eval_instance_list = [11, 26, 29, 30]
            instance_map_pre =  output['instance_map_0'][b]              
            instance_map_post = torch.zeros_like(instance_map_pre)
            for (semantic_id, id_list) in batch['semantic2id'].items():                                
                semantic_mask = (pred_semantic==semantic_id)
                for id_ in id_list:
                    instance_map_post[semantic_mask, id_] = instance_map_pre[semantic_mask, id_]
            _, instance_temp = instance_map_post.max(1)

            if instance_finetuning == True:
                instance_temp_w_boundary = torch.from_numpy(np.load('{}/img{:04d}_instance_temp_{}.npy'.format(result_dir_ft, img_id, suffix))).to(instance_temp)
                instance_temp[mask_build_semantic] = instance_temp_w_boundary[mask_build_semantic]
            else:
                instance_temp_save = copy.deepcopy(instance_temp.detach().cpu().numpy())

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
            panoptic_rgb = np.zeros_like(color)
            for id_ in eval_instance_list:
                panoptic_mask = panoptic_mask | (pred_semantic == id_)
            panoptic_mask = ~panoptic_mask
            panoptic_rgb[panoptic_mask] = color[panoptic_mask]
            instance_rgb = instance_temp_rgb.reshape(h, w, 3)[..., ::-1] * 255
            panoptic_rgb[~panoptic_mask] = instance_rgb[~panoptic_mask]

            mix_panoptic_rgb = cv2.addWeighted((pred_img[..., ::-1] * 255).astype(np.uint8), 0.5, (panoptic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            panoptic_id_map = np.zeros((h, w))
            panoptic_id_map[panoptic_mask] = pred_semantic[panoptic_mask]
            panoptic_id_map[~panoptic_mask] = pred_instance[~panoptic_mask]

            # save
            if render_boundary_mode == False:
                cv2.imwrite('{}/img{:04d}_pred_img_{}.png'.format(result_dir, img_id, suffix), (pred_img[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_pred_depth_{}.png'.format(result_dir, img_id, suffix), pred_depth_rgb)
                cv2.imwrite('{}/img{:04d}_mix_pred_semantic_{}.png'.format(result_dir, img_id, suffix), mix_pred_semantic.astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_pred_semantic_{}.png'.format(result_dir, img_id, suffix), (pred_semantic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_pred_panoptic_{}.png'.format(result_dir, img_id, suffix), (panoptic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_mix_pred_panoptic_{}.png'.format(result_dir, img_id, suffix), mix_panoptic_rgb.astype(np.uint8))
            else:
                np.save('{}/img{:04d}_pred_semantic_{}.npy'.format(result_dir, img_id, suffix), pred_semantic.reshape(-1))
                np.save('{}/img{:04d}_instance_temp_{}.npy'.format(result_dir, img_id, suffix), instance_temp_save)