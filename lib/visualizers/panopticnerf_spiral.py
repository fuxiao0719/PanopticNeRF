import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from lib.config import cfg
import os
from tools.kitti360scripts.helpers.labels import id2label, labels
import torch.nn as nn
from torch.functional import norm

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

class Visualizer:
    def __init__(self, ):
        self.color_crit = lambda x, y: ((x - y)**2).mean()
        self.mse2psnr = lambda x: -10. * np.log(x) / np.log(torch.tensor([10.]))
        self.psnr = []

    def visualize(self, output, batch):
        b = len(batch['rays'])
        for b in range(b):
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            gt_img = batch['rays_rgb'][b].reshape(h, w, 3).detach().cpu().numpy()
            pred_img = torch.clamp(output['rgb_0'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
            pred_depth = output['depth_0'][b].reshape(h, w).detach().cpu().numpy()
            img_id = int(batch["meta"]["tar_idx"].item())
            result_dir = cfg.result_dir
            result_dir = os.path.join(result_dir, batch['meta']['sequence'][0])

            print(result_dir)

            os.system("mkdir -p {}".format(result_dir))
            np.save('{}/img{:04d}_{:04d}_depth.npy'.format(result_dir, cfg.spiral_frame, img_id),(pred_depth))
            pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((pred_depth/pred_depth.max()) * 255).astype(np.uint8),alpha=2), cv2.COLORMAP_JET)
            cv2.imwrite('{}/img{:04d}_{:04d}_depth.png'.format(result_dir, cfg.spiral_frame, img_id), pred_depth)
            _, pred_semantic = output['semantic_map_0'][b].max(1)
            np.save('{}/img{:04d}_{:04d}_pred_semantic.npy'.format(result_dir, cfg.spiral_frame, img_id), pred_semantic.reshape(h, w, -1).detach().cpu().numpy())

            instance_map_pre =  output['instance_map_0'][b]                                    
            instance_map_post = torch.zeros_like(instance_map_pre)
            for (semantic_id, id_list) in batch['semantic2id'].items():                                
                semantic_mask = (pred_semantic==semantic_id)
                for id_ in id_list:
                    instance_map_post[semantic_mask, id_] = instance_map_pre[semantic_mask, id_]
            _, instance_temp = instance_map_post.max(1)
            pred_instance = np.copy(instance_temp.cpu().numpy().reshape(h,w))
            pred_instance += 50
            instance_temp = instance_temp.detach().cpu().numpy().reshape(h,w)
            instance_list = np.unique(instance_temp)
            for inst in instance_list:
                instance_temp[instance_temp == inst] = batch['instance2id'][inst].detach().cpu().numpy()  
            instance2semantic = batch['instance2semantic']                                                  
            instance2semantic = {(k + 50): v.item() for (k, v) in instance2semantic.items()}
            instance_temp_rgb = np.zeros((h, w, 3))
            instance_temp = instance_temp / (500000) * 256**3
            instance_temp_rgb[..., 0] = instance_temp % 256
            instance_temp_rgb[..., 1] = instance_temp / 256 % 256
            instance_temp_rgb[..., 2] = instance_temp / (256 * 256)

            pred_semantic = pred_semantic.reshape(h, w).detach().cpu().numpy()
            color = assigncolor(pred_semantic.reshape(-1), 'semantic')
            cv2.imwrite('{}/img{:04d}_{:04d}_pred_semantic.png'.format(result_dir, cfg.spiral_frame, img_id), (color.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8))
            mix_semantic = cv2.addWeighted((pred_img[..., [2, 1, 0]] * 255).astype(np.uint8), 0.5, (color.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            cv2.imwrite('{}/img{:04d}_{:04d}_mix_semantic.png'.format(result_dir, cfg.spiral_frame, img_id), mix_semantic.astype(np.uint8))

            eval_instance_list = [11, 26, 30, 29]
            panoptic_mask = np.ones_like(pred_semantic) > 1
            color = color.reshape(h, w, 3)
            panoptic_id_map = np.zeros((h, w))
            panoptic_rgb = np.zeros_like(color)
            for id_ in eval_instance_list:
                panoptic_mask = panoptic_mask | (pred_semantic == id_)
            panoptic_mask = ~panoptic_mask
            panoptic_rgb[panoptic_mask] = color[panoptic_mask]
            instance_color = instance_temp_rgb.reshape(h, w, 3)[..., ::-1] * 255
            panoptic_rgb[~panoptic_mask] = instance_color[~panoptic_mask]
            mix_panoptic_rgb = cv2.addWeighted((pred_img[..., [2, 1, 0]] * 255).astype(np.uint8), 0.5, (panoptic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            cv2.imwrite('{}/img{:04d}_{:04d}_pred_panoptic.png'.format(result_dir, cfg.spiral_frame, img_id), (panoptic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8))
            cv2.imwrite('{}/img{:04d}_{:04d}_mix_panoptic.png'.format(result_dir, cfg.spiral_frame, img_id), (mix_panoptic_rgb).astype(np.uint8))
            np.save('{}/img{:04d}_{:04d}_instance2semantic.npy'.format(result_dir, cfg.spiral_frame, img_id), instance2semantic)
            panoptic_id_map[panoptic_mask] = pred_semantic[panoptic_mask]
            panoptic_id_map[~panoptic_mask] = pred_instance[~panoptic_mask]
            np.save('{}/img{:04d}_{:04d}_panoptic_id_map.npy'.format(result_dir, cfg.spiral_frame, img_id), panoptic_id_map)