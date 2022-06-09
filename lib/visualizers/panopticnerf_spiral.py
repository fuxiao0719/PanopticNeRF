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

class surfacenet(nn.Module):
    def __init__(self, device):
        super(surfacenet, self).__init__()
        self.convDelYDelZ = nn.Conv2d(1, 1, 3)
        self.convDelXDelZ = nn.Conv2d(1, 1, 3)
        self.device = device

    def forward(self, x):
        nb_channels = 1
        h, w = x.shape[-2:]
        delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                        [-1.00000, 0.00000, 1.00000],
                                        [0.00000, 0.00000, 0.00000]])
        delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1).to(self.device)
        delzdelx = F.conv2d(x, delzdelxkernel, padding = 'same')
        delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                        [0.00000, 0.00000, 0.00000],
                                        [0.0000, 1.00000, 0.00000]])
        delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1).to(self.device)
        delzdely = F.conv2d(x, delzdelykernel, padding = 'same')
        delzdelz = torch.ones(delzdely.shape, dtype=torch.float64).to(self.device)
        surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
        surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2)[:,:,None,:,:])
        return surface_norm

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
            gt_root = cfg.gt_path
            gt_path = os.path.join(gt_root, '{:010}.png'.format(batch['meta']['tar_idx'].item()))
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            gt_img = batch['rays_rgb'][b].reshape(h, w, 3).detach().cpu().numpy()
            if len(cfg.cascade_samples) > 1:
                pred_img = torch.clamp(output['rgb_1'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
                pred_depth = output['depth_1'][b].reshape(h, w).detach().cpu().numpy()
            else:
                pred_img = torch.clamp(output['rgb_0'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()
                pred_depth = output['depth_0'][b].reshape(h, w).detach().cpu().numpy()
            img_id = int(batch["meta"]["tar_idx"].item())
            result_dir = cfg.result_dir
            result_dir = os.path.join(result_dir, batch['meta']['sequence'][0])
            print(result_dir)

            os.system("mkdir -p {}".format(result_dir))
            # np.save('{}/img{:04d}_depth.npy'.format(result_dir, idx),(pred_depth))
            if 'semantic_point_0' in output:
                pred_semantic_point = output['semantic_point_0']
            
            _, pred_semantic = output['semantic_map_0'][b].max(1)
            if batch['stereo_num'] == 0:
                store_dir = os.path.join(cfg.result_dir, 'image_00')
            else:
                store_dir = os.path.join(cfg.result_dir, 'image_01')
            os.system("mkdir -p {}".format(store_dir))

            if cfg.render_instance:
                instance_map_pre =  output['instance_map_0'][b]                                             # output['instance_map_0']
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
                cv2.imwrite('{}/img{:04d}_pred_instance.png'.format(result_dir, batch["idx"].item()), (instance_temp_rgb.reshape(h, w, 3).astype(np.uint8)))
                mix_instance = cv2.addWeighted((pred_img[..., [2, 1, 0]] * 255).astype(np.uint8), 0.5, (instance_temp_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8),0.5, 0)
                cv2.imwrite('{}/img{:04d}_mix_pred_instance.png'.format(result_dir, batch["idx"].item()),(mix_instance * 255).astype(np.uint8))

            pred_semantic = pred_semantic.reshape(h, w).detach().cpu().numpy()
            color = assigncolor(pred_semantic.reshape(-1), 'semantic')
            cv2.imwrite('{}/img{:04d}_pred_semantic.png'.format(result_dir, batch["idx"].item()), (color.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8))
            mix_semantic = cv2.addWeighted((pred_img[..., [2, 1, 0]] * 255).astype(np.uint8), 0.5, (color.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8), 0.5, 0)
            cv2.imwrite('{}/img{:04d}_mix.png'.format(result_dir, batch["idx"].item()), mix_semantic.astype(np.uint8))

            eval_instance_list = [11, 26, 30, 29]
            if cfg.render_instance == True:
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
                cv2.imwrite('{}/img{:04d}_pred_panoptic.png'.format(result_dir, batch["idx"].item()), (panoptic_rgb.reshape(h, w, 3)[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite('{}/img{:04d}_mix_pred_panoptic.png'.format(result_dir, batch["idx"].item()), (mix_panoptic_rgb).astype(np.uint8))
