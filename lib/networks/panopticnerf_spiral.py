from __future__ import annotations
import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .utils import raw2outputs_label, sample_along_ray, sample_pdf, raw2weights, perturb_samples
from lib.config import cfg
from .mlp_weak_spiral import NeRF
import math
merge_list_car = [27, 28, 29, 30, 31]
merge_list_box = [39]
merge_list_park = [9]
merge_list_gate = [35]
    
class Network(nn.Module):
    def __init__(self, down_ratio=2):
        super(Network, self).__init__()
        self.cascade = len(cfg.cascade_samples) > 1
        self.nerf_0 = NeRF(fr_pos=cfg.fr_pos)

    def render_rays(self, batch, rays, cam_idx, intersection, is_fisheye):
        B, N_rays, N_bbox, _ = intersection.shape
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        scale_factor = torch.norm(rays_d, p=2, dim=2)
        near_depth, far_depth = intersection[..., 0].to(rays), intersection[..., 1].to(rays)
        mask_depth = far_depth > cfg.dist
        near_depth[mask_depth] = -1.
        far_depth[mask_depth] = -1.
        z_vals = sample_along_ray(near_depth, far_depth, cfg.bbox_sp)
        z_vals_bound = torch.cat([z_vals[...,0]-1e-5,z_vals[...,-1]+1e-5],dim=2)
        PDF = torch.ones(z_vals.shape)
        PDF = PDF.reshape(B, N_rays, -1)
        z_vals = z_vals.reshape(z_vals.shape[0], z_vals.shape[1], -1)
        z_vals, sort_index = torch.sort(z_vals,2)
        PDF = torch.take(PDF.to(z_vals), sort_index)
        z_vals = sample_pdf(z_vals, PDF[...,1:], cfg.cascade_samples[0], det=True)
        z_vals = torch.cat([z_vals, z_vals_bound],dim=2)
        mask_bg_points = z_vals < 0.
        z_vals[mask_bg_points] = cfg.dist + 20 * torch.rand(mask_bg_points.sum()).to(rays)
        z_vals, _ = torch.sort(z_vals,2)
        xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None] / scale_factor[...,None,None]

        xyz /= torch.tensor(cfg.scale_factor)[None, None, None].to(xyz)
        ray_dir = rays[..., 3:6]
        ray_dir = ray_dir[:, :, None].repeat(1, 1, cfg.samples_all, 1)

        frame_idx = cfg.train_frames*2 + (cfg.intersection_spiral_frame - cfg.start)
        raw, nablas = self.nerf_0(xyz, z_vals/scale_factor[...,None], ray_dir, frame_idx, cam_idx)
        
        with torch.no_grad():
            B, N_rays, N_samples, _ = xyz.shape
            semantic_idx = (z_vals[...,None] > intersection[:,:,None][:,:,:,:,0]) & (z_vals[...,None] < intersection[:,:,None][:,:,:,:,1])
            idx0_in, idx1_in, idx2_in, idx3_in = torch.where(semantic_idx==True)
            mask_bound = ((z_vals[...,None] - intersection[:,:,None][...,1]< 1e-3) & (z_vals[...,None] - intersection[:,:,None][...,1] > 0)) \
                | ((intersection[:,:,None][...,0] - z_vals[...,None]> 0) & (intersection[:,:,None][...,0] - z_vals[...,None] < 1e-3))
            idx0_bound, idx1_bound, idx2_bound, _ = torch.where(mask_bound==True)
            intersection_temp = intersection[:,:,None].repeat(1,1,cfg.samples_all,1,1)
            one_hot_all = torch.zeros(semantic_idx.shape[0],semantic_idx.shape[1],semantic_idx.shape[2], 50)
            one_hot_all[idx0_in, idx1_in, idx2_in, intersection_temp[idx0_in, idx1_in, idx2_in, idx3_in, 3].long()] = True
            mask_bbox = (z_vals<cfg.dist) & (one_hot_all.to(z_vals).sum(3)==0.)
            one_hot_all[...,0][mask_bbox] = True
            mask_bg = (z_vals>cfg.dist) & (one_hot_all.to(z_vals).sum(3)==0.)
            one_hot_all[...,23][mask_bg] = True
            one_hot_all = one_hot_all.reshape(B, -1, 50)
        
        for i in merge_list_box:
            mask_merge =  (one_hot_all[..., i] == True)
            one_hot_all[:,mask_merge[0], i] = False
            one_hot_all[:,mask_merge[0], 41] = True
        if self.training == True:
            for i in merge_list_car:
                mask_merge =  (one_hot_all[..., i] == True)
                one_hot_all[:,mask_merge[0], i] = False
                one_hot_all[:,mask_merge[0], 26] = True
            for i in merge_list_park:
                mask_merge =  (one_hot_all[..., i] == True)
                one_hot_all[:,mask_merge[0], i] = False
                one_hot_all[:,mask_merge[0], 8] = True
            for i in merge_list_gate:
                mask_merge =  (one_hot_all[..., i] == True)
                one_hot_all[:,mask_merge[0], i] = False
                one_hot_all[:,mask_merge[0], 13] = True
        
        B, N_rays, N_samples, _ = xyz.shape
        # filter points that not in bbox
        raw[...,3][mask_bbox] = 0
        raw[idx0_bound, idx1_bound, idx2_bound, 3] = 0
        # model the sky with one sample point
        mask_bg_density = torch.cat((mask_bg[...,:-5], torch.zeros_like(mask_bg[...,-5:]).to(mask_bg)), dim=-1)
        raw[...,3][mask_bg_density] = 0
        outputs = {}
        outputs['points_semantic_0'] = raw[..., 4:]
        fix_label = one_hot_all.reshape(B, N_rays, N_samples, 50)
        ret_0 = raw2outputs_label(raw, z_vals/scale_factor[...,None], rays_d, fix_label, cfg.white_bkgd, one_hot_all_instance=None, is_test=False)
        if self.training == False or cfg.use_instance_loss == True:
            with torch.no_grad():
                B, N_rays, N_samples, _ = xyz.shape
                semantic_idx = (z_vals[...,None] > intersection[:,:,None][:,:,:,:,0]) & (z_vals[...,None] < intersection[:,:,None][:,:,:,:,1])
                idx0_in, idx1_in, idx2_in, idx3_in = torch.where(semantic_idx==True)
                semantic_mask = (semantic_idx==True)
                mask_bound = ((z_vals[...,None] - intersection[:,:,None][...,1]< 1e-3) & (z_vals[...,None] - intersection[:,:,None][...,1] > 0)) \
                    | ((intersection[:,:,None][...,0] - z_vals[...,None]> 0) & (intersection[:,:,None][...,0] - z_vals[...,None] < 1e-3))
                idx0_bound, idx1_bound, idx2_bound, _ = torch.where(mask_bound==True)
                intersection_temp = intersection[:,:,None].repeat(1,1,cfg.samples_all,1,1)
                id2instance = batch['id2instance']
                instance2id = batch['instance2id']
                one_hot_all_instance = torch.zeros(semantic_idx.shape[0],semantic_idx.shape[1],semantic_idx.shape[2], len(id2instance) + 1)
                for (inst, id) in instance2id.items():
                    mask_inst = (intersection_temp[..., 2] == id)
                    idx0_m, idx1_m, idx2_m, idx3_m = torch.where((semantic_mask & mask_inst)==True)
                    temp_id = torch.ones_like(intersection_temp[idx0_m, idx1_m, idx2_m, idx3_m, 2]) * inst
                    intersection_temp[idx0_m, idx1_m, idx2_m, idx3_m, 2] = temp_id
                one_hot_all_instance[idx0_in, idx1_in, idx2_in, intersection_temp[idx0_in, idx1_in, idx2_in, idx3_in, 2].long()] = True
                one_hot_all_instance[mask_bbox] = False
                one_hot_all_instance[...,-1][mask_bg] = True
            semantic = raw[...,4:].clone()
            semantic_gt = one_hot_all.clone()
            inf = torch.empty_like(semantic_gt).fill_(-float('inf'))
            semantic_gt = torch.where(semantic_gt == 0., inf, semantic_gt)
            m = nn.Softmax(dim=2)
            semantic_gt = m(semantic_gt).to(semantic)
            semantic_gt[torch.isnan(semantic_gt)] = 0.
            # merge car
            for i in merge_list_car:
                semantic[..., i] = (semantic[..., 26].reshape(1, -1, 1) * semantic_gt[..., i][...,None]).reshape(1, N_rays, N_samples) * 4
            semantic[...,26] = (semantic[..., 26].reshape(1, -1, 1) * semantic_gt[..., 26][...,None]).reshape(1, N_rays, N_samples)
            # merge park
            for i in merge_list_park:
                semantic[..., i] = (semantic[..., 8].reshape(1, -1, 1) * semantic_gt[..., i][...,None]).reshape(1, N_rays, N_samples) * 2
            semantic[..., 8] = (semantic[..., 8].reshape(1, -1, 1) * semantic_gt[..., 8][...,None]).reshape(1, N_rays, N_samples)
            # merge gate
            for i in merge_list_gate:
                semantic[..., i] = (semantic[..., 13].reshape(1, -1, 1) * semantic_gt[..., i][...,None]).reshape(1, N_rays, N_samples) * 2
            semantic[..., 13] = (semantic[..., 13].reshape(1, -1, 1) * semantic_gt[..., 13][...,None]).reshape(1, N_rays, N_samples)
            # merge box
            for i in merge_list_box:
                semantic[..., 41] = semantic[..., 41] + semantic[..., i]
                semantic[..., i] = 0
            raw[...,4:] = semantic
            outputs['one_hot_all_instance'] = one_hot_all_instance
            ret_0 = raw2outputs_label(raw, z_vals/scale_factor[...,None], rays_d, fix_label, cfg.white_bkgd, one_hot_all_instance=one_hot_all_instance, is_test = True)
        
        for key in ret_0:
            outputs[key + '_0'] = ret_0[key]
        outputs['semantic_bbox_gt'] = one_hot_all 
        outputs['semantic_filter'] = intersection
        return outputs
    
    def batchify_rays(self, batch, rays, cam_idx, intersection):
        all_ret = {}
        chunk = cfg.chunk_size
        for i in range(0, rays.shape[1], chunk):

            ret = self.render_rays(batch, rays[:,i:i+chunk], cam_idx[:,i:i+chunk], intersection[:,i:i+chunk], is_fisheye=False)
            if self.training == False:
                del ret['points_semantic_0'] , ret['weights_0'], ret['z_vals_0'], ret['semantic_point_0'] , ret['semantic_filter'], ret['one_hot_all_instance']
            
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
            torch.cuda.empty_cache()
        
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret

    def forward(self, batch):
        rays, cam_idx, intersection = batch['rays'], batch['cam_idx'], batch['intersection']
        ret = self.batchify_rays(batch, rays, cam_idx, intersection)
        return ret
