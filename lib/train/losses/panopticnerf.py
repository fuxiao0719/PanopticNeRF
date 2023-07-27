from operator import imod
import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg
from torch.nn import functional as F
import math

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.depth_crit = nn.HuberLoss(reduction='mean')
        self.weights_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.N_rays = cfg.N_rays
        self.N_fe_rays = cfg.N_fe_rays
        self.N_points = cfg.samples_all
        self.channel = 50
        self.epsilon = 5e-3
        self.bg_filter = torch.zeros(cfg.samples_all, 50)
        self.bg_filter[...,23] = 1.
        self.far_cls_filter = torch.zeros(cfg.samples_all, 50)
        self.far_cls_filter[...,0] = 1.
        self.device = 'cuda:'+str(cfg.local_rank)
        self.end_step = cfg.train.epoch * cfg.ep_iter
    
    def get_semantic_loss(self, pseudo_label, semantic_map, semantic_filter, fix_semantic_map, semantic_bbox_gt, points_semantic, weights, stereo_num, mask_bg_far_cls, is_fisheye):
        semantic_loss = 0.
        merge_list_car = [27, 28, 29, 30, 31]
        merge_list_box = [39]
        merge_list_park = [9]
        merge_list_gate = [35]
        
        # merge and filter 2d pseudo semantic
        pseudo_label[mask_bg_far_cls] = 0
        for i in merge_list_car:
            pseudo_label[pseudo_label == i] = 26
        for i in merge_list_box:
            pseudo_label[pseudo_label == i] = 41
        for i in merge_list_park:
            pseudo_label[pseudo_label == i] = 8
        for i in merge_list_gate:
            pseudo_label[pseudo_label == i] = 13
        if cfg.use_pseudo_filter == True:
            semantic_filter = semantic_filter[..., 3]
            for i in merge_list_car:
                semantic_filter[semantic_filter == i] = 26.
            for i in merge_list_box:
                semantic_filter[semantic_filter == i] = 41.
            for i in merge_list_park:
                semantic_filter[semantic_filter == i] = 8.
            for i in merge_list_gate:
                semantic_filter[semantic_filter == i] = 13.
            pseudo_label_temp = pseudo_label[..., None].repeat(1,1,semantic_filter.shape[-1])
            mask_fg_filter, _ = (semantic_filter == pseudo_label_temp).max(-1)
            mask_fg_filter = mask_fg_filter[0]
            mask_pseudo_filter = torch.zeros_like(pseudo_label.reshape(-1), dtype=torch.bool).to(self.device)
            # filter erroneous fisheye pseudo label
            if is_fisheye == True:
                for pseudo_filter_label in cfg.pseudo_filter_labels:
                    mask_pseudo_filter = mask_pseudo_filter | (pseudo_label == pseudo_filter_label).reshape(-1)
                mask_fg_filter = mask_fg_filter & (~mask_pseudo_filter)
            mask_bg_filter = ((pseudo_label == 23) | (pseudo_label == 0)).reshape(-1)
            mask_filter = (mask_fg_filter | mask_bg_filter)
        else:
            mask_filter = torch.ones_like(pseudo_label.reshape(-1).long()).to(self.device) > 0
        
        cross_entropy = nn.CrossEntropyLoss()
        # 2d pred
        tau = 1
        prior_class = torch.zeros(50).to(self.device)
        if is_fisheye == False:
            prior_class[cfg.prior_class_ids] = torch.Tensor(cfg.prior_class).to(self.device)
            N_rays = self.N_rays
        else:
            prior_class[cfg.fe_prior_class_ids] = torch.Tensor(cfg.fe_prior_class).to(self.device)
            N_rays = self.N_fe_rays
        if mask_bg_far_cls.sum() > 0:
            prior_class[0] = (mask_bg_far_cls.sum()/N_rays).to(self.device) # far cls
            prior_class /= prior_class.sum()
        prior_class[prior_class>0] = torch.log(prior_class[prior_class>0])
        if torch.sum(mask_filter) != 0:
            semantic_loss_2d_pred = cross_entropy((semantic_map+tau*prior_class).reshape(-1 ,self.channel)[mask_filter], pseudo_label.reshape(-1).long()[mask_filter])
        else:
            semantic_loss_2d_pred = torch.tensor(0.).to(self.device)
        semantic_loss_2d_pred = cfg.lambda_2d_semantic * semantic_loss_2d_pred
        semantic_loss += semantic_loss_2d_pred
        
        # 2d fix
        semantic_loss_2d_fix = cross_entropy(fix_semantic_map[~(mask_bg_far_cls|mask_pseudo_filter)].reshape(-1 ,self.channel), pseudo_label[~(mask_bg_far_cls|mask_pseudo_filter)].reshape(-1).long())
        semantic_loss_2d_fix = cfg.lambda_fix_semantic * semantic_loss_2d_fix
        semantic_loss += semantic_loss_2d_fix
        
        # 3d primitive
        semantic_bbox_gt = semantic_bbox_gt.reshape(1, N_rays, self.N_points, self.channel)[~mask_bg_far_cls].reshape(1, -1, self.channel)
        semantic_gt = semantic_bbox_gt
        idx0_bg, idx1_bg, idx2_bg = torch.where(semantic_gt==-1.)
        inf = torch.empty_like(semantic_gt).fill_(-float('inf'))
        semantic_gt = torch.where(semantic_gt == 0., inf, semantic_gt)
        m = nn.Softmax(dim=2)
        semantic_gt = m(semantic_gt).to(self.device)
        semantic_gt[idx0_bg, idx1_bg, idx2_bg] = 0.
        msk_max, _ = semantic_gt.reshape(-1 , self.channel).max(1)
        msk = (msk_max >= 0.99999) & (weights[~mask_bg_far_cls].reshape(-1) > cfg.weight_th)
        if torch.sum(msk).item() != 0:
            semantic_loss_3d = cross_entropy(points_semantic[~mask_bg_far_cls].reshape(-1 ,self.channel)[msk, :], semantic_gt.reshape(-1 ,self.channel)[msk, :])
        else:
            semantic_loss_3d = torch.tensor(0.).to(self.device)
        semantic_loss_3d = cfg.lambda_3d_semantic * semantic_loss_3d
        semantic_loss += semantic_loss_3d
        
        if (is_fisheye==False) and (cfg.use_pspnet == True) and (stereo_num == 1):
            semantic_loss = torch.tensor(0.).to(self.device)
            semantic_loss_3d = torch.tensor(0.).to(self.device)
            semantic_loss_2d_pred = torch.tensor(0.).to(self.device)
            semantic_loss_2d_fix = torch.tensor(0.).to(self.device)
        return semantic_loss, semantic_loss_3d, semantic_loss_2d_pred, semantic_loss_2d_fix 
    
    def forward(self, batch):
        output = self.net(batch)
        scalar_stats = {}
        loss = 0
        depth_object = cfg.depth_object

        # instance loss
        if cfg.use_instance_loss == True:
            cross_entropy = nn.CrossEntropyLoss()
            linear_regression_pseudo_label = batch['linear_regression_pseudo_label']
            pseudo_instance_label = linear_regression_pseudo_label[...,1]
            pseudo_semantic_label = linear_regression_pseudo_label[...,0]
            pred_semantic_label = output['semantic_map_0'].max(-1)[1]
            mask_build = (pseudo_semantic_label == 11) & (pred_semantic_label == 11)
            fix_instance_loss = cross_entropy(output['instance_map_0'][mask_build], pseudo_instance_label[mask_build])
            loss += cfg.lambda_fix_instance * fix_instance_loss
            scalar_stats.update({'fix_instance_loss_2d': fix_instance_loss})

        # rgb loss
        if 'rgb_0' in output.keys():
            mask_bg_far_cls = ((output['semantic_bbox_gt'].reshape(1, self.N_rays, -1) == self.bg_filter.reshape(-1)).sum(dim=-1) == (self.N_points * self.channel)).to(self.device) & (batch['pseudo_label'] != 23)
            color_loss = cfg.train.weight_color * self.color_crit(batch['rays_rgb'], output['rgb_0'])
            scalar_stats.update({'color_mse_0': color_loss})
            loss += color_loss
            psnr = -10. * torch.log(color_loss.detach()) / torch.log(torch.Tensor([10.]).to(self.device))
            scalar_stats.update({'psnr_0': psnr})
        
        # fisheye rgb loss
        if 'fisheye_rgb_0' in output.keys():
            fe_mask_bg_far_cls = ((output['fisheye_semantic_bbox_gt'].reshape(1, self.N_fe_rays, -1) == self.bg_filter.reshape(-1)).sum(dim=-1) == (self.N_points * self.channel)).to(self.device) & (batch['fisheye_pseudo_label'] != 23)
            color_fe_loss = self.color_crit(batch['fisheye_rays_rgb'], output['fisheye_rgb_0'])
            fisheye_color_loss = cfg.train.weight_color * color_fe_loss
            scalar_stats.update({'fisheye_color_mse_0': fisheye_color_loss})
            loss += fisheye_color_loss
            fisheye_psnr = -10. * torch.log(fisheye_color_loss.detach()) / torch.log(torch.Tensor([10.]).to(self.device))
            scalar_stats.update({'fisheye_psnr_0': fisheye_psnr})
        
        # weak depth loss
        if ('depth_0' in output.keys()) and ('depth' in batch) and cfg.use_depth == True:
            pred_depth = output['depth_0']
            gt_depth = batch['depth']
            semantic_filter = output['semantic_filter']
            semantic_filter = semantic_filter[..., 3]
            mask_filter_depth = torch.zeros_like(gt_depth).to(semantic_filter) > 1
            for id in depth_object:
                mask_filter, _ = (semantic_filter == id).max(-1)
                mask_filter_depth = mask_filter_depth | mask_filter
            mask = (gt_depth>0) & (gt_depth<100) & mask_filter_depth
            if torch.sum(mask) < 0.5:
                depth_loss = torch.tensor(0.).to(self.device)
            else:
                depth_loss = self.depth_crit(gt_depth[mask], pred_depth[mask])
                depth_loss = depth_loss.clamp(max=0.1)
            scalar_stats.update({'depth_loss': depth_loss})
            loss += cfg.lambda_depth * depth_loss

        # semantic_loss
        semantic_loss, semantic_loss_3d, semantic_loss_2d_pred, semantic_loss_2d_fix = self.get_semantic_loss( 
            batch['pseudo_label'], 
            output['semantic_map_0'], 
            output['semantic_filter'], 
            output['fix_semantic_map_0'], 
            output['semantic_bbox_gt'],
            output['points_semantic_0'], 
            output['weights_0'], 
            batch['stereo_num'], 
            mask_bg_far_cls,
            False 
        )
        
        # fisheye semantic_loss
        fisheye_semantic_loss, fisheye_semantic_loss_3d, fisheye_semantic_loss_2d_pred, fisheye_semantic_loss_2d_fix = self.get_semantic_loss(
            batch['fisheye_pseudo_label'], 
            output['fisheye_semantic_map_0'], 
            output['fisheye_semantic_filter'], 
            output['fisheye_fix_semantic_map_0'], 
            output['fisheye_semantic_bbox_gt'],
            output['fisheye_points_semantic_0'], 
            output['fisheye_weights_0'], 
            batch['stereo_num'], 
            fe_mask_bg_far_cls,
            True
        )
        
        loss += cfg.lambda_learn_semantic * (semantic_loss + fisheye_semantic_loss)
    
        # scalar stats
        if semantic_loss > 0.:
            scalar_stats.update({'semantic_loss_2d_pred': semantic_loss_2d_pred})
            scalar_stats.update({'semantic_loss_2d_fix': semantic_loss_2d_fix})
            scalar_stats.update({'semantic_loss_3d': semantic_loss_3d})
            scalar_stats.update({'semantic_loss': semantic_loss})
        scalar_stats.update({'fisheye_semantic_loss_2d_pred': fisheye_semantic_loss_2d_pred})
        scalar_stats.update({'fisheye_semantic_loss_2d_fix': fisheye_semantic_loss_2d_fix})
        scalar_stats.update({'fisheye_semantic_loss_3d': fisheye_semantic_loss_3d})
        scalar_stats.update({'fisheye_semantic_loss': fisheye_semantic_loss})
        scalar_stats.update({'loss': loss})
        
        image_stats = {}

        return output, loss, scalar_stats, image_stats
