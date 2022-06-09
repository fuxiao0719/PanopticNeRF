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
        self.epsilon_max = 1.0
        self.epsilon_min = 0.2
        self.decay_speed = 0.00005
    
    def get_gaussian(self, depth_gt, depth_samples):
        return torch.exp(-(depth_gt - depth_samples)**2 / (2*self.epsilon**2))

    def get_weights_gt(self, depth_gt, depth_samples):
        # near
        depth_gt = depth_gt.view(*depth_gt.shape, 1)
        weights = self.get_gaussian(depth_gt, depth_samples).detach()
        # empty and dist
        weights[torch.abs(depth_samples-depth_gt)>self.epsilon]=0
        # normalize
        weights = weights / torch.sum(weights,dim=2,keepdims=True).clamp(min=1e-6)
        return weights.detach()

    def kl_loss(self, weights_gt, weights_es):
        return torch.log(weights_gt * weights_es).sum()

    def forward(self, batch):
        output = self.net(batch)
        scalar_stats = {}
        loss = 0
        merge_list_car = [27, 28, 29, 30, 31]
        merge_list_box = [39]
        merge_list_park = [9]
        merge_list_gate = [35]
        depth_object = cfg.depth_object
        
        # rgb loss
        if 'rgb_0' in output.keys():
            color_loss = cfg.train.weight_color * self.color_crit(batch['rays_rgb'], output['rgb_0'])
            scalar_stats.update({'color_mse_0': color_loss})
            loss += color_loss
            psnr = -10. * torch.log(color_loss.detach()) / \
                    torch.log(torch.Tensor([10.]).to(color_loss.device))
            scalar_stats.update({'psnr_0': psnr})
        
        # depth loss
        if ('depth_0' in output.keys()) and ('depth' in batch) and cfg.use_depth == True:
            device = output['rgb_0'].device
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
                depth_loss = torch.tensor(0.).to(device)
            else:
                depth_loss = self.depth_crit(gt_depth[mask], pred_depth[mask])
                depth_loss = depth_loss.clamp(max=0.1)
            scalar_stats.update({'depth_loss': depth_loss})
            loss += cfg.lambda_depth * depth_loss

        # semantic_loss
        if 'semantic_map_0' in output.keys():
            semantic_loss = 0.
            decay = 1.
            device = output['rgb_0'].device
            pseudo_label = batch['pseudo_label']

            # merge and filter 2d pseudo semantic
            for i in merge_list_car:
                pseudo_label[pseudo_label == i] = 26
            for i in merge_list_box:
                pseudo_label[pseudo_label == i] = 41
            for i in merge_list_park:
                pseudo_label[pseudo_label == i] = 8
            for i in merge_list_gate:
                pseudo_label[pseudo_label == i] = 13
            if cfg.pseudo_filter == True:
                B, N_point, channel = output['semantic_map_0'].shape
                semantic_filter = output['semantic_filter']
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
                mask_filter, _ = (semantic_filter == pseudo_label_temp).max(-1)
                mask_filter = mask_filter[0]
                mask_sky = (pseudo_label == 23)
                mask_filter = (mask_sky | mask_filter).reshape(-1)
            else:
                mask_filter = torch.ones_like(pseudo_label.reshape(-1).long()).to(pseudo_label)>0

            cross_entropy = nn.CrossEntropyLoss()
            nll = nn.NLLLoss()
            # 2d pred
            B, N_point, channel = output['semantic_map_0'].shape
            if torch.sum(mask_filter) != 0:
                semantic_loss_2d_pred = nll(torch.log(output['semantic_map_0'].reshape(-1 ,channel)[mask_filter]+1e-5), pseudo_label.reshape(-1).long()[mask_filter])
            else:
                semantic_loss_2d_pred = torch.tensor(0.).to(device)
            semantic_loss_2d_pred = decay * cfg.lambda_semantic_2d  * semantic_loss_2d_pred
            semantic_loss += semantic_loss_2d_pred
            
            # 2d fix
            semantic_loss_2d_fix = nll(torch.log(output['fix_semantic_map_0'].reshape(-1 ,channel)+1e-5), pseudo_label.reshape(-1).long())
            semantic_loss_2d_fix = cfg.lambda_fix * semantic_loss_2d_fix
            semantic_loss += semantic_loss_2d_fix

            # 3d primitive
            semantic_gt = output['semantic_bbox_gt']
            idx0_bg, idx1_bg, idx2_bg = torch.where(semantic_gt==-1.)
            inf = torch.empty_like(semantic_gt).fill_(-float('inf'))
            semantic_gt = torch.where(semantic_gt == 0., inf, semantic_gt)
            m = nn.Softmax(dim=2)
            semantic_gt = m(semantic_gt).to(device)
            semantic_gt[idx0_bg, idx1_bg, idx2_bg] = 0.
            msk_max, _ = semantic_gt.reshape(-1 ,channel).max(1)
            msk = (msk_max >= 0.99999) & (output['weights_0'].reshape(-1) > cfg.weight_th)
            if torch.sum(msk).item() != 0:
                semantic_loss_3d = cross_entropy(output['points_semantic_0'].reshape(-1 ,channel)[msk, :], semantic_gt.reshape(-1 ,channel)[msk, :])
            else:
                semantic_loss_3d = torch.tensor(0.).to(device)
            semantic_loss_3d = cfg.lambda_3d * semantic_loss_3d
            semantic_loss += semantic_loss_3d
            
            if (cfg.use_pspnet == True) and (batch['stereo_num'] == 1):
                semantic_loss = torch.tensor(0.).to(device)
                semantic_loss_3d = torch.tensor(0.).to(device)
                semantic_loss_2d_pred = torch.tensor(0.).to(device)
                semantic_loss_2d_fix = torch.tensor(0.).to(device)
            scalar_stats.update({'semantic_loss_2d_pred': semantic_loss_2d_pred})
            scalar_stats.update({'semantic_loss_2d_fix': semantic_loss_2d_fix})
            scalar_stats.update({'semantic_loss_3d': semantic_loss_3d})
            scalar_stats.update({'semantic_loss': semantic_loss})
            loss += cfg.semantic_weight * semantic_loss
        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

