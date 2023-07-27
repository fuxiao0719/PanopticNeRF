import torch
import torch.nn.functional as F
from lib.config import cfg
import torch.nn as nn
TINY_NUMBER = 1e-6

def sample_along_ray(near, far, N_samples):
    z_steps = torch.linspace(0, 1, N_samples, device=near.device)[None, None]
    z_vals = near[..., None] * (1 - z_steps) + far[..., None] * z_steps
    return z_vals

def sample_along_ray_distributed(near, far, N_samples):
    z_steps = torch.linspace(0, 1, N_samples, device=near.device)[None, None]
    z_vals = near[..., None] * (1 - z_steps) + far[..., None] * z_steps
    return z_vals

def raw2outputs_label_snerf(raw, z_vals, rays_d, white_bkgd=False, one_hot_all_instance=None, is_test = False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)
    rgb = torch.sigmoid(raw[...,:3])
    alpha = raw2alpha(raw[...,3], dists)
    semantic = raw[...,4:]
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(T[..., 0:1]), T], dim=-1)
    weights = alpha * T
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    semantic_map = torch.sum(weights[...,None] * semantic, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    ret = {'rgb': rgb_map, 'depth': depth_map, 'weights': weights, 'z_vals':z_vals, 'semantic_map': semantic_map, 'semantic_point':semantic}
    return ret

def raw2outputs_label(raw, z_vals, rays_d, fix_label, white_bkgd=False, one_hot_all_instance=None, is_test = False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)
    rgb = torch.sigmoid(raw[...,:3])
    alpha = raw2alpha(raw[...,3], dists)
    semantic = raw[...,4:]
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(T[..., 0:1]), T], dim=-1)
    weights = alpha * T
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    semantic_map = torch.sum(weights[...,None] * semantic, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    fix_semantic_map = torch.sum(weights[...,None] * fix_label.to(weights), -2)
    ret = {'rgb': rgb_map, 'depth': depth_map, 'weights': weights, 'z_vals':z_vals, 'semantic_map': semantic_map, 'semantic_point':semantic, 'fix_semantic_map': fix_semantic_map}
    if is_test == True:
        instance_map = torch.sum(weights[...,None] * one_hot_all_instance.to(weights), -2)
        ret['instance_map'] = instance_map
    return ret

def raw2weights(raw, z_vals, rays_d, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    #dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)
    dists = torch.cat([dists, dists[...,-2:-1]], -1) # for lidar samples)
    rgb = torch.sigmoid(raw[...,:3])
    alpha = raw2alpha(raw[...,3], dists)
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(T[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    ret = {'rgb': rgb_map, 'depth': depth_map, 'weights': weights}
    return ret
    
def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)
    
    rgb = torch.sigmoid(raw[...,:3])
    alpha = raw2alpha(raw[...,3], dists)

    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(T[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    ret = {'rgb': rgb_map, 'depth': depth_map, 'weights': weights}
    return ret

def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]
    return z_vals

def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER      # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00       # prevent outlier samples
    
    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
    denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples
