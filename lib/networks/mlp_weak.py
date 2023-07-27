import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import tinycudann as tcnn

class NeRF(nn.Module):
    def __init__(self, D=cfg.feature_layers, W=256, fr_pos=10, fr_view=4, skips=[cfg.feature_layers-4]):
        super(NeRF, self).__init__()
        self.skips = skips
        self.img_code_ch = cfg.img_code_ch
        self.scene_code_ch = cfg.scene_code_ch
        self.pe0, input_ch = get_embedder(fr_pos, 0)
        self.pe1, input_ch_views = get_embedder(fr_view, 0)
        
        # scene feature (hash)
        self.scene_hash_encoding  = tcnn.NetworkWithInputEncoding(
            n_input_dims = 3,
            n_output_dims= W,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 2,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,
            },
        )
        
        # scene feature (mlp)
        self.scene_mlp_linears_num = 6
        self.scene_mlp_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) for i in range(self.scene_mlp_linears_num)] + \
            [nn.Linear(W, W)]
            )
        self.scene_mlp_output = nn.ModuleList(
            [nn.Linear(W * 2, W)] + \
            [nn.Linear(W, W)]
            )
        self.feature_linear = nn.Linear(W, W)
        
        # alpha
        self.alpha_linear = nn.Linear(W, 1)
        
        # rgb & appearance embeddings
        self.rgb_linears = nn.ModuleList([nn.Linear(input_ch_views + W + self.img_code_ch + self.scene_code_ch, W//2)])
        self.rgb_output= nn.Linear(W//2, 3)
        std = 1e-4
        self.img_codes = nn.Parameter(torch.empty(cfg.train_frames*4, self.img_code_ch))
        self.scene_code = nn.Parameter(torch.empty(self.scene_code_ch))
        self.img_codes.data.uniform_(-std, std)
        self.scene_code.data.uniform_(-std, std)

        # semantic
        self.semantic_linears_num = 4
        self.semantic_linears = nn.ModuleList([nn.Linear(W, W) if i==0 else nn.Linear(W, W) for i in range(self.semantic_linears_num)])
        self.semantic_output1 = nn.Linear(W, W//2)
        self.semantic_output2 = nn.Linear(W//2, 50)

        # weight initialization
        self.scene_mlp_linears.apply(weights_init)
        self.scene_mlp_output.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linears.apply(weights_init)
        self.rgb_output.apply(weights_init)
        self.semantic_linears.apply(weights_init)
        self.semantic_output1.apply(weights_init)
        self.semantic_output2.apply(weights_init)
        nn.init.constant_(self.alpha_linear.bias, 0.2)

    def get_normal(self, xyz, z_vals):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(xyz.device)], -1).reshape(-1, 1)

        with torch.enable_grad():
            xyz = xyz.requires_grad_(True)
            h_hash = self.scene_hash_encoding((xyz+1.)/2).float()
            h_mlp = self.pe0(xyz)
            for i, l in enumerate(self.scene_mlp_linears):
                h_mlp = self.scene_mlp_linears[i](h_mlp)
                h_mlp = F.relu(h_mlp)
            h = torch.cat((h_mlp, h_hash), dim=-1)
            for i, l in enumerate(self.scene_mlp_output):
                h = self.scene_mlp_output[i](h)
                h = F.relu(h)

            alpha = self.alpha_linear(h)
            nablas = torch.autograd.grad(
                raw2alpha(alpha, dists),
                xyz,
                torch.ones_like(raw2alpha(alpha, dists), device=xyz.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            nablas = - nablas
        
        return nablas
        
    def forward(self, xyz, z_vals, ray_dir, frame_idx):

        B, N_rays, N_samples = xyz.shape[:3]
        device = xyz.device
        xyz, ray_dir = xyz.reshape(-1, 3), ray_dir.reshape(-1, 3)
        ray_dir = ray_dir / ray_dir.norm(dim=-1, keepdim=True)
        
        # scene feature
        h_hash = self.scene_hash_encoding((xyz+1.)/2).float()
        h_mlp = self.pe0(xyz)
        for i, l in enumerate(self.scene_mlp_linears):
            h_mlp = self.scene_mlp_linears[i](h_mlp)
            h_mlp = F.relu(h_mlp)
        h = torch.cat((h_mlp, h_hash), dim=-1)
        for i, l in enumerate(self.scene_mlp_output):
            h = self.scene_mlp_output[i](h)
            h = F.relu(h)

        feature = self.feature_linear(h)
        alpha = self.alpha_linear(h)

        # normal
        if self.training == True:
            # discard normal prediction in training
            nablas = torch.zeros(B, N_rays, N_samples, 3).to(device)
        else:
            nablas = self.get_normal(xyz, z_vals)
        
        # semantic
        for i, l in enumerate(self.semantic_linears):
            h = self.semantic_linears[i](h)
            h = F.relu(h)
        semantic = self.semantic_output1(h)
        semantic = self.semantic_output2(F.relu(semantic))
        if self.training == False:
            m = nn.Softmax(dim=1)
            semantic = m(semantic)
        
        # rgb
        input_views = self.pe1(ray_dir)
        img_code = self.img_codes[frame_idx].repeat(feature.shape[0],1)
        scene_code = self.scene_code.repeat(feature.shape[0],1)
        h = torch.cat([feature, input_views, img_code, scene_code], dim=-1)
        for i, l in enumerate(self.rgb_linears):
            h = self.rgb_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_output(h)  

        if self.training == False:
            rgb = rgb.detach()
            alpha = alpha.detach()
            semantic = semantic.detach()
            nablas = nablas.detach()
        
        outputs = torch.cat([rgb, alpha, semantic], -1)

        return outputs.reshape(B, N_rays, N_samples, 4+50), nablas.reshape(B, N_rays, N_samples, 3)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)