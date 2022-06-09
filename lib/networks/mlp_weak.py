import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, fr_pos=10, fr_view=4, skips=[4]):
        """
        """
        super(NeRF, self).__init__()
        self.skips = skips
        self.pe0, input_ch = get_embedder(fr_pos, 0)
        self.pe1, input_ch_views = get_embedder(fr_view, 0)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) if i not in self.skips else \
             nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)

        self.semantic_linears_num = 4
        self.semantic_linears = nn.ModuleList(
            [nn.Linear(W, W) for i in range(self.semantic_linears_num)])
        self.semantic_output1 = nn.Linear(W, W//2)
        self.semantic_output2 = nn.Linear(W//2, 50)

        self.rgb_linears.apply(weights_init)
        self.semantic_output1.apply(weights_init)
        self.semantic_output2.apply(weights_init)
        self.semantic_linears.apply(weights_init)
        self.pts_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, xyz, ray_dir):
        B, N_rays, N_samples = xyz.shape[:3]
        xyz, ray_dir = xyz.reshape(-1, 3), ray_dir.reshape(-1, 3)
        ray_dir = ray_dir / ray_dir.norm(dim=-1, keepdim=True)

        input_pts, input_views = self.pe0(xyz), self.pe1(ray_dir)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # alpha
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

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
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.rgb_linears):
            h = self.rgb_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)  

        outputs = torch.cat([rgb, alpha, semantic], -1)
        return outputs.reshape(B, N_rays, N_samples, 4+50)

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