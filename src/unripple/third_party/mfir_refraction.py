# Copied from https://github.com/iafoss/refractive-mfir-benchmark (data.py and eval.py, commit 45eee65)
# so that benchmark clips can be generated without data.py's training-only imports
# (transformers, albumentations, a CogVideoX tokenizer). NOT our code.
# Changes: dropped the torch.autocast decorators and data.py's first get_refraction (it is
# redefined further down in data.py, so only the second one was ever used). Math is unchanged.
#
# MIT License
#
# Copyright (c) 2026 CVPRW 2026 "A unified Benchmark for Multi-Frame Image Restoration under Severe Refractive Warping" Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F

# amplitude per wave type and level, from eval.py (used as both `scale` and `depth` in gen_video)
amplitudes = {
    'ocean':{'low':0.329, 'mid':0.574, 'high':0.983, 'extreme':1.642},
    'shallow':{'low':0.138, 'mid':0.244, 'high':0.433, 'extreme':0.695},
    'sine':{'low':0.204, 'mid':0.358, 'high':0.617, 'extreme':1.027},
    'ripple':{'low':0.315, 'mid':0.548, 'high':0.943, 'extreme':1.6},
}
paths = {
    'ocean': 'Ocean_waves', 'shallow': 'Shallow_waves',
    'sine': 'Sine_waves', 'ripple': 'Ripples',
}


@torch.no_grad()
def get_refraction(norm_r,norm_s,n=1.33):
    return n*torch.cross(norm_s, torch.cross(-norm_s, norm_r, dim=-1), dim=-1) - \
      norm_s*torch.nan_to_num(torch.sqrt(1 - torch.pow(n*torch.linalg.norm(torch.cross(norm_s, norm_r, dim=-1), dim=-1),2))).unsqueeze(-1)

@torch.no_grad()
def gen_video(x, norm_s, scale=[0.75,1.25], depth=[0.75,1.25], n=1.33, mode='val', x2_sample=False): # B L C H W
    if not isinstance(scale,list) and not isinstance(scale,set) and not isinstance(scale,tuple): scale = (scale,scale)
    if not isinstance(depth,list) and not isinstance(depth,set) and not isinstance(depth,tuple): depth = (depth,depth)
    B,_,_,H,W = x.shape
    _,L,_,h,w = norm_s.shape
    device = x.device

    idx_i, idx_j = torch.arange(H, device=device)[:,None], torch.arange(W, device=device)[None,:]
    idx_i, idx_j = (idx_i/(H-1) - 0.5), (idx_j/(W-1) - 0.5)
    grid0 = torch.stack([idx_j.expand([H,-1]), idx_i.expand([-1,W])],-1).unsqueeze(0)

    if W >= H: rh,rw = H/W,1
    else: rh,rw = 1,W/H
    if mode == 'val':
        lh,sh = rh*torch.ones(B, device=device),(0.5-rh/2)*torch.ones(B, device=device)
        lw,sw = rw*torch.ones(B, device=device),(0.5-rw/2)*torch.ones(B, device=device)
    else:
        lh,lw = (0.25+0.75*torch.rand(B, device=device))*rh,(0.25+0.75*torch.rand(B, device=device))*rw
        sh,sw = (1.0 - lh)*torch.rand(B, device=device), (1.0 - lw)*torch.rand(B, device=device)

    norms = []
    for i in range(B):
        norm = norm_s[i, :, :, int(h*sh[i]):int(h*(sh[i] + lh[i])), int(w*sw[i]):int(w*(sw[i] + lw[i]))]
        norm = F.interpolate(norm, size=(H,W),mode='bilinear')
        norms.append(norm)
    norm_s = torch.stack(norms,0).permute(0,1,3,4,2)
    norm_s = F.normalize(norm_s,dim=-1).flatten(0,1)

    s = scale[0] + (scale[1] - scale[0])*torch.rand(1, device=device)
    d = depth[0] + (depth[1] - depth[0])*torch.rand(1, device=device)
    n0 = torch.zeros(1,H,W,3, device=device)
    n0[:,:,:,2] = 1
    v = get_refraction(n0,-(norm_s*s+n0*(1 - s)), n=n)[:,:,:,:2]
    grid = 2.0*(v*d + grid0) #-1,1 range
    if not x2_sample:
        deformed = F.grid_sample(x.expand(-1,L,-1,-1,-1).flatten(0,1), grid,
                                        mode='bicubic', padding_mode='zeros', align_corners=True)
    else:
        grid = F.interpolate(grid.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)
        deformed = F.grid_sample(x.expand(-1,L,-1,-1,-1).flatten(0,1), grid,
                                            mode='bicubic', padding_mode='zeros', align_corners=True)
        deformed = F.interpolate(deformed, scale_factor=0.5, mode='bilinear', antialias=True)
    return torch.nan_to_num(deformed.view(B,L,-1,H,W)).clip(0,1)
