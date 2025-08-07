from torch import nn

from .SwinTransformer import SwinB
import torch.nn.functional as F
import torch
import math
from einops import rearrange
import numpy as np  
import cv2 

def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU())

def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32, device='cuda') 

    def __call__(self, b, h, w):
        mask = torch.zeros([b, h, w], dtype=torch.bool, device='cuda')
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  
        if self.normalize:
            eps = 1e-6
            y_embed = ((y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale).cuda()
            x_embed = ((x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale).cuda()

        dim_t = self.temperature ** (2 * (self.dim_t // 2) / self.num_pos_feats) 

        pos_x = x_embed[:, :, :, None] / dim_t 
        pos_y = y_embed[:, :, :, None] / dim_t  
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # b 128 h w


class MSFFM(nn.Module):  
    def __init__(self, channel, num_heads, pool_ratios=[2, 4, 8]):
        super(MSFFM, self).__init__()
        self.att=nn.MultiheadAttention(channel,num_heads=num_heads,dropout=0.2)
        # self.att2=nn.MultiheadAttention(channel,num_heads,dropout=0.1)

        self.pos=PositionEmbeddingSine(num_pos_feats=channel//2,normalize=True)
        self.dropout1=nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.1)
        self.activate=nn.ReLU()
        self.norm1=nn.LayerNorm(channel)
        self.norm2=nn.LayerNorm(channel)
        self.linear1=nn.Linear(channel,channel*2)
        self.linear2=nn.Linear(channel*2,channel)

        self.pool_ratios=pool_ratios
       

    def forward(self,x):
        b,c,h,w=x.shape
        q=x
        pools=[]
        pool_poses=[]
        for pool_ratio in self.pool_ratios:
            pool_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool=F.adaptive_avg_pool2d(x,pool_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))

            pool_pos=self.pos(pool.shape[0], pool.shape[2], pool.shape[3])
            pool_poses.append(rearrange(pool_pos, 'b c h w -> (h w) b c'))  

        pools=torch.cat(pools,0)
        pool_poses=torch.cat(pool_poses,0)

        q_pos=self.pos(q.shape[0],q.shape[2],q.shape[3])
        q_pos=rearrange(q_pos, 'b c h w -> (h w) b c')
        q=rearrange(q, 'b c h w -> (h w) b c')

        att = q + self.dropout1(self.att(q+q_pos, pools + pool_poses, pools)[0])
        att = self.norm1(att)
        att = att + self.dropout2(self.linear2(self.dropout1(self.activate(self.linear1(att)).clone())))
        att = self.norm2(att)

        return rearrange(att,"(h w) b c -> b c h w", h=h, w=w)

class AdaptivePatchModule(nn.Module):
    def __init__(self, output_dim, patch_size, dropout_rate=[0.1,0.2]):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size

        # Multi-layer Perceptron for local features
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm1 = nn.LayerNorm(output_dim // 2)
        self.dropout1 = nn.Dropout(dropout_rate[0])

        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(dropout_rate[1])

        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        
        # Learnable parameters
        # 一维张量，包含output_dim个元素
        self.prompt = nn.Parameter(torch.randn(output_dim, requires_grad=True))
        # 二维张量，output_dim*output_dim
        self.top_down_transform = nn.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        # Rearranging dimensions: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Extract local patches
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P * W/P, P * P, C)

        # Calculate mean of each patch to reduce dimensionality
        local_patches = local_patches.mean(dim=-1)  # (B, H/P * W/P, P * P)

        # First MLP
        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, output_dim // 2)
        local_patches = self.norm1(local_patches)  # Layer normalization
        local_patches = self.dropout1(local_patches)  # Apply dropout

        # Second MLP
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)
        local_patches = self.norm2(local_patches)  # Layer normalization
        local_patches = self.dropout2(local_patches)  # Apply dropout

        # Calculate local attention
        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)

        # Weighted local output
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        # Calculate cosine similarity with prompt
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # (B, N, 1)

        # Clamping mask values
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask  # Apply mask

        # Transformation with top-down matrix
        local_out = local_out @ self.top_down_transform

        # Reshape and interpolate
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)  # (B, output_dim, H/P, W/P)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)

        # Final convolution
        output = self.conv(local_out)

        return output

class AdaptiveMultiBranchPatchModule(nn.Module):
    def __init__(self,in_channels,out_channels) :
        super().__init__()
        self.lga1=AdaptivePatchModule(out_channels,2)
        self.lga2=AdaptivePatchModule(out_channels,4)
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU())
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.drop = nn.Dropout2d(0.1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x=self.conv1(x)
        l1=self.lga1(x)
        l2=self.lga2(x)
        l3=self.conv2(x)
        x=l1+l2+l3
        x=self.relu(self.bn(self.drop(x)))
        return x


class CAMRB(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=16):
        super(CAMRB, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio

        # 高度深度都变成1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.activate=nn.ReLU()

        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1)

        self.conv=make_cbr(self.in_channels,self.out_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activate(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activate(self.fc1(max_pool_out)))

        out = avg_out + max_out
        out=self.sigmoid(out) 

        x=self.conv(x)
        out=out*x+x
        return out

class SAMRB(nn.Module):
    def __init__(self,out_channels):
        super(SAMRB, self).__init__()
        self.conv = nn.Conv2d(2, out_channels, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)# 计算某一个维度上的平均均值，保持维度值不变
        max_out, _ = torch.max(x, dim=1, keepdim=True)# 返回指定维度上的最大值以及索引
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out=self.sigmoid(out)

        out=out*x+x
        return out

class CSAMRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSAMRB, self).__init__()
        self.CAMRB=CAMRB(in_channels,out_channels)
        self.SAMRB=SAMRB(out_channels)

    def forward(self,x):
        x=self.CAMRB(x)
        x=self.SAMRB(x)
        return x
        

class Nets(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinB(pretrained=True)
        emb_dim = 128

        self.output5 =AdaptiveMultiBranchPatchModule(1024, emb_dim)
        self.output4 =AdaptiveMultiBranchPatchModule(512, emb_dim)
        self.output3 =AdaptiveMultiBranchPatchModule(256, emb_dim)
        self.output2 =AdaptiveMultiBranchPatchModule(128, emb_dim)
        self.output1 =AdaptiveMultiBranchPatchModule(128, emb_dim)


        self.CSAMRB5 =CSAMRB(emb_dim, emb_dim)
        self.CSAMRB4 =CSAMRB(emb_dim, emb_dim)
        self.CSAMRB3 =CSAMRB(emb_dim, emb_dim)
        self.CSAMRB2 =CSAMRB(emb_dim, emb_dim)
        self.CSAMRB1 =CSAMRB(emb_dim, emb_dim)

        self.msffm5=MSFFM(emb_dim,num_heads=1)
        self.msffm4=MSFFM(emb_dim,num_heads=2)
        self.msffm3=MSFFM(emb_dim,num_heads=4)
        self.msffm2=MSFFM(emb_dim,num_heads=8)
        self.msffm1=MSFFM(emb_dim,num_heads=8)

        self.conv=make_cbr(emb_dim,1)
        self.last_activation = nn.Sigmoid()

    def forward(self,x):
        feature = self.backbone(x)
        e5 = feature[4]  
        e4 = feature[3]  
        e3 = feature[2]  
        e2 = feature[1] 
        e1 = feature[0] 

        x5 = self.output5(e5)  
        x4 = self.output4(e4) 
        x3 = self.output3(e3)     
        x2 = self.output2(e2)
        x1 = self.output1(e1)

        x5 = self.CSAMRB5(x5)  
        x4 = self.CSAMRB4(x4) 
        x3 = self.CSAMRB3(x3)     
        x2 = self.CSAMRB2(x2)
        x1 = self.CSAMRB1(x1)
        
        x5 = self.msffm5(x5)  
        x4 = self.msffm4(x4)  
        x3 = self.msffm3(x3)  
        x2 = self.msffm2(x2) 
        x1 = self.msffm1(x1) 

        x5 = resize_as(x5,x)  
        x4 = resize_as(x4,x)  
        x3 = resize_as(x3,x)  
        x2 = resize_as(x2,x) 
        x1 = resize_as(x1,x) 

        e=x5+x4+x3+x2+x1

        

        return self.last_activation(self.conv(e))


