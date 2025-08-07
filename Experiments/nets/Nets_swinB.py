from torch import nn
from .SwinTransformer import SwinB
import torch.nn.functional as F

def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU())

def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)

class Nets(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinB(pretrained=True)
        emb_dim = 128

        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.conv=make_cbr(emb_dim,1)
        self.last_activation = nn.Sigmoid()

    def forward(self,x):
        feature = self.backbone(x)
        e5 = resize_as(self.output5(feature[4]),x)  
        e4 = resize_as(self.output4(feature[3]),x)  
        e3 = resize_as(self.output3(feature[2]),x)  
        e2 = resize_as(self.output2(feature[1]),x) 
        e1 = resize_as(self.output1(feature[0]),x) 

        e=e5+e4+e3+e2+e1

        return self.last_activation(self.conv(e))


