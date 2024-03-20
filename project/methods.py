import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, channel_cnt):
        super(GroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(group_cnt=32, channel_cnt=channel_cnt, eps=1e-6, affine=True)

    def forward(self, x):
        return self.group_norm(x)
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# residual block for resnet structure, learn to make loss self.block(x) -> 0
class ResidualBock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        self.channels_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if(self.in_channels != self.out_channels):
            return self.channels_up(x) + self.block(x)
        else:
            return x + self.block(x)
        
class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv_layer = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv_layer(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv_layer = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        return self.conv_layer(x)

# self attention layer usage -> non-local block module
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        self.group_norm = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.group_norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A