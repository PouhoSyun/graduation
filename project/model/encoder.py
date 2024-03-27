import torch.nn as nn
import methods 

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        res_block_cnt = 2
        resolution = 256
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        for layer in range(len(channels) - 1):
            in_channels = channels[layer]
            out_channels = channels[layer + 1]
            for res_block in range(res_block_cnt):
                layers.append(methods.ResidualBock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(methods.NonLocalBlock(in_channels))
            if layer != len(channels) - 2:
                layers.append(methods.DownSampleBlock(channels[layer + 1]))
                resolution //= 2
        
        layers.append(methods.ResidualBock(channels[-1], channels[-1]))
        layers.append(methods.NonLocalBlock(channels[-1]))
        layers.append(methods.ResidualBock(channels[-1], channels[-1]))
        layers.append(methods.GroupNorm(channels[-1]))
        layers.append(methods.Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)