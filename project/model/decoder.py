import torch.nn as nn
import methods

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        res_block_cnt = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  methods.ResidualBock(in_channels, in_channels),
                  methods.NonLocalBlock(in_channels),
                  methods.ResidualBock(in_channels, in_channels)]
        
        for layer in range(len(channels)):
            out_channels = channels[layer]
            for res_block in range(res_block_cnt):
                layers.append(methods.ResidualBock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(methods.NonLocalBlock(in_channels))
            if layer != 0:
                layers.append(methods.UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(methods.GroupNorm(in_channels))
        layers.append(methods.Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)