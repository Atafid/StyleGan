import torch
import torch.nn.functional as func
import torch.utils.data
from torch import nn
from labml_helpers.module import Module
import math as m
from .utils import EqualizedLinear, EqualizedWeight, ConvModLayer, UpSample


class MappingNetwork(nn.Module):

    def __init__(self, n_layers, img_features):
        super().__init__()

        layers = []

        for i in range(n_layers):
            layers.append(EqualizedLinear(img_features, img_features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        z = func.normalize(z, dim=1)

        return (self.network(z))


class StyleBlock(nn.Module):
    def __init__(self, in_features, out_features, w_dim):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.w_dim = w_dim

        self.to_style = EqualizedLinear(w_dim, in_features, bias=1.0)

        self.conv = ConvModLayer(
            self.in_features, self.out_features, size=3)

        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise=None):
        A = self.to_style(w)

        x = self.conv(x, A)

        if (noise is not None):
            x = x+self.scale_noise[None, :, None, None]*noise

        return (self.activation(x+self.bias[None, :, None, None]))


class RGBBlock(nn.Module):
    def __init__(self, features, w_dim):
        super().__init__()

        self.features = features

        self.w_dim = w_dim

        self.to_style = EqualizedLinear(self.w_dim, self.features, bias=1.)

        self.conv = ConvModLayer(
            features, 3, size=1, demod=False)
        self.bias = nn.Parameter(torch.zeros(3))

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, w):
        A = self.to_style(w)

        x = self.conv(x, A)

        return (self.activation(x+self.bias[None, :, None, None]))


class GeneratorBlock(nn.Module):
    def __init__(self, in_features, out_features, w_dim):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.w_dim = w_dim

        self.style1 = StyleBlock(in_features, out_features, w_dim)
        self.style2 = StyleBlock(out_features, out_features, w_dim)

        self.rgbBlock = RGBBlock(out_features, w_dim)

    def forward(self, x, w, noise):
        x = self.style1(x, w, noise[0])
        x = self.style2(x, w, noise[1])

        rgb = self.rgbBlock(x, w)

        return (x, rgb)


class Generator(nn.Module):
    def __init__(self, img_res, w_dim, n_features=32, max_features=512):
        super().__init__()

        features = [min(max_features, n_features*(2**i))
                    for i in range(int(m.log2(img_res))-2, -1, -1)]

        self.num_blocks = len(features)

        self.const = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        self.first_block = StyleBlock(features[0], features[0], w_dim)

        self.rgb = RGBBlock(features[0], w_dim)

        block_list = [GeneratorBlock(features[i-1], features[i], w_dim)
                      for i in range(1, self.num_blocks)]
        self.blocks = nn.ModuleList(block_list)

        self.upsample = UpSample()

    def forward(self, w, noise):
        batch_size = w.shape[1]

        x = self.const.expand(batch_size, -1, -1, -1)
        x = self.first_block(x, w[0], noise[0][1])

        rgb = self.rgb(x, w[0])

        for i in range(1, self.num_blocks):
            x = self.upsample(x)

            x, rgb_new = self.blocks[i-1](x, w[i], noise[i])
            rgb = self.upsample(rgb)+rgb_new

        return (rgb)


class GeneratorLoss(Module):
    def forward(self, f_fake):
        return (-f_fake.mean())
