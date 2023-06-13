from torch import nn
from labml_helpers.module import Module
import torch.nn.functional as func
import math as m
from .utils import EqualizedConv2d, DownSample, MiniBatchStdDev, EqualizedLinear


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fst_part = nn.Sequential(DownSample(), EqualizedConv2d(
            in_features, out_features, size=1, padding=0))

        self.snd_part = nn.Sequential(EqualizedConv2d(in_features, in_features, size=3, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      EqualizedConv2d(in_features, out_features, size=3, padding=1), nn.LeakyReLU(0.2, True))

        self.downsample = DownSample()
        self.scale = 1/m.sqrt(2)

    def forward(self, x):
        x_fst = self.fst_part(x)

        x = self.snd_part(x)
        x = self.downsample(x)

        return ((x_fst+x)*self.scale)


class Discriminator(nn.Module):
    def __init__(self, img_res, n_features=64, max_features=512):
        super().__init__()

        self.from_rgb = nn.Sequential(EqualizedConv2d(
            3, n_features, 1), nn.LeakyReLU(0.2, True))

        log_res = int(m.log2(img_res))

        features_size = [min(max_features, n_features*(2**i))
                         for i in range(log_res-1)]

        num_blocks = len(features_size)-1
        self.blocks = nn.Sequential(*[DiscriminatorBlock(
            features_size[i], features_size[i+1]) for i in range(num_blocks)])

        self.std_dev = MiniBatchStdDev()

        last_features = features_size[-1]+1
        self.last_conv = EqualizedConv2d(last_features, last_features, 3)

        self.final = EqualizedLinear(2 * 2 * last_features, 1)

    def forward(self, x):
        x = x-0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.last_conv(x)
        x = x.reshape(x.shape[0], -1)

        return (self.final(x))


class DiscriminatorLoss(Module):
    def forward(self, f_real, f_fake):
        return (func.relu(1-f_real).mean(), func.relu(1+f_fake).mean())
