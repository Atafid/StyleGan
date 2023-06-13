import torch
import torch.nn.functional as func
import torch.utils.data
from torch import nn
import math as m
import numpy as np


class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=0.):
        super().__init__()

        self.bias = nn.Parameter(torch.ones(out_features)*bias)
        self.weight = EqualizedWeight([out_features, in_features])

    def forward(self, z):
        return (func.linear(z, self.weight(), bias=self.bias))


class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.c = 1/m.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return (self.weight*self.c)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, size, padding=0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.kern_size = size
        self.padding = padding

        self.weight = EqualizedWeight([out_features, in_features, size, size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        return (func.conv2d(x, self.weight(), bias=self.bias, padding=self.padding))


class ConvModLayer(nn.Module):
    def __init__(self, in_features, out_features, size, demod=True, eps=1e-8):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.demod = demod
        self.padding = (size-1)//2

        self.eps = eps

        self.weight = EqualizedWeight([out_features, in_features, size, size])

    def forward(self, x, s):
        b, i, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]

        weights = weights*s

        if (self.demod):
            sigma_inv = torch.rsqrt(
                (weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = func.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        self.smooth = Smooth()

    def forward(self, x):
        return (self.smooth(self.up(x)))


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()

        self.smooth = Smooth()

    def forward(self, x):
        x = self.smooth(x)
        return (func.interpolate(x, (x.shape[2]//2, x.shape[3]//2), mode='bilinear', align_corners=False))


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()

        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]

        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()

        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(-1, 1, h, w)
        x = self.pad(x)

        x = func.conv2d(x, self.kernel)
        return (x.view(b, c, h, w))


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()

        self.group_size = group_size

    def forward(self, x):
        assert (x.shape[0] % self.group_size == 0)

        grouped = x.view(self.group_size, -1)

        std = torch.sqrt(grouped.var(dim=0)+1e-8)
        std = std.mean().view(1, 1, 1, 1)

        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)

        return (torch.cat([x, std], dim=1))


class GradientPenalty(nn.Module):
    def forward(self, x, d):
        batch_size = x.shape[0]

        gradients, * \
            _ = torch.autograd.grad(
                outputs=d, inputs=x, grad_outputs=d.new_ones(d.shape), create_graph=True)
        gradients = gradients.reshape(batch_size, -1)

        norm = gradients.norm(2, dim=-1)

        return (torch.mean(norm**2))


class PathLengthPenalty(nn.Module):
    def __init__(self, beta):
        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):
        device = x.device

        img_size = x.shape[2] * x.shape[3]

        y = torch.randn(x.shape, device=device)

        output = (x*y).sum() / m.sqrt(img_size)
        gradients, *_ = torch.autograd.grad(outputs=output, inputs=w, grad_outputs=torch.ones(
            output.shape, device=device), create_graph=True)

        norm = (gradients**2).sum(dim=2).mean(dim=1).sqrt()

        if (self.steps > 0):
            a = self.exp_sum_a / (1-self.beta**self.steps)
            loss = torch.mean((norm-a)**2)

        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()

        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1-self.beta)
        self.steps.add_(1.)

        return (loss)
