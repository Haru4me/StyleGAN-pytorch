import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, pi
import random

from torch.autograd import Function


# PixelNorm (z -> Normalization(z))
class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLR:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


"""
    Mapping Network (f)
"""


class MapNet(nn.Module):

    def __init__(self, inp=512, num=6, alpha=0.2):

        super(MapNet, self).__init__()

        layers = [PixelNorm()]

        for _ in range(num):
            layers.append(EqualLinear(inp, inp))
            layers.append(nn.LeakyReLU(alpha))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


"""
    Adding Noise Block
"""


class AddNoise(nn.Module):

    def __init__(self, inp_channel):

        super(AddNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, inp_channel, 1, 1))

    def forward(self, x):

        size = list(x.size())
        size[1] = 1
        noise = torch.randn(size)

        return x + self.weight * noise


"""
    Adaptive Instance Norm
"""


class AdaIN(nn.Module):

    def __init__(self, in_channel, style_size):

        super(AdaIN, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.afine = nn.Linear(style_size, 2*in_channel)

        self.afine.bias.data[:in_channel] = 1
        self.afine.bias.data[in_channel:] = 0

    def forward(self, x, w):
        y_s, y_b = self.afine(w).unsqueeze(2).unsqueeze(3).chunk(2, 1)
        return y_s*self.norm(x)+y_b


"""
    Syntesis Network (g)
"""


class Blur(nn.Module):

    def __init__(self, channel):

        super(Blur, self).__init__()

        kernel_size = 3
        sigma = 1

        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2
        variance = sigma**2

        gaussian_kernel = (1./(2.*pi*variance)) *\
            torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channel, 1, 1, 1)
        gaussian_filter = nn.Conv2d(in_channels=channel, out_channels=channel,
                                    kernel_size=3, groups=channel, padding=1, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        self.blur = gaussian_filter

    def forward(self, input):
        return self.blur(input)


class StyleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, style_dim=512, upsample=False):

        super(StyleBlock, self).__init__()

        net = []

        if upsample == True:
            net.append(nn.Upsample(scale_factor=2, mode='nearest'))

        net.append(EqualConv2d(in_channel, out_channel,
                               kernel_size, padding=padding))
        net.append(Blur(out_channel))
        net.append(equal_lr(AddNoise(out_channel)))

        self.block1 = nn.Sequential(*net)
        self.A = AdaIN(out_channel, style_dim)
        self.block2 = nn.Sequential(
            EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
            equal_lr(AddNoise(out_channel))
        )

    def forward(self, img, style):
        img = F.leaky_relu(self.A(self.block1(img), style), 0.2)
        return F.leaky_relu(self.A(self.block2(img), style), 0.2)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, kernel_size2=None, padding2=None, downsample=False):

        super(ConvBlock, self).__init__()

        if kernel_size2 is None and padding2 is None:
            kernel_size2 = kernel_size
            padding2 = padding

        net = [Blur(out_channel),
                EqualConv2d(out_channel, out_channel, 
                kernel_size2, padding=padding2)
            ]

        if downsample == True:
            net.append(nn.AvgPool2d(2))

        net.append(nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(*net)
        self.block1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel,kernel_size, padding=padding),
            nn.LeakyReLU(0.2)
        )


    def forward(self, img):
        res = self.block1(img)
        return self.block2(res)


class SynNet(nn.Module):

    def __init__(self):

        super(SynNet, self).__init__()

        self.styled = nn.ModuleList(
            [
                StyleBlock(512, 512, 3, 1, upsample=True),  # 8
                StyleBlock(512, 512, 3, 1, upsample=True),  # 16
                StyleBlock(512, 512, 3, 1, upsample=True),  # 32
                StyleBlock(512, 256, 3, 1, upsample=True),  # 64
                StyleBlock(256, 128, 3, 1, upsample=True),  # 128
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),                     # 8
                EqualConv2d(512, 3, 1),                     # 16
                EqualConv2d(512, 3, 1),                     # 32
                EqualConv2d(256, 3, 1),                     # 64
                EqualConv2d(128, 3, 1)                      # 128
            ]
        )

    def forward(self, style, step=0, alpha=1, break_point=None):

        out = torch.ones((style[0].shape[0], 512, 4, 4), dtype=torch.float)

        for i, conv in enumerate(self.styled):
            
            if break_point is not None and i <= break_point:
                style_it = style[0]
            elif break_point is not None and i > break_point:
                style_it = style[1]
            else:
                style_it = style[0]

            if step > 0 and i == step:
                prev = out
            
            out = conv(out, style_it)

            if i == step:
                out = self.to_rgb[i](out)

                if i > 0 and 0 <= alpha < 1:
                    print(prev.shape, out.shape)
                    skip = self.to_rgb[i-1](prev)
                    skip = F.interpolate(
                        skip, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip + alpha * out

                break

        return out


"""
    Styled Generator (G)
"""

MEANS = {
    'mean': torch.mean,
    'max': torch.max,
    'moda': torch.mode,
    'median': torch.median
    }


class StyleGenerator(nn.Module):

    def __init__(self):

        super(StyleGenerator, self).__init__()

        self.syntnet = SynNet()
        self.mapnet = MapNet()

    def forward(self, latent, step=0, alpha=1, style_weight=0, 
                break_point=None, mean_way=None):

        styles = []

        if type(latent) not in (list, tuple):
            latent = [latent]

        for i in latent:
            styles.append(self.mapnet(i))

        if mean_way is not None:

            styles_norm = []

            for style in styles:

                if mean_way == 'mean':
                    mean_style = MEANS[mean_way](style, dim=0, keepdim=True)
                else:
                    mean_style = MEANS[mean_way](
                        style, dim=0, keepdim=True).values

                styles_norm.append(
                    mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.syntnet(styles, step=step, alpha=alpha, break_point=break_point)


"""
    Styled Decoder (D)
"""


class StyleDiscriminator(nn.Module):

    def __init__(self):
        
        super(StyleDiscriminator, self).__init__()

        self.progress = nn.ModuleList(
            [
                ConvBlock(128, 256, 3, 1, downsample=True),         # 64
                ConvBlock(256, 512, 3, 1, downsample=True),         # 32
                ConvBlock(512, 512, 3, 1, downsample=True),         # 16
                ConvBlock(512, 512, 3, 1, downsample=True),         # 8
                ConvBlock(513, 512, 3, 1, downsample=True)
            ]
        )

        def fromRGBLayer(out_channel):
            layer =  nn.Sequential(
                EqualConv2d(3, out_channel, 1),
                nn.LeakyReLU(0.2)
            )

            return layer

        self.from_rgb = nn.ModuleList(
            [
                fromRGBLayer(128),                                  # 64
                fromRGBLayer(256),                                  # 32
                fromRGBLayer(512),                                  # 16
                fromRGBLayer(512),                                  # 8
                fromRGBLayer(512)                                   # 4
            ]
        )

        self.conval = ConvBlock(512, 512, 3, 1, 4, 0)
        self.n_layer = len(self.progress)
        self.linear = EqualLinear(512, 1)

    def forward(self, img, step=0, alpha=1):

        for i in range(step,-1,-1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](img)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean().expand(out.size(0), 1, 8, 8)
                out = torch.cat([out, mean_std], 1)
            
            out = self.progress[index](out)

            if i > 0 and i == step and 0 <= alpha < 1:
                
                skip = F.avg_pool2d(input, 2)
                skip = self.from_rgb[index + 1](skip)
                out = (1 - alpha) * skip_rgb + alpha * out
        
        out = self.linear(self.conval(out).squeeze(2).squeeze(2))

        return out





