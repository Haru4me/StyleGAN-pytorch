import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

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

    
# FC-Layer with Equalized learning rate
class EqualLinear(nn.Module):

    def __init__(self, inp, outp):

        super(EqualLinear, self).__init__()

        linear = nn.Linear(inp, outp)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, x):
        return self.linear(x)


"""
    Mapping Network (f)
"""

class MapNet(nn.Module):

    def __init__(self, inp=512, num=8, alpha=0.2):

        super(MapNet,self).__init__()

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
        self.afine = EqualLinear(style_size, 2*in_channel)

        self.afine.linear.bias.data[:in_channel] = 1
        self.afine.linear.bias.data[in_channel:] = 0

    def forward(self, x, w):
        y_s, y_b = self.afine(w).unsqueeze(2).unsqueeze(3).chunk(2, 1)
        return y_s*self.norm(x)+y_b


"""
    Syntesis Network (g)
"""

#------------------------НЕ-МОЕ--------------------------------------


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(
            input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, padding=0):

        super(FusedDownsample, self).__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):

        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class BlurFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):

    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(
            grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):

    def __init__(self, channel):

        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

#--------------------------------------------------------------------


class EqualConv2d(nn.Module):

    def __init__(self, *args, **kwargs):

        super(EqualConv2d, self).__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class StyleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, 
                 padding=1, style_dim=512, upsample_type=None, 
                 init=False):

        super(StyleBlock, self).__init__()

        layer = []
        if upsample_type == None:
            layer = [EqualConv2d(in_channel, out_channel,
                                 kernel_size, padding=padding)]
        elif upsample_type == 'fusion':
            layer = [FusedUpsample(in_channel, out_channel, kernel_size, padding=padding),
                     Blur(out_channel)]
        elif upsample_type == 'normal':
            layer = [nn.Upsample(scale_factor=2, mode='nearest'),
                     EqualConv2d(in_channel, out_channel, kernel_size, padding=padding), 
                     Blur(out_channel)]
        
        layer.append(equal_lr(AddNoise(out_channel)))
        self.A1 = AdaIN(out_channel, style_dim)
        self.block1 = nn.Sequential(*layer)

        self.A2 = AdaIN(out_channel, style_dim)
        self.block2 = nn.Sequential(EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
                                    equal_lr(AddNoise(out_channel)))
        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)

    def forward(self, x, w):

        x = F.leaky_relu(self.A1(self.block1(x),w), 0.2)
        out = F.leaky_relu(self.A1(self.block2(x),w), 0.2)

        return out



class SynNet(nn.Module):
    
    def __init__(self):
        
        super(SynNet, self).__init__()

        self.progression = nn.ModuleList(
            [
                StyleBlock(512, 512, 3, 1, init=True),         # 4
                StyleBlock(512, 512, 3, 1, upsample='normal'), # 8
                StyleBlock(512, 512, 3, 1, upsample='normal'), # 16
                StyleBlock(512, 512, 3, 1, upsample='normal'), # 32
                StyleBlock(512, 256, 3, 1, upsample='normal'), # 64
                StyleBlock(256, 128, 3, 1, upsample='fusion'), # 128
                StyleBlock(128, 64, 3, 1, upsample='fusion'),  # 256
                StyleBlock(64, 32, 3, 1, upsample='fusion'),   # 512
                StyleBlock(32, 16, 3, 1, upsample='fusion'),   # 1024
            ]
        )
        
        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1), 
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )
    
    def forward(self, style, step=0, alpha=-1, mixing_range=None):
        
        batchsize = style.size(0)
        x = torch.ones((batchsize, 512, 4, 4), dtype=torch.tensor)





