import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], att_type='sigmoid'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.att_type = att_type

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        if self.att_type == 'sigmoid':
            scale = F.sigmoid(channel_att_sum).unsqueeze(
                2).unsqueeze(3).expand_as(x)
        elif self.att_type == 'softmax':
            scale = F.softmax(channel_att_sum, dim=1).unsqueeze(
                2).unsqueeze(3).expand_as(x)
        else:
            raise ValueError(
                'Invalid attention type: {}'.format(self.att_type))
        return scale, x*scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, att_type='sigmoid'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size-1) // 2, relu=False)
        self.att_type = att_type

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        if self.att_type == 'sigmoid':
            scale = F.sigmoid(x_out)  # broadcasting
        elif self.att_type == 'softmax':
            scale = F.softmax(x_out, dim=1)
        else:
            raise ValueError(
                'Invalid attention type: {}'.format(self.att_type))
        return scale, x*scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, att_type='sigmoid'):
        super(CBAM, self).__init__()
        logger.info(f"{att_type} is used")
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types, att_type)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(att_type)
            logger.info("SpatialGate is used")

    def forward(self, x):
        x_ch, x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_sp, x_out = self.SpatialGate(x_out)
        return x_ch, x_sp
