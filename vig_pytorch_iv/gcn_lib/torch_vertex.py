# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from loguru import logger
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        logger.debug(f"In MRConv2d x.shape: {x.shape}")

        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])

        logger.debug(f"In MRConv2d x_i.shape: {x_i.shape}")
        logger.debug(f"In MRConv2d x_j.shape: {x_j.shape}")

        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)],
                      dim=2).reshape(b, 2 * c, n, _)

        logger.debug(f"In MRConv2d x.shape after reshape: {x.shape}")
        ret = self.nn(x)
        return ret


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        ret = self.nn(torch.cat([x_i, x_j - x_i], dim=1))
        max_value, _ = torch.max(ret, -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        ret = self.nn1(x_j)
        x_j, _ = torch.max(ret, -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, args, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(
            in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(args,
                                                      kernel_size, dilation, stochastic, epsilon)
        self.args = args

    def forward(self, x, relative_pos=None):
        """
        In DyGraphConv2d x.shape: torch.Size([128, 320, 14, 14])
        In DyGraphConv2d x.shape after reshape: torch.Size([128, 320, 196, 1])
        In DyGraphConv2d edge_index.shape after self.dilated_knn_graph: torch.Size([2, 128, 196, 17])
            - edge_index: (2, batch_size, num_points, k) from DenseDilated
        In DyGraphConv2d x.shape after super(DyGraphConv2d, self).forward: torch.Size([128, 640, 196, 1])
        """
        logger.debug(f"In DyGraphConv2d x.shape: {x.shape}")
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        logger.debug(f"In DyGraphConv2d x.shape after reshape: {x.shape}")
        # NOTE batch 내 에서 섞은 image 와 연결 torch.concat([x.unsqueeze(2), x[torch.randperm(x.shape[0])].unsqueeze(2)], dim=2)
        if y is not None:
            logger.debug(f"In DyGraphConv2d y.shape after reshape: {y.shape}")

        iv_edge_index, v_edge_index = self.dilated_knn_graph(
            x, y, relative_pos)
        logger.debug(
            f"In DyGraphConv2d edge_index.shape after self.dilated_knn_graph: {iv_edge_index.shape}")

        iv_x = super(DyGraphConv2d, self).forward(
            x, iv_edge_index, y).reshape(B, -1, H, W).contiguous()
        v_x = super(DyGraphConv2d, self).forward(
            x, v_edge_index, y).reshape(B, -1, H, W).contiguous()
        logger.debug(
            f"In DyGraphConv2d x.shape after super(DyGraphConv2d, self).forward: {x.shape}")

        return iv_x, v_x


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, args, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.args = args
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.graph_conv = DyGraphConv2d(args, in_channels, in_channels * 2, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r)

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.fmask = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                                                                                        int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        """
        In Grapher x.shape: torch.Size([128, 320, 14, 14])                 
        In Grapher x.shape after fc1: torch.Size([128, 320, 14, 14])
        In Grapher x.shape after graph_conv: torch.Size([128, 640, 14, 14])
        In Grapher x.shape after fc2: torch.Size([128, 320, 14, 14]) 
        """
        logger.debug(f"In Grapher x.shape: {x.shape}")
        _tmp = x
        x = self.fc1(x)
        logger.debug(f"In Grapher x.shape after fc1: {x.shape}")
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        iv_x, v_x = self.graph_conv(x, relative_pos)
        B = iv_x.shape[0]  # NOTE 처음이라 batch 로 처리하도록 구현
        logger.debug(f"In Grapher x.shape after graph_conv: {x.shape}")
        out = self.fc2(torch.concat([iv_x, v_x], dim=0))
        logger.debug(f"In Grapher x.shape after fc2: {x.shape}")
        out = self.drop_path(out) + _tmp.repeat(2, 1, 1, 1)
        iv_x, v_x = out[:B], out[B:]
        ivv_x = nn.functional.softmax(
            self.fmask, dim=1) * iv_x \
            + self.args.la_v*v_x
        return torch.concat([ivv_x, iv_x, v_x], dim=0)

        # return nn.functional.softmax(self.fmask, dim=1) * iv_x + self.args.la_v * v_x, iv_x, v_x
