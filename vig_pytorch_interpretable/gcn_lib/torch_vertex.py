# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
from loguru import logger
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.cbam import CBAM, CBAM_spatial


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
        if args.bam_type == 'cbam':
            bam_type = CBAM
        elif args.bam_type == 'cbam_channel':
            bam_type = CBAM
        elif args.bam_type == 'cbam_spatial':
            bam_type = CBAM_spatial
        elif args.bam_type == 'bam':
            # NOTE bam 은 너무 느려서 못 씀
            # bam_type = BAM
            raise NotImplementedError(
                'bam_type:{} is too slow. so, it is not supported'.format(args.bam_type))
        else:
            raise NotImplementedError(
                'bam_type:{} is not supported'.format(args.bam_type))
        self.att = bam_type(
            in_channels, args.bam_reduction_ratio, no_spatial=True if args.bam_type == 'cbam_channel' else False, att_type=args.bam_att_type)

        self._init_image_paths()

    def _init_image_paths(self):
        self.image_paths = None

    def _set_image_paths(self, image_paths):
        self.image_paths = image_paths

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

        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        logger.debug(
            f"In DyGraphConv2d edge_index.shape after self.dilated_knn_graph: {edge_index.shape}")

        # NOTE edge 를 생성하고 나서 explain 을 만들어서 적용해야 SIGNET 과 동일
        att = self._build_explain(x=x.reshape(B, C, H, W).contiguous()).reshape(
            B, C, -1, 1).contiguous()
        indices_H, indices_W = self._extract_explain(att, H, W)
        self._visualize_explain(indices_H, indices_W, H, W, edge_index)
        x = x * att

        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        logger.debug(
            f"In DyGraphConv2d x.shape after super(DyGraphConv2d, self).forward: {x.shape}")

        return x.reshape(B, -1, H, W).contiguous()

    def _build_explain(self, x):
        scalar = 20
        eps = 1e-10
        B, C, H, W = x.shape
        att = self.att(x)
        att_softmax = torch.nn.functional.softmax(
            att.view(B, C, -1), dim=2).view(B, C, H, W)
        att_max = att_softmax.view(B, C, -1).max(dim=2)[0].view(B, C, 1, 1)

        att = att_softmax / (att_max + eps)
        att = (2 * att - 1)/(2 * scalar) + 1
        return att

    def _extract_explain(self, att, H, W, topk=20):
        # [B, C, H*W, 1] -> [B, 1, H*W, 1] -> [B, H*W]
        att_for_explain = att.mean(1, keepdim=True).squeeze(3).squeeze(1)
        indices = torch.topk(att_for_explain, topk, dim=1)[1]
        indices_H, indices_W = indices // W, indices % W
        return indices_H, indices_W

    def _visualize_explain(self, indices_H, indices_W, H, W, edge_index, output_path='/home/robert.lim/main/other_methods/my_GNN_SimpleNet/images'):
        if self.image_paths is None:  # NOTE predict 전에 _preprocessing_predict 에서 설정하도록 되어있음
            return

        image_paths = self.image_paths

        indices_H, indices_W, edge_index = indices_H.cpu().numpy(
        ), indices_W.cpu().numpy(), edge_index.cpu().numpy()

        for batch_i, (image_path, Hs, Ws) in enumerate(zip(image_paths, indices_H, indices_W)):
            image = Image.open(image_path)
            image_np = np.array(image)

            patch_size_H, patch_size_W = image_np.shape[0] // H, image_np.shape[1] // W
            fig, ax = plt.subplots(1)
            ax.imshow(image_np)

            # 패치 강조
            pos1ds = Hs*W+Ws
            for cur_pos1d, h, w in zip(pos1ds, Hs, Ws):
                rect = patches.Rectangle((w * patch_size_W, h * patch_size_H), patch_size_W,
                                         patch_size_H, linewidth=1, edgecolor='r', facecolor='none')
                edges_from_dist = edge_index[:, batch_i, h*W+w, :]
                edges_for_explain = np.intersect1d(
                    edges_from_dist[0], pos1ds)
                ax.add_patch(rect)
                for dst_pos1d in edges_for_explain:
                    srch, srcw = cur_pos1d // W, cur_pos1d % W
                    dsth, dstw = dst_pos1d // W, dst_pos1d % W
                    # NOTE draw
                    src_center = ((srcw + 0.5) * patch_size_W,
                                  (srch + 0.5) * patch_size_H)
                    dst_center = ((dstw + 0.5) * patch_size_W,
                                  (dsth + 0.5) * patch_size_H)
                    ax.plot([src_center[0], dst_center[0]], [
                            src_center[1], dst_center[1]], 'b-', linewidth=0.5)
                # get intersection between edges ans positions_1d

            plt.axis('off')
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(
                output_path, '_'.join(image_path.split('/')[-2:])), bbox_inches='tight', pad_inches=0)
            plt.close()

        self._init_image_paths()


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
        x = self.graph_conv(x, relative_pos)
        logger.debug(f"In Grapher x.shape after graph_conv: {x.shape}")
        x = self.fc2(x)
        logger.debug(f"In Grapher x.shape after fc2: {x.shape}")
        x = self.drop_path(x) + _tmp
        return x
