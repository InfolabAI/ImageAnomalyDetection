# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from loguru import logger
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from .gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            # set padding to preserve image size
            nn.Conv2d(in_dim, out_dim//4, 3, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeepGCN, self).__init__()
        channels = args.n_filters
        k = args.k
        act = args.act
        norm = args.norm
        bias = args.bias
        epsilon = args.epsilon
        stochastic = args.use_stochastic
        conv = args.conv
        self.n_blocks = args.n_blocks
        drop_path = args.drop_path
        patchsize = args.patchsize

        self.stem = Stem(in_dim=args.target_embed_dimension,
                         out_dim=channels, act=act)
        # self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(
            k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)  # NOTE knn 의 k 를  layer 진행에 따라 점점 늘림
        max_dilation = patchsize**2 // max(num_knn)

        if args.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(args, channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                              bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4,
                                          act=act, drop_path=dpr[i])
                                      ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(args, channels, num_knn[i], 1, conv, act, norm,
                                              bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4,
                                          act=act, drop_path=dpr[i])
                                      ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(args.dropout),
                              nn.Conv2d(1024, args.n_classes, 1, bias=True))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        x = self.stem(inputs)  # + self.pos_embed

        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        out = self.prediction(x).squeeze(-1).squeeze(-1)
        out = self.last(out)
        return out

    def set_image_paths(self, image_paths):
        self.backbone[-1][0].graph_conv.image_paths = image_paths


def vig_224_gelu(args):
    model = DeepGCN(args)

    return model, args.n_filters
