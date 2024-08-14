import torch
import torch.nn as nn
from trainer.trainer_simplenet import Trainer_SimpleNet
from vig_pytorch_pretrained.gcn_lib import Grapher
from trainer.vig_wrapper import VIG_wrapper, common_process


class Trainer_Ours_Score_PatchLevel_Mid_Few(Trainer_SimpleNet):
    def set_backbone(self, backbone, device):
        """
        graphcore 에서 다른 backbone 을 사용하기 위함
        """
        self.n_blocks = 12  # number of basic blocks in the backbone
        self.k = 9  # neighbor num (default:9)
        self.conv = 'mr'  # graph conv layer {edge, mr}
        self.act = 'gelu'
        self.norm = 'batch'
        self.bias = True  # bias of conv layer True or False
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.num_knn = [int(x.item()) for x in torch.linspace(
            self.k, 2*self.k, self.n_blocks)]  # number of knn's k

        backbone.layer1 = self._intervene_layer(backbone.layer1, 0)
        backbone.layer2 = self._intervene_layer(backbone.layer2, 1)
        backbone.layer3 = self._intervene_layer(backbone.layer3, 2)
        backbone.layer4 = self._intervene_layer(backbone.layer4, 3)

        self.backbone = backbone.to(device)
        self._mark_only_vig_as_trainable(self.backbone)

    def _intervene_layer(self, sequential, i):
        channels = sequential[0].conv1.in_channels
        layer = Grapher(channels, self.num_knn[i], 1, self.conv, self.act,
                        self.norm, self.bias, self.use_stochastic, self.epsilon, 1)
        sequential = nn.Sequential(layer, sequential)
        return sequential

    def _is_vig(self, name):
        for vig_name in ['fc1', 'fc2', 'graph_conv']:
            if vig_name in name:
                return True
        return False

    def _mark_only_vig_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not self._is_vig(n):
                p.requires_grad = False
            else:
                p.requires_grad = True
        """
        name examples
        ['conv1.weight', 'bn1.weight', 'bn1.bias'
        
        , 'layer1.0.fc1.0.weight', 'layer1.0.fc1.0.bias', 'layer1.0.fc1.1.weight', 'layer1.0.fc1.1.bias', 'layer1.0.graph_conv.gconv.nn.0.weight', 'layer1.0.graph_conv.gconv.nn.0.bias', 'layer1.  0.graph_conv.gconv.nn.1.weight', 'layer1.0.graph_conv.gconv.nn.1.bias', 'layer1.0.fc2.0.weight', 'layer1.0.fc2.0.bias', 'layer1.0.fc2.1.weight', 'layer1.0.fc2.1.bias'
        
        , 'layer1.1.0.conv1.weight', 'layer1.1.0.bn1.weight', 'layer1.1.0.bn1.bias', 'layer1.1.0.conv2.weight', 'layer1.1.0.bn2.weight', 'layer1.1.0.bn2.bias', 'layer1.1.0.conv3.weight', 'layer1.1.0.bn3.weight', 'layer1.1.0.bn3.bias', 'layer1.1.0.downsample.0.weight', 'layer1.1.0.downsample.1.we ight', 'layer1.1.0.downsample.1.bias', 'layer1.1.1.conv1.weight', 'layer1.1.1.bn1.weight', 'layer1.1.1.bn1.bias', 'layer1.1.1.conv2.weight', 'layer1.1.1.bn2.weight', 'layer1.1.1.bn2.bias', 'layer1.1.1.conv3.weight', 'layer1.1.1 .bn3.weight', 'layer1.1.1.bn3.bias', 'layer1.1.2.conv1.weight', 'layer1.1.2.bn1.weight', 'layer1.1.2.bn1.bias', 'layer1.1.2.conv2.weight', 'layer1.1.2.bn2.weight', 'layer1.1.2.bn2.bias', 'layer1.1.2.conv3.weight', 'layer1.1.2.b n3.weight', 'layer1.1.2.bn3.bias'
        
        , ... , 'fc.weight', 'fc.bias']
        """
