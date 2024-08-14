from trainer.trainer_patchcore import Trainer_PatchCore
from vig_pytorch_pretrained.vig import vig_224_gelu
import torch
import common


class Trainer_GraphCore(Trainer_PatchCore):
    def set_backbone(self, backbone, device):
        self.backbone = vig_224_gelu("b").to(device)
        self.backbone.name, self.backbone.seed = "vision_gnn", 0
        self.backbone.eval()

    def _embed(self, images):
        """Returns feature embeddings for images."""

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        # [batch_size, #tokens, #features]
        features = [features[layer] for layer in self.layers_to_extract_from]
        """ 2개 layers 에서 가져왔을 때, ti model 일때,
        features[0].shape == torch.Size([8, 192, 14, 14]) == [batch_size, #features, patch_H, patch_W]
        features[1].shape == torch.Size([8, 192, 14, 14])
        ...
        """

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        """ 2개 layers 에서 가져왔을 때
        features[0][0].shape == torch.Size([8, 196, 192, 3, 3])
        features[1][0].shape == torch.Size([8, 196, 192, 3, 3])
        ...
        """

        patch_shapes = [x[1] for x in features]

        features = [x.reshape(-1, *x.shape[-3:])
                    for x, patch_shape in features]
        """ 2개 layers 에서 가져왔을 때
        features[0].shape == torch.Size([1568, 192, 3, 3])
        features[1].shape == torch.Size([1568, 192, 3, 3])
        """

        # avgpooling 정해진 dim 1536 (layer 2 의 channel 수 + layer 3 의 channel 수)
        features = self.forward_modules["preprocessing"](features)
        """ features.shape == torch.Size([1568, 2, 1536]) """

        features = self.forward_modules["preadapt_aggregator"](
            features)  # further pooling
        """ features.shape == torch.Size([1568, 1536]) """

        return features, patch_shapes
