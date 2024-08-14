import torch
import torch.nn as nn
from trainer.trainer_ours_score import Trainer_Ours_Score
from trainer.vig_wrapper import VIG_wrapper, common_process


class Trainer_Ours_Score_PatchLevel(Trainer_Ours_Score):
    def _embed(self, images, evaluation=False, original_feature_list=False):
        """Returns feature embeddings for images."""

        B = len(images)  # batch size
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()  # wideresnet
            features = self.forward_modules["feature_aggregator"](
                images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer]
                    for layer in self.layers_to_extract_from]  # layer 2, 3 to list

        # i==0 feat.shape == torch.Size([5 (batch), 512, 36, 36])
        # i==1 feat.shape == torch.Size([5, 1024, 18, 18])
        H, W = features[0].shape[-2], features[0].shape[-1]
        for i, feat in enumerate(features):
            if i != 0:
                features[i] = torch.nn.functional.interpolate(
                    features[i], size=(H, W), mode='bilinear')

        if self.args.vig_backbone_pooling:
            ret = self._channel_pooling(features)
        else:
            ret = torch.concat(features, dim=1)

        # NOTE ret.shape == torch.Size([1, 1536, 36, 36])
        return ret, (1, 0)

    def _get_len_true_fake(self, input_):
        H, W = input_.shape[-2], input_.shape[-1]
        return H*W*len(input_)//2, H*W*len(input_)//2

    def _wrap(self, model, args):
        return VIG_wrapper_score(model, args)


class VIG_wrapper_score(VIG_wrapper):
    def _define_prediction(self):
        self.model.prediction.append(nn.Linear(1024, 1, bias=True))

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        return x.reshape(-1)

    def _preprocess(self, x):
        """
        (Pdb) p x.shape
            torch.Size([2, 64, 36, 36])
        (Pdb) torch.nn.Unfold(1, 1)(x).shape
            torch.Size([2, 64, 1296])
        (Pdb) torch.nn.Unfold(1, 1)(x).reshape(2, 64, 1, 1, -1).shape
            torch.Size([2, 64, 1, 1, 1296])
        (Pdb) torch.nn.Unfold(1, 1)(x).reshape(2, 64, 1, 1, -1).permute(0, 4, 1, 2, 3).shape
            torch.Size([2, 1296, 64, 1, 1])
        (Pdb) torch.nn.Unfold(1, 1)(x).reshape(2, 64, 1, 1, -1).permute(0, 4, 1, 2, 3).reshape(-1, 64,1,1).shape
            torch.Size([2592, 64, 1, 1])
        """

        unfolded_x = torch.nn.Unfold(1, 1)(x)
        unfolded_x = unfolded_x.reshape(*unfolded_x.shape[:2], 1, 1, -1)
        unfolded_x = unfolded_x.permute(0, 4, 1, 2, 3)
        unfolded_x = unfolded_x.reshape(-1, *unfolded_x.shape[-3:])
        return unfolded_x

    def forward(self, x):
        x, orig_x = common_process(
            x, self.model.backbone, self.model.prediction, self._patch_to_batch, self._batch_to_patch, preprocess_func=self._preprocess)
        x = self.model.last(x)

        ret = self._post_processing(x, orig_x)
        return ret
