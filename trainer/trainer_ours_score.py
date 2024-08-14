import torch
import torch.nn as nn
import sys
from vig_pytorch.vig import vig_224_gelu
from loguru import logger
from trainer.trainer_simplenet import Trainer_SimpleNet
from trainer.vig_wrapper import VIG_wrapper
from simplenet import Projection


class Trainer_Ours_Score(Trainer_SimpleNet):
    def _get_vig(self, args):
        model, n_filters = vig_224_gelu(args)
        return model, n_filters

    def initialize_model(self, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs):
        model, n_filters = self._get_vig(self.args)
        self.discriminator = self._wrap(model, self.args)
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension, n_filters, pre_proj, proj_layer_type, conv=True)

        self.discriminator.to(self.device)
        # self.elapsed_time_test(self.discriminator)
        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin = dsc_margin
        if self.pre_proj > 0:
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(
                self.pre_projection.parameters(), lr*.1)

    def _wrap(self, model, args):
        return VIG_wrapper_score(model, args)

    def _preprocessing_train_disc(self, features):
        self.B, self.C, self.H, self.W = features.shape
        features = features.reshape(self.B, -1)
        return features

    def _postprocessing_train_disc(self, true_feats, fake_feats):
        true_feats = true_feats.reshape(self.B, self.C, self.H, self.W)
        fake_feats = fake_feats.reshape(self.B, self.C, self.H, self.W)
        return super()._postprocessing_train_disc(true_feats, fake_feats)


class VIG_wrapper_score(VIG_wrapper):
    def _define_prediction(self):
        self.model.prediction.append(nn.Linear(1024, 1, bias=True))

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        return x.reshape(-1)
