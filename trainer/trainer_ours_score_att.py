import torch
import torch.nn as nn
import sys
from vig_pytorch_att.vig import vig_224_gelu
from loguru import logger
from trainer.trainer_ours_score import Trainer_Ours_Score
from trainer.vig_wrapper import VIG_wrapper
from simplenet import Projection


class Trainer_Ours_Score_Att(Trainer_Ours_Score):
    # NOTE ViG 내부에 edge 선택 전 att 를 도입해서 invariant feature 에 대해 edge 를 생성하고, invariant feature 로 GNN 이 작동하도록 도움
    def initialize_model(self, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs):
        # NOTE vig_224_gelu 의 source 가 바뀜
        model, n_filters = vig_224_gelu(self.args)
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
