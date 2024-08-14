import torch.nn as nn
import torch
import sys

from loguru import logger
from trainer.trainer_ours_score import Trainer_Ours_Score
from trainer.vig_wrapper import VIG_wrapper


class Trainer_Ours_Attention_Score(Trainer_Ours_Score):
    # NOTE ViG 가 attention 을 생성하고, 이를 적용한 feature 로 바로 score 로 생성하는 모듈
    def _wrap(self, model, args):
        return VIG_WRAPPER_ATT(model, args)


class VIG_WRAPPER_ATT(VIG_wrapper):
    def _define_prediction(self):
        in_channels = self.model.prediction[0].in_channels
        self.model.prediction.append(
            nn.Conv2d(1024, in_channels, 1, bias=True))
        self.model.prediction.append(nn.Softmax(dim=1))

        self.last = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.BatchNorm1d(in_channels//2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_channels//2, in_channels//4),
            nn.BatchNorm1d(in_channels//4),
            nn.LeakyReLU(0.2),
            nn.Linear(in_channels//4, 1, bias=True)
        )

    def _post_processing(self, x, orig_x):
        # [B, ch, 1, 1]
        ret = torch.nn.functional.adaptive_avg_pool2d(orig_x, 1) * x
        # [B, ch, 1, 1] => [B, ch]
        ret = ret.squeeze(3).squeeze(2)
        # [B, ch, 1, 1] => [B, ch] => [B, 1] => [B]
        ret = self.last(ret).squeeze(1)
        return ret
