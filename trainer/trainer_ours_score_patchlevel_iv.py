import torch
import torch.nn as nn
import sys
import math
from vig_pytorch.vig import vig_224_gelu
from loguru import logger
from trainer.trainer_ours_score_patchlevel import Trainer_Ours_Score_PatchLevel
from trainer.trainer_ours_score_iv import Trainer_Ours_Score_Iv
from trainer.vig_wrapper import VIG_wrapper, common_process
from simplenet import Projection


class Trainer_Ours_Score_PatchLevel_Iv(Trainer_Ours_Score_PatchLevel, Trainer_Ours_Score_Iv):
    def _wrap(self, model, args):
        return VIG_wrapper_score_patchlevel_iv(model, args)


class VIG_wrapper_score_patchlevel_iv(VIG_wrapper):
    def _define_prediction(self):
        breakpoint()
        self.model.prediction.append(nn.Linear(1024, 1, bias=True))

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

    def _midprocess(self, x):
        # ivv_X 만 다음 layer 로 전달, 마지막 layer 에서는 midprocess 를 call 하지 않도록 짜여져 있음
        return x[:x.shape[0]//3]

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        B = x.shape[0]//3
        ivv_x, iv_x, v_x = x[:B], x[B:B*2], x[B*2:]
        return ivv_x, iv_x, v_x

    def forward(self, x):
        x, orig_x = common_process(
            x, self.model.backbone, self.model.prediction, self._patch_to_batch, self._batch_to_patch, preprocess_func=self._preprocess, midprocess_func=self._midprocess)
        x = self.model.last(x)

        ret = self._post_processing(x, orig_x)
        return ret
