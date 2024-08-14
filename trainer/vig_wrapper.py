import torch
import torch.nn as nn
import sys
from loguru import logger


def common_process(x, backbone, prediction, patch_to_batch, batch_to_patch, midprocess_func=None, preprocess_func=None):
    """
    last layer 만 제외할 수 있도록 함"""
    cur_level = logger._core.min_level
    orig_x = x.clone()

    x, H_W = batch_to_patch(x)
    for i in range(len(backbone)):
        x = backbone[i](x)
        if midprocess_func is not None:
            if i != len(backbone) - 1:
                x = midprocess_func(x)
        logger.remove()
    x = patch_to_batch(x, H_W)

    logger.add(sys.stdout, level="INFO")
    if preprocess_func is not None:
        x = preprocess_func(x)
    # [B, ch, H, W] => [B, ch, 1, 1]
    x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    x = prediction(x)
    x = x.squeeze(3).squeeze(2)

    return x, orig_x


class VIG_wrapper(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        # delete attribute from self.model
        self._remove_stem()
        del self.model.prediction[-1]

        # backbone layer 수 조절
        for i in list(range(len(self.model.backbone)))[::-1]:
            if i >= args.n_blocks:
                del self.model.backbone[i]

        # backbone 내 FFN 제거
        # for i in range(len(self.model.backbone)):
        #    del self.model.backbone[i][-1]

        self._define_prediction()
        self._move_pred_to_last()

    def _remove_stem(self):
        del self.model.stem

    def _define_prediction(self):
        raise NotImplementedError

    def _post_processing(self, x, orig_x):
        raise NotImplementedError

    def _move_pred_to_last(self):
        self.model.last = self.model.prediction[-1]
        self.model.prediction = self.model.prediction[:-1]

    def forward(self, x):
        x, orig_x = common_process(
            x, self.model.backbone, self.model.prediction, self._patch_to_batch, self._batch_to_patch)
        x = self.model.last(x)

        ret = self._post_processing(x, orig_x)
        return ret

    def _batch_to_patch(self, x):
        if not self.training:
            return x, 0

        if not self.args.true_false_edge:
            return x, 0

        # NOTE 절반은 true, 절반은 false 라는 가정하에 짜여짐
        split_len = x.shape[0]//2
        H_W = x.shape[2]
        # [B, C, H, W] => [B//2, C, H*2, W]
        out = torch.concat([x[:split_len], x[split_len:]], dim=2)

        # assert (x[0, 0, 0, 0] + x[split_len, 0, 0, 0]) == (out[0, 0, 0, 0] + out[0, 0, H_W, 0])  # NOTE assert 결과 문제없음
        return out, H_W

    def _patch_to_batch(self, x, H_W):
        if not self.training:
            return x

        if not self.args.true_false_edge:
            return x

        split_len = x.shape[0]
        # [B, C, H, W] => [B//2, C, H*2, W]
        out = torch.concat([x[:, :, :H_W], x[:, :, H_W:]], dim=0)

        # assert (x[0, 0, 0, 0] + x[0, 0, H_W, 0]) == (out[0, 0, 0, 0] + out[split_len, 0, 0, 0])  # NOTE assert 결과 문제없음
        return out


class ExceptLast(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, x):
        return common_process(x, self.wrapper.model.backbone, self.wrapper.model.prediction, self.wrapper._patch_to_batch, self.wrapper._batch_to_patch)[0]
