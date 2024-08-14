
import os
import abc
import pickle
from typing import List
from typing import Union

import faiss
import tqdm
import numpy as np
import torch

from trainer.trainer_patchcore import Trainer_PatchCore


class Trainer_Uniformaly(Trainer_PatchCore):
    def initialize_model(self, **kwargs):
        super().initialize_model(**kwargs)
        self.patch_scorer = PatchScorer(0.05, False)
        raise Exception(
            "TODO self.attention_mask module 추가를 위해서는, ViT 에서 attention 가져오는 것을 WideResNet50 에서 구현해야 하므로 일단 STOP 함")
        # self.attention_mask = AttentionMask(self.backbone, self.layers_to_extract_from[-1], patchsize, rollout='sup' in self.backbone.name)

    def _preprocessing_memory_bank(self, _image, _features):
        _attn_mask = self.attention_mask.get_attention(
            _image.cuda())

        # binarize
        _attn_mask = _attn_mask > self.thres

        # apply masking
        _features *= _attn_mask.detach().unsqueeze(-1)

        nonzero_idx = torch.unique(
            _features.nonzero(as_tuple=True)[1])
        _features = _features[:, nonzero_idx,
                              :].reshape(-1, _features.shape[-1])
        return _features

    def _preprocessing_predict(self, _image, image_scores):
        attn_mask = self.attention_mask.get_attention(_image)
        if self.thres > 0.0:
            # attn_mask.shape == [1, 784]
            attn_mask = attn_mask > self.thres
        else:
            pass
        image_scores *= attn_mask.detach().cpu().numpy()
        return image_scores


class PatchScorer:
    def __init__(self, k, return_index):
        self.k = k
        self.return_index = return_index

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)

        # x.shape == [1, 784, 1]

        # k_num = int(round(self.k*x.shape[1])) # NOTE round ONNX 호환 불가로 torch.round 로 변경
        k_num = torch.round(torch.tensor(self.k*x.shape[1])).int().item()
        topk = torch.topk(x, min(x.shape[1], k_num), dim=1)
        x = torch.mean(topk.values, dim=1).reshape(-1)

        if self.return_index:
            if was_numpy:
                return x.numpy(), topk.indices.numpy()
            return x, topk.indices

        if was_numpy:
            return x.numpy()
        return x
