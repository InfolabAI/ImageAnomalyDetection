import torch.nn as nn
import torch
import sys

from loguru import logger
from vig_pytorch.vig import vig_224_gelu
from trainer.trainer_patchcore import Trainer_PatchCore
from simplenet import Projection
from trainer.vig_wrapper import VIG_wrapper


class Trainer_Ours_Attention(Trainer_PatchCore):
    # NOTE ViG 가 attention 을 생성하고, 이를 적용한 feature 로 memory bank 하는 모듈
    def initialize_model(self, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs):
        super().initialize_model()
        model, n_filters = vig_224_gelu(self.args)
        self.discriminator = VIG_WRAPPER_ATT(model, self.args)
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

    def _pretrain_model(self, training_data, val_data, test_data, dataset_name):
        self._meta_train(training_data, val_data, test_data, dataset_name)

    def _train_discriminator(self, input_data):
        ret = super()._train_discriminator(input_data)
        self._fill_memory_bank(input_data)
        return ret

    def _fill_memory_bank(self, input_data):
        super().initialize_model()  # anomaly scorer model 초기화
        super()._fill_memory_bank(input_data)

    def _preprocessing_memory_bank(self, _image, _features):
        return self._preprocessing_features_predict(_features)

    def _preprocessing_features_predict(self, _features):
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            _features = self.pre_projection(_features)
            _features = self.discriminator(_features)
        return _features

    def _preprocessing_train_disc(self, features):
        self.B, self.C, self.H, self.W = features.shape
        features = features.reshape(self.B, -1)
        return features

    def _postprocessing_train_disc(self, true_feats, fake_feats):
        true_feats = true_feats.reshape(self.B, self.C, self.H, self.W)
        fake_feats = fake_feats.reshape(self.B, self.C, self.H, self.W)
        return super()._postprocessing_train_disc(true_feats, fake_feats)

    def _loss_function(self, input_, true_feats_size, fake_feats_size):
        return self._loss_att(input_, true_feats_size, fake_feats_size), torch.zeros(1), torch.zeros(1)

    def _loss_att(self, input_, true_feats_size, fake_feats_size):
        ret = self.discriminator(input_)
        true_feat_att = ret[:true_feats_size]
        fake_feat_att = ret[fake_feats_size:]

        t1, t2, f1, f2 = self.sampler(true_feat_att, fake_feat_att)
        # get loss
        # NOTE reduction 이 sum 인데 sampler 에서 sample_ratio 로 추출한만큼 loss 가 작아졌을테니, 다시 증폭함
        amp = 1/self.args.sample_ratio
        true_loss = torch.nn.functional.mse_loss(t1, t2, reduction='sum') * amp
        fake_loss = torch.nn.functional.mse_loss(f1, f2, reduction='sum') * amp
        true_fake_loss = torch.nn.functional.mse_loss(
            t1, f1, reduction='sum') + torch.nn.functional.mse_loss(t2, f2, reduction='sum')
        true_fake_loss = true_fake_loss * amp / 2

        return self.args.lambda_t * true_loss + self.args.lambda_f * fake_loss + self.args.lambda_tf * true_fake_loss

    def sampler(self, true_feat_att, fake_feat_att):
        true_len = len(true_feat_att)
        true_sample_len = int(true_len * self.args.sample_ratio)
        fake_len = len(fake_feat_att)
        fake_sample_len = int(fake_len * self.args.sample_ratio)

        # get ids for sampling NOTE sample_ratio 가 0.5 이하라는 가정이 있음
        # sample true
        true_ids = torch.randperm(true_len)
        true_ids1, true_ids2 = true_ids[:true_sample_len], true_ids[-true_sample_len:]
        # convert ids1, ids2 into bool tensor
        true_ids_bool1, true_ids_bool2 = torch.zeros(
            true_len, dtype=torch.bool), torch.zeros(true_len, dtype=torch.bool)
        true_ids_bool1[true_ids1] = True
        true_ids_bool2[true_ids2] = True

        # sample fake
        fake_ids = torch.randperm(fake_len)
        fake_ids1, fake_ids2 = fake_ids[:fake_sample_len], fake_ids[-fake_sample_len:]
        # convert ids1, ids2 into bool tensor
        fake_ids_bool1, fake_ids_bool2 = torch.zeros(
            fake_len, dtype=torch.bool), torch.zeros(fake_len, dtype=torch.bool)
        fake_ids_bool1[fake_ids1] = True
        fake_ids_bool2[fake_ids2] = True

        t1, t2, f1, f2 = true_feat_att[true_ids_bool1], true_feat_att[
            true_ids_bool2], fake_feat_att[fake_ids_bool1], fake_feat_att[fake_ids_bool2]
        return t1, t2, f1, f2


class VIG_WRAPPER_ATT(VIG_wrapper):
    def _define_prediction(self):
        self.model.prediction.append(
            nn.Conv2d(1024, self.model.prediction[0].in_channels, 1, bias=True))
        self.model.prediction.append(nn.Softmax(dim=1))

    def _post_processing(self, x, orig_x):
        # [B, ch, 1, 1]
        ret = torch.nn.functional.adaptive_avg_pool2d(orig_x, 1) * x
        # [B, ch, 1, 1] => [B, ch]
        return ret.squeeze(3).squeeze(2)
