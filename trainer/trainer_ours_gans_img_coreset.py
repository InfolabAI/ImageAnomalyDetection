# NOTE 성능이 안 좋아서 DEPRECATED
import torch
import numpy as np
import json
import os
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import sys
from torchsampler import ImbalancedDatasetSampler
from vig_pytorch.vig import vig_224_gelu
from loguru import logger
from trainer.trainer_ours_gans_img import Trainer_Ours_GANs_IMG, CustomDataset
from trainer.trainer_patchcore import NearestNeighbourScorer, ApproximateGreedyCoresetSampler, FaissNN
from trainer.vig_wrapper import VIG_wrapper
from simplenet import Projection
from models.gans_dcgans import ManageGANs
from torchvision.utils import save_image


class Trainer_Ours_GANs_IMG_Coreset(Trainer_Ours_GANs_IMG):
    # NOTE 현재 성능이 잘 안나오는 버전
    def initialize_model(self, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs):
        super().initialize_model(pre_proj, proj_layer_type, meta_epochs,
                                 aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs)
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=1, nn_method=FaissNN(False, 8))
        self.featuresampler = ApproximateGreedyCoresetSampler(
            self.args.gan_coreset_p, self.device)  # NOTE coreset ratio 에 따라, VIG 의 num_classes 가 결정됨
        self.custom_training_data = None

    def _modify_vig(self, num_classes):
        self.num_classes = num_classes
        if self.discriminator.model.last.out_features != num_classes:
            # self.discriminator.model.prediction[-1] = nn.Conv2d(1024,
            #                                                    num_classes, 1, bias=True).to(self.device)
            self.discriminator.model.last = nn.Linear(
                1024, num_classes, bias=True).to(self.device)

            self.dsc_opt.add_param_group(
                {'params': self.discriminator.model.last.parameters()})  # NOTE 제대로 동작하는지 확인 필요

    def _save_vig_modified_pred(self):
        """
        optim.step() 전에 call
        """
        self.check_weight = self.discriminator.model.prediction[-1].weight.data.mean(
        )

    def _check_vig_modified_pred(self):
        """
        optim.step() 후에 call
        NOTE 현재 weight 바뀌는 것 확인했고, scheduler 는 기본으로 꺼져있어서 확인할 필요가 없음을 확인했음
        """
        # NOTE update 전 후 weight 가 바뀌었는지 확인. 바뀌어야 함.
        cond1 = self.discriminator.model.prediction[-1].weight.data.mean(
        ) != self.check_weight
        # NOTE scheduler step 후, param group 간 lr 에 차이가 없는지 확인. 차이가 없어야 함. 모든 lr 이 같으면 mean 뺐을 때, 0 이어야 함.
        aa = np.array([param_group['lr']
                      for param_group in self.dsc_opt.param_groups])
        cond2 = (aa - aa.mean()).sum() == 0

        # NOTE 둘 다 True 이어야 함
        return cond1 and cond2

    def _fill_memory_bank(self, input_data):
        """
        self.pre_projection 은 train_discriminator 에서 함께 학습하므로, 여기서는 통과하지 않음
        """
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)[0]

        features_reshape = []
        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                feat = _image_to_features(image)
                features_reshape.append(feat.reshape(feat.shape[0], -1).cpu())
                features.append(feat.cpu())

        features = np.concatenate(features, axis=0)
        features_reshape = np.concatenate(features_reshape, axis=0)
        features_memory = self.featuresampler.run(features_reshape)
        self._modify_vig(features_memory.shape[0])

        self.anomaly_scorer.fit(detection_features=[features_memory])

        # get target
        target = self.anomaly_scorer.predict(
            [features_reshape])[-1].reshape(-1)  # 마지막이 가장 가까운 vector 의 idx 라 -1 임

        """
        예를 들어, features:[#patches, channel, H, W] ,features_reshape: [#patches, #features], target: [#patches]
        """

        return features, target

    def _loss_function(self, input_, gan_x, target):
        gan_x = gan_x.to(self.device)
        return self.update_GDC(input_, gan_x, target)

    def _postprocessing_train_disc(self, true_feats):
        return true_feats

    def _pre_train_discriminator(self, training_data):
        if self.custom_training_data is None:
            inputs, targets = self._fill_memory_bank(training_data)
            custom_dataset = CustomDataset(inputs, targets)
            self.custom_training_data = torch.utils.data.DataLoader(
                custom_dataset, sampler=ImbalancedDatasetSampler(custom_dataset), batch_size=500, num_workers=1)
            # custom_dataset, shuffle=True, batch_size=500)  # , num_workers=1)

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        self.nG.train()
        self.nD.train()
        logger.info(f"Training discriminator...")
        self._pre_train_discriminator(input_data)
        input_data.dataset.gan_mode = True
        all_loss = []
        all_output = []
        all_target = []
        with tqdm.tqdm(total=self.gan_epochs, position=0) as pbar:
            while pbar.n < self.gan_epochs:
                for img, target in tqdm.tqdm(self.custom_training_data, desc="inner loop", position=1, leave=False):
                    if pbar.n >= self.gan_epochs:
                        break

                    self.dsc_opt.zero_grad()
                    self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    true_feats = img.to(torch.float).to(self.device)
                    target = target.to(self.device)
                    true_feats = self.pre_projection(
                        true_feats)  # feature adapter

                    # generator
                    true_feats = self._preprocessing_train_disc(true_feats)

                    input_ = self._postprocessing_train_disc(
                        true_feats)

                    loss, output = self._loss_function(
                        input_, next(iter(input_data))['image_gans'], target)

                    self.proj_opt.step()

                    all_loss.append(loss)
                    all_output.append(output)
                    all_target.append(target)

                    if self.cos_lr:
                        self.dsc_schl.step()
                    # bool_ = self._check_vig_modified_pred()
                    # get accuracy
                    all_output_cat = torch.cat(all_output, dim=0)
                    all_target_cat = torch.cat(all_target, dim=0)
                    acc = 100 * (
                        (all_output_cat.argmax(1) ==
                         all_target_cat).sum().item() / len(all_target_cat))

                    all_loss_sum = sum(all_loss) / len(input_data)
                    cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                    pbar_str = f"loss:{round(all_loss_sum, 5)} "
                    pbar_str += f"lr:{round(cur_lr, 6)} "
                    pbar_str += f"acc:{round(acc, 2)}% "
                    pbar_str += f"errD: {self.errD}, errG: {self.errG}"
                    pbar.set_description_str(pbar_str)
                    pbar.update(1)

        input_data.dataset.gan_mode = False
        self.errD, self.errG = 0, 0
        return round(all_loss_sum, 5)

    def _wrap(self, model, args):
        return VIG_wrapper_score(model, args)


class VIG_wrapper_score(VIG_wrapper):
    def _define_prediction(self):
        self.model.prediction.append(
            nn.Linear(1024, 2, bias=True))  # OK, noisedOK

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        return x.reshape(x.shape[0], -1)
