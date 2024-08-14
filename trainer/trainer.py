import os
import shutil
import pickle
import wandb
import PIL
import re
from loguru import logger

import math
import numpy as np
import torch
import sys
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import common
import metrics
from utils import plot_segmentation_images
from vig_pytorch.vig import vig_224_gelu
from trainer.classes import EarlyStopping, ElapsedTimer
import psutil
from simplenet import Discriminator, Projection
import pandas as pd

from intra_class_variance.mmd import MMD
from intra_class_variance.coral import CORAL
from intra_class_variance.mse import MSE
from intra_class_variance.calinski_harabasz_index import CalinskiHarabaszIndex
from intra_class_variance.davies_bouldin_index import DaviesBouldinIndex


class Trainer:
    def __init__(self, device):
        """anomaly detection class."""
        # super(SimpleNet, self).__init__()
        self.device = device

    def set_backbone(self, backbone, device):
        """
        graphcore 에서 다른 backbone 을 사용하기 위함
        """
        # for name, module in self.backbone.named_modules():
        #    print(name)
        self.backbone = backbone.to(device)

    def set_aggregator(self, train_backbone):
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        return feature_aggregator

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,  # 1536
        target_embed_dimension,  # 1536
        patchsize=3,  # 3
        patchstride=1,
        embedding_size=None,  # 256
        meta_epochs=1,  # 40
        aed_meta_epochs=1,
        gan_epochs=1,  # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2,  # 2
        dsc_hidden=None,  # 1024
        dsc_margin=.8,  # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0,  # 1
        proj_layer_type=0,
        onnx=None,
        **kwargs,
    ):
        pid = os.getpid()

        self.onnx = onnx
        self.elapsed_timer = ElapsedTimer()
        self.args = kwargs['args']

        def show_mem():
            return (psutil.Process(pid).memory_info())

        self.set_backbone(backbone, device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = self.set_aggregator(train_backbone)
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        # self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std

        # AED
        self.aed_meta_epochs = aed_meta_epochs
        self.pre_proj = pre_proj
        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1

        params_initialization = {
            "dsc_layers": dsc_layers,
            "dsc_hidden": dsc_hidden,
            "pre_proj": pre_proj,
            "proj_layer_type": proj_layer_type,
            "meta_epochs": meta_epochs,
            "aed_meta_epochs": aed_meta_epochs,
            "gan_epochs": gan_epochs,
            "dsc_margin": dsc_margin,
            "dsc_lr": dsc_lr,
            "lr": lr,
            "device": self.device,
            "args": self.args,
        }
        self.initialize_model(**params_initialization)
        self.set_ea_modules()

    def set_ea_modules(self):
        if self.pre_proj > 0:
            self.ea_modules = [self.pre_projection, self.discriminator]
        else:
            self.ea_modules = [self.discriminator]
        self.ea = EarlyStopping(args=self.args, ea_modules=self.ea_modules, patience=self.args.ea_patience,
                                delta=self.args.ea_delta, warmup=self.args.ea_warmup)

    def initialize_model(self, **kwargs):
        raise NotImplementedError

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

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
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(
                    math.sqrt(L)), C).permute(0, 3, 1, 2)

        if original_feature_list:
            return features

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        """
        featuers[0].shape == torch.Size(
            [5, 1296, 512, 3, 3]) # patch_shapes == [36, 36] from layer 2
        # patch_shapes == [18, 18] from layer 3
        features[1].shape == torch.Size([5, 324, 1024, 3, 3])
        """
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            # layer 간 dim 맞춘 후 합치는 과정
            _features = features[i]
            patch_dims = patch_shapes[i]

            """ _features.shape == torch.Size([5, 324, 1024, 3, 3]) """
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], * _features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            """ _features.shape == torch.Size([5, 1024, 3, 3, 18, 18]) """
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            """ _features.shape == torch.Size([46080, 18, 18]) """
            _features = F.interpolate(_features.unsqueeze(1), size=(
                ref_num_patches[0], ref_num_patches[1]), mode="bilinear", align_corners=False,)
            """_features.shape == torch.Size([46080, 1, 36, 36]) """
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            """_features.shape == torch.Size([5, 36, 36, 1024, 3, 3]) """
            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:])
            """_features.shape == torch.Size([5, 1296, 1024, 3, 3]) """
            features[i] = _features

        """
        features[0].shape == torch.Size([5, 1296, 512, 3, 3])
        features[1].shape == torch.Size([5, 1296, 1024, 3, 3])
        """

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        """
        features[0].shape == torch.Size([6480, 512, 3, 3])
        features[1].shape == torch.Size([6480, 1024, 3, 3])
        """

        if "vig" in self.args.mainmodel:
            if self.args.vig_backbone_pooling:
                ret = self._channel_pooling(features)
            else:
                ret = torch.concat(features, dim=1)
            return ret, patch_shapes

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        # pooling each feature to same channel and stack together

        features = self.forward_modules["preprocessing"](
            features)  # avgpooling 정해진 dim 1536 (layer 2 의 channel 수 + layer 3 의 channel 수)
        """ features.shape == torch.Size([6480, 2, 1536])
        """
        features = self.forward_modules["preadapt_aggregator"](
            features)  # further pooling
        """ feautes.shape == torch.Size([6480, 1536])
        """

        return features, patch_shapes

    def _channel_pooling(self, features):
        C = features[0].shape[1]  # channel
        H = features[0].shape[2]
        W = features[0].shape[3]
        for i, feat in enumerate(features):
            feat = feat.permute(0, 2, 3, 1).reshape(
                feat.shape[0], -1, feat.shape[1])
            feat = torch.nn.functional.adaptive_avg_pool1d(feat, C)
            feat = feat.reshape(feat.shape[0], H, W, C)
            features[i] = feat.permute(0, 3, 1, 2)

        ret = torch.stack(features, dim=1).mean(1)
        return ret

    def _evaluate(self, scores, segmentations, features, labels_gt, masks_gt):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]

        # NOTE 속도를 위해 anomaly map 안 말들고 segmentations 를 [-1, -1, ..] 로 지정함 in _predict()
        if len(masks_gt) > 0 and segmentations[0] != -1:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / \
                    max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)

            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
            # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
                                      norm_segmentations)
        else:
            full_pixel_auroc = -1
            pro = -1

        return auroc, full_pixel_auroc, pro

    def eval_intra_class_variance(self, training_data, val_data, test_data, dataset_name):
        data_loader_dict = {"train": training_data,
                            "val": val_data, "test": test_data}
        df_list = []
        for loader_name, data_loader in tqdm.tqdm(data_loader_dict.items()):
            if data_loader is not None:
                # try:
                df = self._eval_variance(
                    data_loader, dataset_name, loader_name)
                # except:
                if df is not None:
                    df_list.append(df)
                else:
                    logger.critical(
                        f"Error in dataset_name, {dataset_name}, {loader_name}, len_data_loader: {len(data_loader)}")

        return pd.concat(df_list)

    def _eval_variance(self, dataloader, dataset_name, loader_name):
        _ = self.forward_modules.eval()

        img_features_list = []
        patch_features_list = []
        img_labels_gt = []
        mm = common.MeanMapper(1536)

        # feature 추출
        for data in dataloader:
            if isinstance(data, dict):
                image = data["image"].to(torch.float).to(self.device)
            with torch.no_grad():
                patch_level_features, patch_shapes = self._embed(
                    image, evaluation=True)
                patch_level_features = patch_level_features.reshape(
                    image.shape[0], -1, patch_level_features.shape[-1])
                image_level_features = self._embed(
                    image, evaluation=True, original_feature_list=True)
                image_level_features = torch.stack(
                    [mm(feat) for feat in image_level_features]).mean(0)

            img_labels_gt.extend(data["is_anomaly"].numpy().tolist())

            img_features_list.append(image_level_features.cpu())
            patch_features_list.append(patch_level_features.cpu())

        img_features = torch.cat(img_features_list, dim=0)
        patch_features = torch.cat(patch_features_list, dim=0)
        img_labels = np.array(img_labels_gt)
        # NOTE patch_labels 사용 안함, 이유는 patch-level 도 같은 위치 patch 끼리 image-level 처럼 batch 단위로 비교하기 때문
        # patch_labels = np.repeat( np.array(img_labels_gt), patch_shapes[0][0]*patch_shapes[0][1]).tolist()  # 각 이미지의 patch 수만큼 label 을 반복. NOTE patch_shapes[0] == [36, 36] 에 맞춰져 있기 때문

        metric_class_list = [MMD, CORAL, MSE,
                             CalinskiHarabaszIndex, DaviesBouldinIndex]

        if img_features.shape[0] <= 1:
            return None

        df_list = []
        for metric_class in metric_class_list:
            # patch-level
            metric = metric_class(device=self.device,
                                  features=patch_features, labels=img_labels)
            df = metric.eval_variance(
                "patch-level", dataset_name, loader_name)
            df_list.append(df)

            # image-level
            metric = metric_class(device=self.device,
                                  features=img_features, labels=img_labels)
            df = metric.eval_variance(
                "image-level", dataset_name, loader_name)
            df_list.append(df)

        df = pd.concat(df_list)

        return df

    def train(self, training_data, val_data, test_data, dataset_name):
        return self._meta_train(training_data, val_data, test_data, dataset_name)

    def _meta_train(self, training_data, val_data, test_data, dataset_name):

        self.dataset_name = dataset_name
        state_dict = {}
        ckpt_path = None
        # NOTE saved model 을 불러와서 infer 하는 부분. 나중에 SimpleNet 에서 가져오자

        best_record = None
        self._pre_meta_train(training_data, val_data, test_data, dataset_name)

        for i_mepoch in range(self.meta_epochs):
            logger.info(f"\n\n----- {i_mepoch} -----")

            loss = self._train_discriminator(training_data)

            self._additional_process(
                training_data, val_data, test_data, dataset_name, i_mepoch)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt = self.predict(
                val_data)
            auroc, full_pixel_auroc, pro = self._evaluate(
                scores, segmentations, features, labels_gt, masks_gt)

            logger.info(f"----- {i_mepoch} VAL I-AUROC:{round(auroc, 4)}")
            if self.args.wan:
                wandb.log(
                    {f"{dataset_name}_val_auroc": auroc, f"{dataset_name}_loss": loss})

            self.ea(auroc, i_mepoch)
            if self.ea.early_stop:
                break  # NOTE 중요. early stopping 해야 함.

        if ckpt_path is not None:
            torch.save(state_dict, ckpt_path)

        scores, segmentations, features, labels_gt, masks_gt = self.predict(
            test_data)
        self._save_fault_images(test_data, scores)
        auroc, full_pixel_auroc, pro = self._evaluate(
            scores, segmentations, features, labels_gt, masks_gt)

        best_record = [auroc, full_pixel_auroc,
                       pro, segmentations, labels_gt, scores]

        return best_record

    def _pre_meta_train(self, training_data, val_data, test_data, dataset_name):
        """
        trainer_ours_gans_img 를 위한 분리"""
        pass

    def _additional_process(self, training_data, val_data, test_data, dataset_name, i_mepoch):
        """
        trainer_ours_gans_img 를 위한 분리"""
        pass

    def _get_len_true_fake(self, input_):
        """
        Trainer_Ours_Score_PatchLevel 에서는 중간에 patch 들을 batch 로 변환하기 때문에 //2 * H * W 가 필요함
        """

        return len(input_)//2, len(input_)//2

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        logger.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    out = self._preprocessing_image(data_item)

                    # generator
                    """
                    (Pdb) p self.mix_noise
                        1
                    (Pdb) p self.noise_std
                        0.015
                    (Pdb) p true_feats.shape
                        torch.Size([10368, 1536])
                    (Pdb) p noise_idxs.shape
                        torch.Size([10368])
                    (Pdb) p noise_idxs
                        tensor([0, 0, 0,  ..., 0, 0, 0]) 모두 0
                    (Pdb) p noise_one_hot NOTE 각 class 에 대해 one-hot 이므로, mix_noise 가 2면, noise_idxs 가 모두 0 이므로 [[1, 0], ..., [1, 0]] 이 됨
                        tensor([[1],
                                [1],
                                ...,
                                [1]], device='cuda:3') 모두 1
                    (Pdb) p noise_one_hot.shape
                        torch.Size([10368, 1])
                    (Pdb) p noise
                        tensor([[[-0.0109,  0.0233,  0.0140,  ..., -0.0098,  0.0228,  0.0128]],
                                ...,
                                [[ 0.0067, -0.0105,  0.0124,  ..., -0.0389,  0.0042, -0.0099]]],
                            device='cuda:3')
                    (Pdb) p noise.shape
                        torch.Size([10368, 1, 1536])
                    (Pdb) p noise.shape NOTE (noise * noise_one_hot.unsqueeze(-1)).sum(1) 이거 이후
                        torch.Size([10368, 1536])
                    (Pdb) p noise NOTE 어차피 1을 곱한 것이라 상관 없음
                        tensor([[-0.0109,  0.0233,  0.0140,  ..., -0.0098,  0.0228,  0.0128],
                                ...,
                                [ 0.0067, -0.0105,  0.0124,  ..., -0.0389,  0.0042, -0.0099]],
                            device='cuda:3')
                    """
                    out = self._preprocessing_train_disc(out)

                    true_feats, fake_feats = self.apply_augment(out)
                    input_ = self._postprocessing_train_disc(
                        true_feats, fake_feats)

                    len_true, len_fake = self._get_len_true_fake(input_)
                    loss, p_true, p_fake = self._loss_function(
                        input_, len_true, len_fake)

                    self._loss_backward(loss)
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()

                    self.dsc_opt.step()

                    self._loss_aggregate(all_loss, loss)
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(
                        embeddings_list).std(0).mean(-1)

                if self.cos_lr:
                    self.dsc_schl.step()

                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{self._loss_str( all_loss)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

        return self._loss_str(all_loss, ret=True)

    def _loss_backward(self, loss):
        # 0번 빼고 나머지는 print 를 위한값일때를 위한 function
        if isinstance(loss, list):
            loss[0].backward()
        else:
            loss.backward()

    def _loss_aggregate(self, all_loss, loss):
        # 0번 빼고 나머지는 print 를 위한값일때를 위한 function
        if isinstance(loss, list):
            all_loss.append([l.detach().cpu().item() for l in loss])
        else:
            all_loss.append(loss.detach().cpu().item())

    def _loss_str(self, all_loss, ret=False):
        """
        0번 빼고 나머지는 print 를 위한값일때를 위한 function
        """
        if isinstance(all_loss[0], list):
            total = [0] * len(all_loss[0])
            for loss in all_loss:
                for i, l in enumerate(loss):
                    total[i] += l
            total = [round(t / len(all_loss), 5) for t in total]
            if ret:
                # wandb 를 위한 return 용 이므로, loss[0] 의 값만 반환
                total = total[0]
        else:
            total = round(sum(all_loss) / len(all_loss), 5)

        return total

    def _preprocessing_image(self, dict_):
        img = dict_["image"]
        img = img.to(torch.float).to(self.device)
        true_feats = self._embed(img, evaluation=False)[0]
        if self.pre_proj > 0:
            true_feats = self.pre_projection(
                true_feats)  # feature adapter

        return true_feats

    def _loss_function(self, input_, true_feats_size, fake_feats_size):
        return self._loss_score(input_, true_feats_size, fake_feats_size)

    def _preprocessing_train_disc(self, features):
        raise NotImplementedError

    def _postprocessing_train_disc(self, true_feats, fake_feats):
        input_ = torch.cat([true_feats, fake_feats])
        return input_

    def apply_augment(self, true_feats):
        noise_idxs = torch.randint(
            0, self.mix_noise, torch.Size([true_feats.shape[0]]))  # mix_noise 중에 무엇을 사용할 것인지 선택하는 index
        noise_one_hot = torch.nn.functional.one_hot(
            noise_idxs, num_classes=self.mix_noise).to(self.device)  # (N, K)
        noise = torch.stack([
            torch.normal(0, self.noise_std * 1.1 **
                         (k), true_feats.shape)
            for k in range(self.mix_noise)], dim=1).to(self.device)  # (N, K, C)

        # 여기서 mix_noise 중에 일부만 선택되고 나머지는 0이 됨. sum 을 통해 선택된 mix_noise 만 남음
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        fake_feats = true_feats + noise  # 생성된 feature
        return true_feats, fake_feats

    def _loss_score(self, input_, true_feats_size: int, fake_feats_size: int):
        # image 당 하나의 score 생성
        scores = self.discriminator(input_)

        true_scores = scores[:true_feats_size]
        fake_scores = scores[fake_feats_size:]

        # true 와 false images 가 th 보다 높거나 낮도록 학습. 높아야 OK. th==0.5
        th = self.dsc_margin
        p_true = (true_scores.detach() >= th).sum() / \
            len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / \
            len(fake_scores)
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)

        loss = true_loss.mean() + fake_loss.mean()
        return loss, p_true, p_fake

    def _predict(self, image):
        raise NotImplementedError

    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        img_paths = []
        scores = []
        patch_scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                self._preprocessing_predict_dataloader(data)
                _scores, _masks, _feats, _patch_scores = self._predict(image)
                for score, mask, feat, ps, is_anomaly in zip(_scores, _masks, _feats, _patch_scores, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)
                    patch_scores.append(ps)

        # save image_path and patch_scores and args(patchsize)
        if self.args.save_patch_scores and dataloader.dataset.split.value == 'val':
            # normalize patch_scores
            patch_scores = np.stack(patch_scores)
            patch_scores = (patch_scores - patch_scores.min()) / \
                (patch_scores.max() - patch_scores.min())

            with open(f'./output_labels/{self.args.mainmodel}_{dataloader.dataset.classnames_to_use}_{self.args.patchsize}_{dataloader.dataset.split.value}_patch_scores.pkl', 'wb') as f:
                pickle.dump([img_paths, patch_scores, self.args.patchsize], f)

        return scores, masks, features, labels_gt, masks_gt

    def _preprocessing_predict_dataloader(self, data):
        """
        Trainer_Ours_Score_PatchLevel_Interpretable 에서 image_path 위에 explain 을 출력하기 위함
        """
        pass

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def _save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

    def _save_fault_images(self, data, scores):
        if not self.args.save_fault_images:
            return

        assert 'good' in [
            d[1] for d in data.dataset.data_to_iterate], "data_to_iterate[1] should mean class of each sample. e.g., good, defect, crack.."

        dataset_name = re.sub(r"[\s,\[\]\']", "", str(
            data.dataset.classnames_to_use))
        save_path = f'./saved_fault_images/{self.args.mainmodel}/{dataset_name}'
        # remove previous images
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        scores = torch.tensor(scores)
        labels = torch.tensor(
            [0 if d[1] == 'good' else 1 for d in data.dataset.data_to_iterate])
        ok_scores = scores[labels == 0]
        ng_scores = scores[labels == 1]
        ok_paths = [d[2]
                    for d in data.dataset.data_to_iterate if d[1] == 'good']
        ng_paths = [d[2]
                    for d in data.dataset.data_to_iterate if d[1] != 'good']

        # 모든 NG 가 OK 보다 높은 score 를 가지는 것이 이상적
        # NG 중에 가장 낮은 score 보다 높은 score 를 가진 OK images 를 저장하면 너무 많이 저장되므로, th 를 조금 조정했을 때를 가정해서 NG score 의 중간값보다 높은 score 를 가진 OK images 를 저장
        # ngbase = (ng_scores.max() + ng_scores.min()) / 2
        # target_ok_ids = torch.arange(ok_scores.shape[0])[
        #    ok_scores > ngbase]

        # NOTE 위 코드는 갯수가 예측이 안되므로 OK images 를 높은 score 순서대로 5개 저장
        target_ok_ids = torch.argsort(ok_scores, descending=True)[:5]

        for id in target_ok_ids:
            img = PIL.Image.open(ok_paths[id])
            img.save(
                os.path.join(save_path, f'okfault_p{ok_scores[id]:.2f}_id{id}_ngfrom{ng_scores.min():.2f}to{ng_scores.max():.2f}.png'))

        # 비교를 위해 잘 된 OK images 도 저장
        target_ok_ids = torch.argsort(ok_scores)[:5]
        for id in target_ok_ids:
            img = PIL.Image.open(ok_paths[id])
            img.save(
                os.path.join(save_path, f'okgood_p{ok_scores[id]:.2f}_id{id}_ngfrom{ng_scores.min():.2f}to{ng_scores.max():.2f}.png'))

        # OK 중에 가장 높은 score 보다 낮은 score 를 가진 NG images 를 저장하면 너무 많이 저장되므로, th 를 조금 조정했을 때를 가정해서 OK score 의 중간값보다 낮은 score 를 가진 NG images 를 저장
        # okbase = (ok_scores.max() + ok_scores.min()) / 2
        # target_ng_ids = torch.arange(ng_scores.shape[0])[
        #    ng_scores < okbase]

        # NOTE 위 코드는 갯수가 예측이 안되므로 NG images 를 낮은 score 순서대로 5개 저장
        target_ng_ids = torch.argsort(ng_scores)[:5]

        for id in target_ng_ids:
            img = PIL.Image.open(ng_paths[id])
            img.save(
                os.path.join(save_path, f'ngfault_p{ng_scores[id]:.2f}_id{id}_okfrom{ok_scores.min():.2f}to{ok_scores.max():.2f}.png'))

        # 비교를 위해 잘 된 NG images 도 저장
        target_ng_ids = torch.argsort(ng_scores, descending=True)[:5]
        for id in target_ng_ids:
            img = PIL.Image.open(ng_paths[id])
            img.save(
                os.path.join(save_path, f'nggood_p{ng_scores[id]:.2f}_id{id}_okfrom{ok_scores.min():.2f}to{ok_scores.max():.2f}.png'))


# Image handling classes.


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            features: [torch.Tensor, bs x c x w x h]
        Returns:
            unfolded_features: [torch.Tensor, bs * w//
                stride * h//stride, c, patchsize, patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        """
        features.shape == torch.Size([1, 512, 36, 36]) == [B, C, W, H] from layer 2
        self.patchsize == 3
        self.stride == 1
        padding == 1
        """
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)

        # unfolded_features == torch.Size([8, 4608, 1296])
        unfolded_features = unfolder(features)
        """
        unfolded_features.shape == torch.Size([1, 4608, 1296]) == [B, 3*3*512, 36*36] NOTE 전체 feature 수가 patshsize**2 만큼 늘어남. 즉, patch 간 겹치는 부분이 큼.
        """
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        """
        unfolded_features.shape == torch.Size([1, 512, 3, 3, 1296])
        """
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        """
        unfolded_features.shape == torch.Size([1, 1296, 512, 3, 3])
        """

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
