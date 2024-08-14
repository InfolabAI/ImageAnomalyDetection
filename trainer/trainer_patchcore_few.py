import common
from tqdm import tqdm
import torch

from copy import deepcopy
from trainer.trainer_patchcore import Trainer_PatchCore
from fewshot.loss import FewLoss
from fewshot.batch_sampler import FewBatchSampler


class Trainer_PatchCore_Few(Trainer_PatchCore):
    def _set_dataloader_few(self, training_dataloader):
        if self.backbone_train_dataloader is None:
            ds = deepcopy(training_dataloader.dataset)
            ds.few_shot_mode = True
            ds.imgpaths_per_class, ds.data_to_iterate = ds.get_image_data()
            labels = [int(data[1] != "good")for data in ds.data_to_iterate]
            # NOTE training dataloader 를 따로 분리하지 않고 사용하면, coreset 에 normal, abnormal 둘 다 포함되는 문제가 생김
            self.backbone_train_dataloader = torch.utils.data.DataLoader(
                ds,
                num_workers=training_dataloader.num_workers,
                batch_sampler=FewBatchSampler(
                    labels=labels, n_support=self.args.n_support, iterations=self.args.few_iterations),
            )

        return self.backbone_train_dataloader

    def set_backbone(self, backbone, device):
        # for name, module in self.backbone.named_modules():
        #    print(name)
        self.backbone = backbone.to(device)
        self.backbone_opt = torch.optim.Adam(
            self.backbone.parameters(), lr=1e-5)
        self.calc_loss = FewLoss(self.args.n_support).to(device)
        self.backbone_train_dataloader = None

    def set_aggregator(self, train_backbone=True):
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone=True
        )
        return feature_aggregator

    def _train_embed(self, images, evaluation=False):
        self.forward_modules["feature_aggregator"].train()  # wideresnet
        features = self.forward_modules["feature_aggregator"](
            images, eval=evaluation)
        features = [features[layer]
                    for layer in self.layers_to_extract_from]
        features = self.forward_modules["preprocessing"](
            features)  # avgpooling 정해진 dim 1536 (layer 2 의 channel 수 + layer 3 의 channel 수)
        features = self.forward_modules["preadapt_aggregator"](
            features)  # further pooling
        return features

    def _pretrain_model(self, training_data, val_data, test_data, dataset_name):
        training_data = self._set_dataloader_few(training_data)
        self.backbone.train()
        losses = []
        with tqdm(enumerate(training_data), total=len(training_data)) as titer:
            for i, data in titer:
                # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
                labels = data["is_anomaly"].to(self.device)
                image = data["image"].to(self.device)

                # 이전 batch에서 계산된 가중치를 초기화
                self.backbone_opt.zero_grad()

                # forward + back propagation 연산
                outputs = self._train_embed(image)
                train_loss, train_acc = self.calc_loss(outputs, labels)
                train_loss.backward()
                self.backbone_opt.step()
                titer.set_postfix(
                    {"iter": i, "loss": float(train_loss.cpu().detach()), "acc": float(train_acc.cpu().detach())})
                losses.append(float(train_loss.cpu().detach()))

        return losses
