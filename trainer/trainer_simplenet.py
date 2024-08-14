from loguru import logger
import wandb
import torch
import tqdm
from utils import plot_segmentation_images
import torch
from trainer.trainer import Trainer
from simplenet import Discriminator, Projection


class Trainer_SimpleNet(Trainer):
    def initialize_model(self, dsc_layers, dsc_hidden, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, dsc_lr, lr, **kwargs):
        self.discriminator = Discriminator(
            self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)

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

    def _get_pred_scores(self, features):
        """
        trainer_ours_gans 는 다른 함수 필요
        """
        return -self.discriminator(features)

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        self.elapsed_timer.reset()
        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 evaluation=True)
            self.elapsed_timer.elapsed(
                f"{self.dataset_name}_feature extraction", batchsize)
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = self._get_pred_scores(features)
            self.elapsed_timer.elapsed(
                f"{self.dataset_name}_discriminator", batchsize)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            masks, features = self._get_segmentations(
                patch_scores, batchsize, patch_shapes)

        # list(masks), list(features), list(patch_scores)
        return list(image_scores), [-1], [-1], [-1]

    def _get_segmentations(self, patch_scores, batchsize, patch_shapes):
        return None, None
        # patch_scores = self.patch_maker.unpatch_scores(
        #    patch_scores, batchsize=batchsize
        # )
        # scales = patch_shapes[0]
        # patch_scores = patch_scores.reshape(
        #    batchsize, scales[0], scales[1])
        # features = features.reshape(batchsize, scales[0], scales[1], -1)
        # masks, features = self.anomaly_segmentor.convert_to_segmentation(
        #    patch_scores, features)
        # self.elapsed_timer.elapsed(
        #    f"{self.dataset_name}_anomaly map generation", batchsize)
        # return masks, features

    def _preprocessing_train_disc(self, features):
        """
        Ours mask 버전 구현을 위한 module 화
        """
        return features
