import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import pickle
import tqdm
from trainer.trainer_ours_score_patchlevel import Trainer_Ours_Score_PatchLevel, VIG_wrapper_score
from trainer.vig_wrapper import VIG_wrapper, common_process
from vig_pytorch_interpretable.vig import vig_224_gelu


class Trainer_Ours_Score_PatchLevel_Interpretable(Trainer_Ours_Score_PatchLevel):
    def _get_vig(self, args):
        model, n_filters = vig_224_gelu(args)
        return model, n_filters

    def _preprocessing_predict_dataloader(self, data):
        """
        Trainer_Ours_Score_PatchLevel_Interpretable 에서 image_path 위에 explain 을 출력하기 위함
        """
        self.discriminator.model.set_image_paths(
            data['image_path'])  # NOTE torch_vectex.py 에서 이것이 설정되어있을때만 explain 을 출력하고 저장함

    def _get_segmentations(self, patch_scores, batchsize, patch_shapes):
        breakpoint()
        # (1, 1, 28, 28) > (1, 1, 1, 28, 28)
        patch_scores = self.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        )
        scales = patch_shapes[0]
        # (1, 1, 28, 28) > (1, 1, 1, 28, 28)
        patch_scores = patch_scores.reshape(
            batchsize, scales[0], scales[1])
        features = features.reshape(batchsize, scales[0], scales[1], -1)
        masks, features = self.anomaly_segmentor.convert_to_segmentation(
            patch_scores, features)
        return masks, features

    def train(self, training_data, val_data, test_data, dataset_name):
        """
        training_data, val_data, test_data 는 모두 ApproximationDataset 의 instance
        """
        img_paths, patch_scores, patchsize = self.load_labels()
        train_dataset = ApproximationDataset(
            val_data.dataset, img_paths, patch_scores)
        training_data = DataLoader(
            train_dataset, batch_size=training_data.batch_size, shuffle=True, num_workers=1)
        return self._meta_train(training_data, training_data, training_data, dataset_name)

    def _meta_train(self, training_data, val_data, test_data, dataset_name):
        self.dataset_name = dataset_name

        for i_mepoch in range(self.meta_epochs):
            logger.info(f"\n\n----- {i_mepoch} -----")
            loss = self._train_discriminator(training_data)
            scores, segmentations, features, labels_gt, masks_gt = self.predict(
                val_data)

        exit(0)
        return best_record

    def _predict(self, images):
        self.pre_proj = 0  # NOTE 현재 pre_proj 를 사용하지 않는 상태이므로 0으로 설정
        return super()._predict(images)

    def _get_pred_scores(self, features):
        return self.discriminator(features)

    def load_labels(self):
        # load
        with open(f"/home/robert.lim/main/other_methods/my_GNN_SimpleNet/output_labels/patchcore_['capsule']_3_val_patch_scores.pkl", 'rb') as f:
            img_paths, patch_scores, patchsize = pickle.load(f)

        return img_paths, patch_scores, patchsize

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        logger.info(f"Training discriminator...")

        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            all_loss = []

            for i_epoch in range(self.gan_epochs):
                losses = []

                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    imgs, labels = data_item['image'], data_item['label']
                    # self.dec_opt.zero_grad()
                    i_iter += 1
                    imgs, labels = imgs.to(torch.float).to(self.device), labels.to(
                        torch.float).to(self.device)
                    out = self._embed(imgs, evaluation=False)[0]

                    scores = self.discriminator(out)
                    loss = torch.nn.functional.mse_loss(
                        scores.squeeze(0), labels)

                    loss.backward()
                    self.dsc_opt.step()
                    losses.append(loss.cpu().item())

                losses = round(sum(losses) / len(losses), 5)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{losses} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)
                all_loss.append(losses)

        return round(sum(all_loss) / len(all_loss), 5)

    def _wrap(self, model, args):
        return VIG_wrapper_score_stem(model, args)


class ApproximationDataset(Dataset):
    def __init__(self, dataset, image_paths, patch_scores):
        self.dataset = dataset
        self.mapping_dict = {img_path: patch_score for img_path, patch_score in zip(
            image_paths, patch_scores)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return {
            'image': data['image'],
            'label': self.mapping_dict[data['image_path']],
            'image_path': data['image_path'],
            'is_anomaly': data['is_anomaly']
        }


class VIG_wrapper_score_stem(VIG_wrapper_score):
    def _define_prediction(self):
        self.model.prediction.append(nn.Conv2d(1024, 1, 1, bias=True))

    def _remove_stem(self):
        pass

    def forward(self, x):
        ret = self.model(x)
        return ret

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        return x.reshape(-1)
