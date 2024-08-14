# NOTE 성능이 안 좋아서 DEPRECATED
import torch
import json
import os
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import sys
from vig_pytorch.vig import vig_224_gelu
from loguru import logger
from trainer.trainer_ours_score import Trainer_Ours_Score
from trainer.vig_wrapper import VIG_wrapper, ExceptLast
from simplenet import Projection
from models.gans_dcgans import ManageGANs
from torchvision.utils import save_image
# Probability-based
from pytorch_ood.detector import MaxSoftmax, MCD, TemperatureScaling, KLMatching, Entropy
from pytorch_ood.detector import ODIN, EnergyBased, MaxLogit, OpenMax  # Logit-based
from pytorch_ood.detector import Mahalanobis, KNN, ViM, SHE, RMD  # Feature-based
from pytorch_ood.detector import ASH, DICE, ReAct  # Activation-based


class Trainer_Ours_GANs_IMG(Trainer_Ours_Score):
    # NOTE 현재 성능이 잘 안나오는 버전
    def initialize_model(self, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs):
        super().initialize_model(pre_proj, proj_layer_type, meta_epochs,
                                 aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs)

        self.mg = ManageGANs(imgsize=32, ngf=100, nc=3, device=self.device)
        nG, self.nD = self.mg.get_GD()
        self.nG = GeneratorWrapper(
            nG, self.discriminator.model.backbone[0][0].fc1[0].in_channels, self.device)
        self.optimD = torch.optim.Adam(
            self.nD.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
        self.optimG = torch.optim.Adam(
            self.nG.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        self.real_label = 1
        self.gen_label = 0
        self.num_classes = 2  # OK, noisedOK

        self.beta = self.args.gan_beta

        self.errD, self.errG, self.errG_KL = 0, 0, 0
        self.i_iter = 0
        self.metric_loader = GenMetricDataLoader()

    def update_GD(self, x):
        real_target = torch.FloatTensor(x.shape[0], 1).fill_(
            self.real_label).to(self.device)
        fake_target = torch.FloatTensor(x.shape[0], 1).fill_(
            self.gen_label).to(self.device)

        ####
        # update G
        self.optimG.zero_grad()
        self.noise = self.mg.get_noise(x.shape[0])
        gen, _ = self.nG(self.noise)
        errG = self.criterion(self.nD(gen), real_target)
        errG.backward()

        self.optimG.step()
        self.errG += errG.item()

        ####
        # update D
        # train with real
        self.optimD.zero_grad()  # NOTE zero_grad 는 optimG.step() 이 종료된 이후에 실행해야 함
        errD_real = self.criterion(self.nD(x), real_target)
        # train with fake
        errD_fake = self.criterion(self.nD(gen.detach()), fake_target)
        errD = (errD_real + errD_fake)/2
        errD.backward()

        self.optimD.step()
        self.errD += errD.item()

        return gen

    def update_GDC(self, x, gan_x, target):
        gan_real_target = torch.FloatTensor(gan_x.shape[0], 1).fill_(
            self.real_label).to(self.device)
        gan_fake_target = torch.FloatTensor(gan_x.shape[0], 1).fill_(
            self.gen_label).to(self.device)

        ####
        # update G
        self.optimG.zero_grad()
        self.noise = self.mg.get_noise(gan_x.shape[0])
        # errG_D
        gen, gen_feat = self.nG(self.noise)
        errG_D = self.criterion(self.nD(gen), gan_real_target)
        # errG_KL
        gen_feat = self.patch_maker.patchify(gen_feat)
        KL_fake_output = F.log_softmax(self._get_disc_ret(gen_feat))
        uniform_dist = torch.Tensor(
            KL_fake_output.size(0), self.num_classes).fill_((1./self.num_classes)).to(self.device)
        errG_KL = F.kl_div(KL_fake_output, uniform_dist) * \
            self.num_classes
        errG = errG_D*self.beta + errG_KL
        errG.backward()

        self.optimG.step()
        self.errG += errG.item()

        ####
        # update D
        # train with real
        self.optimD.zero_grad()  # NOTE zero_grad 는 optimG.step() 이 종료된 이후에 실행해야 함
        errD_real = self.criterion(self.nD(gan_x), gan_real_target)
        # train with fake
        errD_fake = self.criterion(self.nD(gen.detach()), gan_fake_target)
        errD = (errD_real + errD_fake)/2
        errD.backward()

        self.optimD.step()
        self.errD += errD.item()

        ####
        # update clf
        self.dsc_opt.zero_grad()
        # metric_loader aggregator
        self.metric_loader.aggreator(self._get_disc_input_shape(x), target)
        # errC
        output = self._get_disc_ret(x)
        errC = nn.CrossEntropyLoss()(output, target)
        # errC_KL
        # gen_feat = self.patch_maker.patchify(gen_feat.detach()) 위에서 이미 patchify 했음
        KL_fake_output = F.log_softmax(self._get_disc_ret(gen_feat.detach()))
        uniform_dist = torch.Tensor(
            KL_fake_output.size(0), self.num_classes).fill_((1./self.num_classes)).to(self.device)
        errC_KL = F.kl_div(KL_fake_output, uniform_dist) * \
            self.num_classes
        loss = errC*self.beta + errC_KL

        loss.backward()

        return loss.detach().item(), output

    def save_gans_output(self, name, additional_folder=None):
        if self.args.wan:
            return
        self.noise = self.mg.get_noise(25)
        self.nG.eval()
        gen, _ = self.nG(self.noise)
        gan_image_path = os.path.join(
            self.gan_path, "gans" if additional_folder is None else additional_folder)
        os.makedirs(gan_image_path, exist_ok=True)
        save_image(gen.data[:25], os.path.join(
            gan_image_path, f"generated_{name}.png"), nrow=5, normalize=True)
        self.nG.train()

    def _loss_function(self, input_, true_feats_size, fake_feats_size, gan_x):
        true_target = torch.ones(true_feats_size, dtype=int).to(self.device)
        fake_target = torch.zeros(fake_feats_size, dtype=int).to(self.device)
        target_ = torch.cat([true_target, fake_target])
        loss, output = self.update_GDC(input_, gan_x, target_)
        return loss, output, target_

    def _get_disc_input_shape(self, x):
        """
        input shape 을 discriminator 에 맞게 변환. 
        - gans_img 는 이미 이 형태라 상관없음.
        - gans_img_coreset 은 이 형태로 변환이 필요함. memory bank feature 에 대한 tensor 이기 때문.
        """
        return x.reshape(-1, self.C, self.H, self.W)

    def _get_disc_ret(self, x):
        x = self._get_disc_input_shape(x)
        return self.discriminator(x)

    def _print_GANs_err(self):
        if self.errD == 0:
            return
        logger.info(
            f"errD: {self.errD}, errG: {self.errG}, errG_KL: {self.errG_KL}")
        self.errD, self.errG, self.errG_KL = 0, 0, 0

    def _wrap(self, model, args):
        return VIG_wrapper_score(model, args)

    def _pre_meta_train(self, training_data, val_data, test_data, dataset_name):
        gan_path = self.gan_path = os.path.join(
            "saved_model", "gan", dataset_name)
        os.makedirs(gan_path, exist_ok=True)
        G_path = os.path.join(gan_path, "G.pth")
        D_path = os.path.join(gan_path, "D.pth")
        if os.path.exists(os.path.join(G_path)):
            self.nG.load(G_path)
            self.nD.load_state_dict(torch.load(D_path))
            logger.info("Loaded GANs model")
            self.save_gans_output("0_loaded")
        else:
            self._train_gans(training_data, dataset_name)
            torch.save(self.nG.state_dict(), G_path)
            torch.save(self.nD.state_dict(), D_path)
            logger.info("Saved GANs model")

    def _train_gans(self, input_data, dataset_name):
        """
        NOTE DEPRECATED. 이거 대신에 train_gan.py 로 대체함"""
        self.nG.train()
        self.nD.train()
        logger.info(f"Training GANs...")
        input_data.dataset.gan_mode = True  # 288x288 은 처리하지 않게하여 속도 높임
        gan_dataloader = torch.utils.data.DataLoader(
            input_data.dataset, batch_size=64, shuffle=True, num_workers=4)

        # code to load json file into dict
        with open("/home/robert.lim/main/other_methods/my_GNN_SimpleNet/saved_model/gans_iter.json", "r") as read_file:
            epochs = json.load(read_file)[dataset_name]

        i_epoch = 0
        gens = []
        reals = []
        with tqdm.tqdm(total=epochs) as pbar:
            while i_epoch < epochs:
                # NOTE iter(gan_dataloader) 를 만드는 시간이 dataset 마다 천차만별로 다름 PCBNG 는 매우 빠르고, PCBNG_0.01_M 이 좀 더 느리고, mvtec_capsule 은 매우 느림
                # for data_item in gan_dataloader:
                img = next(iter(gan_dataloader))['image_gans'].to(self.device)
                gen = self.update_GD(img)

                gens.append(gen.detach().cpu())
                reals.append(img.detach().cpu())
                i_epoch += 1

                pbar.update(1)
                pbar.set_description_str(
                    f"errD: {self.errD}, errG: {self.errG}")

                if i_epoch % 500 == 0:
                    self.save_gans_output(i_epoch)
                    if not self.args.wan:
                        real_image_path = os.path.join(
                            self.gan_path, "real")
                        os.makedirs(real_image_path, exist_ok=True)
                        save_image(
                            torch.concat(reals).data[:16], os.path.join(real_image_path, f"real_{i_epoch}.png"), nrow=4, normalize=True)

        self.errD, self.errG = 0, 0
        input_data.dataset.gan_mode = False

    def _preprocessing_image(self, data_item):
        return [super()._preprocessing_image(data_item),
                data_item['image_gans'].to(self.device)]

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        self.nG.train()
        self.nD.train()
        logger.info(f"Training discriminator...")
        all_loss = []
        all_output = []
        all_target = []
        with tqdm.tqdm(total=self.gan_epochs, position=0) as pbar:
            while pbar.n < self.gan_epochs:
                for data_item in tqdm.tqdm(input_data, desc="inner loop", position=1, leave=False):
                    if pbar.n >= self.gan_epochs:
                        break

                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    true_feats, gan_x = self._preprocessing_image(data_item)
                    out = self._preprocessing_train_disc(true_feats)
                    true_feats, fake_feats = self.apply_augment(out)
                    feats = self._postprocessing_train_disc(
                        true_feats, fake_feats)
                    loss, output, target = self._loss_function(
                        feats, len(true_feats), len(fake_feats), gan_x)

                    self.proj_opt.step()
                    all_loss.append(loss)
                    all_output.append(output)
                    all_target.append(target)

                    if self.cos_lr:
                        self.dsc_schl.step()

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
        self.errD, self.errG = 0, 0

        return round(all_loss_sum, 5)

    def _additional_process(self, training_data, val_data, test_data, dataset_name, i_mepoch):
        """
        train_discriminator 다음 실행되는 함수"""
        self.save_gans_output(f"{i_mepoch}_with_clf", "clf")

        def _fit(model):
            model.fit(self.metric_loader(), self.device)

        def _not_fit(model):
            pass

        metric_dict = {
            "maxsoftmax": [MaxSoftmax(self.discriminator), _not_fit],
            # var 모드는 모두 0이 나와서 못 씀
            "mcd": [MCD(self.discriminator, mode='mean'), _not_fit],
            "temperaturescaling": [
                TemperatureScaling(self.discriminator), _fit],
            "klmatching": [KLMatching(self.discriminator), _fit],
            "entropy": [Entropy(self.discriminator), _not_fit],
            "maxlogit": [MaxLogit(self.discriminator), _not_fit],
            "openmax": [OpenMax(self.discriminator), _fit],
            "energybased": [EnergyBased(self.discriminator), _not_fit],
            "odin": [ODIN(self.discriminator), _not_fit],
            "mahalanobis": [Mahalanobis(self.discriminator), _fit],
            "knn": [KNN(self.discriminator), _fit],
            "vim": [ViM(ExceptLast(self.discriminator), d=100, w=self.discriminator.model.last.weight,
                        b=self.discriminator.model.last.bias), _fit],  # NOTE d 에 따라 달라짐
            # NOTE coreset 에 대한 class 를 사용하면, 일부 class 는 sample 이 1개도 없을 수 있는데, rmd 는 모든 class 에 적어도 1개의 sample 이 있지 않으면 에러 발생
            # "rmd": [RMD(self.discriminator), _fit],
            "she": [SHE(ExceptLast(self.discriminator), head=self.discriminator.model.last), _fit],
            # NOTE variant 에 따라 달라짐
            # "ash": [ASH(ExceptLast(self.discriminator), head=self.discriminator.model.last, variant="ash-s"), _fit], # NOTE linear 가 아닌 conv 를 이용한 last layer 를 요구함
            "react": [ReAct(ExceptLast(self.discriminator), head=self.discriminator.model.last), _fit],
            "dice": [DICE(ExceptLast(self.discriminator), w=self.discriminator.model.last.weight,
                          b=self.discriminator.model.last.bias, p=0.1), _fit],  # NOTE p 에 따라 달라짐
        }
        self.metric = metric_dict[self.args.ood_metric][0]
        metric_dict[self.args.ood_metric][1](self.metric)

    def _get_pred_scores(self, features):
        """ torch.no_grad(). eval() 상태에서 image 1개마다 call 되는 함수
        """
        self._print_GANs_err()
        # score = ODIN(self.discriminator)(features)
        score = self.metric(features)
        return score
        # soft_ret = torch.nn.functional.softmax(ret, -1)
        # NOTE softmax 결과 중 높은 값이 작을수록 entropy 가 높으므로, 1/max 를 anomaly score 로 사용가능
        # min_score = 1/soft_ret.max(1)[0]
        # return min_score  # + 0.1*fake_score


class VIG_wrapper_score(VIG_wrapper):
    def _define_prediction(self):
        # self.model.prediction.append(
        #    nn.Conv2d(1024, 2, 1, bias=True))  # OK, noisedOK
        self.model.prediction.append(
            nn.Linear(1024, 2, bias=True)
        )

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        return x.reshape(x.shape[0], -1)

#
# class Pooling_Linear(nn.Module):
#    def __init__(self, in_features, out_features, bias=True):
#        super().__init__()
#        self.linear = nn.Linear(in_features, out_features, bias)
#        self.weight = self.linear.weight
#        self.bias = self.linear.bias
#
#    def forward(self, x):
#        x = x.mean(-1).mean(-1)
#        return self.linear(x)
#
#


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        if targets is None:
            targets = torch.zeros(len(data), dtype=torch.long)
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_labels(self):
        return self.targets


class GenMetricDataLoader:
    def __init__(self):
        self.init()

    def init(self):
        self.data, self.targets = [], []

    def aggreator(self, data, target):
        data, target = data.detach().cpu(), target.detach().cpu()
        self.data.append(data)
        self.targets.append(target)

    def __call__(self):
        if len(self.data) == 0:
            return None
        loader = torch.utils.data.DataLoader(CustomDataset(
            torch.concat(self.data), torch.concat(self.targets)), batch_size=500, shuffle=True, num_workers=1)
        self.init()
        return loader


class GeneratorWrapper(nn.Module):
    def __init__(self, G, vig_channels, device):
        super().__init__()
        self.G = G
        self.adapter = nn.Conv2d(64, vig_channels, 1).to(device)

    def forward(self, z):
        out, adapt_out = self.G(z)
        adapt_out = self.adapter(adapt_out)
        return out, adapt_out

    def load(self, path):
        self.G.load_state_dict(torch.load(path))
