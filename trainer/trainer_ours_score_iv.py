import torch
import tqdm
import torch.nn as nn
from vig_pytorch_iv.vig import vig_224_gelu
from loguru import logger
from trainer.trainer_ours_score import Trainer_Ours_Score
from trainer.vig_wrapper import VIG_wrapper, common_process
from simplenet import Projection


class Trainer_Ours_Score_Iv(Trainer_Ours_Score):
    def initialize_model(self, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, lr, **kwargs):
        model, n_filters = vig_224_gelu(self.args)
        self.discriminator = self._wrap(model, self.args)
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

    def _wrap(self, model, args):
        return VIG_wrapper_score_iv(model, args)

    def _loss_score(self, input_, true_feats_size: int, fake_feats_size: int):
        ivv_s, iv_s, v_s = self.discriminator(input_)
        ivv_s, iv_s, v_s = torch.nn.functional.sigmoid(
            ivv_s), torch.nn.functional.sigmoid(iv_s), torch.nn.functional.sigmoid(v_s)

        intervention_times = min(
            self.args.intervention_times, len(v_s) - 1)
        la = self.args.la_penalty

        # Get penalty
        select = torch.randperm(len(v_s))[:intervention_times].to(v_s.device)
        # torch.sigmoid(v_s).detach()[select]  # [I,1]
        alls = v_s.detach()[select]
        allc = iv_s.squeeze(1).expand(
            intervention_times, iv_s.shape[0])  # [I,E]
        conf = allc*alls

        true_scores = conf[:, :true_feats_size].flatten()
        fake_scores = conf[:, fake_feats_size:].flatten()
        lossf = torch.nn.BCELoss(reduction='none')
        intervention_loss = torch.concat([lossf(true_scores, torch.ones_like(
            true_scores)), lossf(fake_scores, torch.zeros_like(fake_scores))], dim=0)
        intervention_loss = intervention_loss.view(intervention_times, -1)
        intervention_loss = intervention_loss.mean(dim=-1)

        # intervention_true_loss = torch.clip(-true_scores + th, min=0)
        # intervention_fake_loss = torch.clip(fake_scores + th, min=0)

        # intervention_loss = torch.concat(
        #    [intervention_true_loss, intervention_fake_loss], dim=-1).mean(dim=-1)

        # NOTE intervention_times 가 매우 크면 var() 가 엄청나게 커짐
        env_mean, env_var = intervention_loss.mean(), (intervention_loss *
                                                       intervention_times).var()
        penalty = la * (env_mean + env_var)

        # Get original loss
        true_scores = iv_s[:true_feats_size].flatten()
        fake_scores = iv_s[fake_feats_size:].flatten()
        th = torch.concat(
            [true_scores.detach(), fake_scores.detach()]).median()
        p_true = (true_scores.detach() >= th).sum() / \
            len(true_scores)
        p_fake = (fake_scores.detach() < th).sum() / \
            len(fake_scores)

        # logger.info(f"th: {th:.2f}")

        lossf = torch.nn.BCELoss()
        orig_loss = lossf(true_scores, torch.ones_like(
            true_scores)) + lossf(fake_scores, torch.zeros_like(fake_scores))

        loss = orig_loss + penalty
        # logger.info( f"{orig_loss.item()}, {penalty.item()}, alls max {alls.max().item():.2f}, alls min {alls.min().item():.2f}, iv_s max {iv_s.max().item():.2f}, iv_s min {iv_s.min().item():.2f}, v_s max {v_s.max().item():.2f}, v_s min {v_s.min().item():.2f}")

        return [loss, orig_loss, penalty], p_true, p_fake

        # faster approximate version of spatial-temporal of DIDA
        # select = torch.randperm(len(sy))[:intervention_times].to(sy.device)
        # alls = torch.sigmoid(sy).detach()[select].unsqueeze(-1)  # [I,1]
        # allc = cy.expand(intervention_times, cy.shape[0])  # [I,E]
        # conf = allc*alls
        # alle = edge_label.expand(intervention_times, edge_label.shape[0])
        # crit = torch.nn.BCELoss(reduction='none')
        # env_loss = crit(conf.flatten(), alle.flatten())
        # env_loss = env_loss.view(intervention_times, sy.shape[0]).mean(dim=-1)
        # env_mean = env_loss.mean()
        # env_var = torch.var(env_loss*intervention_times)
        # penalty = env_mean+env_var

    def _get_pred_scores(self, features):
        """
        ivv_x, iv_x, v_x 중 iv_x 만 score 로 사용
        """
        return -(self.discriminator(features)[1])


class VIG_wrapper_score_iv(VIG_wrapper):
    def _define_prediction(self):
        self.model.prediction.append(nn.Linear(1024, 1, bias=True))

    def _post_processing(self, x, orig_x):
        # [B, 1, 1, 1] => [B]
        B = x.shape[0]//3
        ivv_x, iv_x, v_x = x[:B], x[B:B*2], x[B*2:]
        return ivv_x, iv_x, v_x

    def _midprocess(self, x):
        # ivv_X 만 다음 layer 로 전달, 마지막 layer 에서는 midprocess 를 call 하지 않도록 짜여져 있음
        return x[:x.shape[0]//3]

    def forward(self, x):
        x, orig_x = common_process(
            x, self.model.backbone, self.model.prediction, self._patch_to_batch, self._batch_to_patch, midprocess_func=self._midprocess)
        x = self.model.last(x)

        ret = self._post_processing(x, orig_x)
        return ret
