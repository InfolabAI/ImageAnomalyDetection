# NOTE 성능이 안 좋아서 DEPRECATED
import torch.nn as nn
from trainer.trainer_ours_gans_img import Trainer_Ours_GANs_IMG


class Trainer_Ours_NoGANs_IMG(Trainer_Ours_GANs_IMG):
    def update_GDC(self, x, gan_x, target):
        ####
        # update clf
        self.dsc_opt.zero_grad()
        # metric_loader aggregator
        self.metric_loader.aggreator(x, target)
        # errC
        output = self._get_disc_ret(x)
        errC = nn.CrossEntropyLoss()(output, target)
        loss = errC

        loss.backward()

        return loss.detach().item(), output
