# NOTE 성능이 안 좋아서 DEPRECATED
import torch.nn as nn
from trainer.trainer_ours_gans_img_coreset import Trainer_Ours_GANs_IMG_Coreset


class Trainer_Ours_NoGANs_IMG_Coreset(Trainer_Ours_GANs_IMG_Coreset):
    def update_GDC(self, x, gan_x, target):
        ####
        # update clf
        self.dsc_opt.zero_grad()
        # metric_loader aggregator
        self.metric_loader.aggreator(self._get_disc_input_shape(x), target)
        # errC
        output = self._get_disc_ret(x)
        errC = nn.CrossEntropyLoss()(output, target)
        loss = errC

        loss.backward()

        return loss.detach().item(), output
