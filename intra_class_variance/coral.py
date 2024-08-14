# https://github.com/jindongwang/transferlearning/blob/master/code/distance/coral_pytorch.py
import torch
from intra_class_variance.variance import Variance


class CORAL(Variance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_name = "CORAL"
        self.num_samples = 500

    def _calculate_variance(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss.cpu().item()
