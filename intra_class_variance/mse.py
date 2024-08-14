import torch
from intra_class_variance.variance import Variance


class MSE(Variance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_name = "MSE"
        self.num_samples = 500

    def _calculate_variance(self, source, target):
        loss = torch.nn.functional.mse_loss(source, target)
        return loss.cpu().item()
