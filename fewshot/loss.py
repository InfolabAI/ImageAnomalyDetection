# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class FewLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(FewLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    # NOTE 결과적으로, target 이 random 하게 섞여있어도(오히려 random 이기에 support, query 가 매번 바뀌며 일반화 됨) N-way, K-shot 이라는 것만 보장되면, loss 계산이 가능함

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    """
    (Pdb) p input.shape
        torch.Size([600, 64])
    (Pdb) p target.shape
        torch.Size([600])
    """
    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    """
    (Pdb) p len(classes)
        60
    (Pdb) p len(target)
        600
    (Pdb) p classes                                                                                                  
        tensor([   0,  209,  268,  315,  319,  467,  475,  ... , 3975, 3979, 4002, 4061])
    (Pdb) p target                                                                                                   
        tensor([3265, 3526, 2687,  ..., 3015, 4061, 2687, 1508], device='cuda:0')
    (Pdb) p n_query
        5
    (Pdb) p n_support
        5
    """

    support_idxs = list(map(supp_idxs, classes))
    """
    (Pdb) p len(support_idxs) # target 중 각 class 에 해당하는 idxs 를 고른다음 n_support 만큼 자른 것
        60
    (Pdb) p support_idxs
        [tensor([ 67, 142, 257, 303, 420]), tensor([  7, 193, 307, 325, 350]), ... , tensor([ 25,  48,  69, 137, 281])]
    """

    # model 이 600 개 input 에 대해 이미 feature 를 생성했으므로, 위에서 만든 idxs 로 class 마다 n_support 개를 골라 mean 해서 class 마다 prototype 을 만든다.
    prototypes = torch.stack([input_cpu[idx_list].mean(0)
                             for idx_list in support_idxs])
    # dataloader 의 출력은 random 하지만, 각 class 마다 10개의 sample 이 있다는 것은 보장됨. 즉, sample 0~4 가 support, 5~9 가 query 가 된다는 것이 보장됨.
    # FIXME when torch will support where as np
    query_idxs = torch.stack(
        list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    """
    (Pdb) p query_idxs
        tensor([472, 500, 503, ... , 567, 580, 597])
    (Pdb) p len(query_idxs)
        300
    (Pdb) p prototypes
        tensor([[0.3408, 0.5575, 0.4635,  ..., 1.0871, 0.7133, 0.8635], ...,
                [0.9519, 0.2991, 0.6851,  ..., 0.3080, 0.5883, 1.0857]], grad_fn=<StackBackward0>)
    (Pdb) p prototypes.shape
        torch.Size([60, 64])
    """

    query_samples = input.to('cpu')[query_idxs]
    """
    (Pdb) p query_samples.shape
        torch.Size([300, 64])
    """
    # loss 계산을 위해, 각 query 마다, 모든 prototypes 와의 거리를 구함
    dists = euclidean_dist(query_samples, prototypes)
    """
    (Pdb) p dists.shape
        torch.Size([300, 60])
    (Pdb) p dists
    tensor([[14.3274, 34.4233, 30.4582,  ..., 20.0609, 20.3648, 41.7301], ...,
            [37.1853, 20.2853, 20.1492,  ..., 30.1236, 39.0684, 17.1026]], grad_fn=<SumBackward1>)
    """

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    """
    (Pdb) p log_p_y.shape
        torch.Size([60, 5, 60])
    """

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    """
    (Pdb) p target_inds.shape
        torch.Size([60, 5, 1])
    (Pdb) p target_inds.squeeze(2)
    tensor([[ 0,  0,  0,  0,  0], 
            [ 1,  1,  1,  1,  1], ... , 
            [59, 59, 59, 59, 59]])
    """

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val
