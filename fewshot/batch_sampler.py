# coding=utf-8
import numpy as np
import torch


class FewBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, n_support, iterations, classes_per_it=2):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(FewBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = n_support*2  # num_support + num_query
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty(
            (len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            # classes 가 labels 와 다르다는 가정하게 label_idx 를 구하는 것이므로, 같다면 필요없음
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(
                np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            # class sampling
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)  # s == slice(0, 10, None)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[
                    self.classes == c].item()  # label_idx == 919
                sample_idxs = torch.randperm(
                    self.numel_per_class[label_idx])[:spc]  # sample_idxs == tensor([11, 16,  7, 10, 15,  0,  3,  6,  4, 17])
                # class 마ㅏ s 개의 sample 을 batch 에 넣는데, 1차원에 순차적으로 offset 변경해가며 넣는다.
                batch[s] = self.indexes[label_idx][sample_idxs]
            # batch 를 섞는다. 즉, way, shot 모두 섞임
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
