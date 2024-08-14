import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger


class Variance:
    def __init__(self, device, features: torch.tensor, labels: np.array):
        """
        Parameters
        ----------
        features : shape==[#samples, #features]
        labels : shape==[#samples]
        """
        self.features = features.cpu()
        self.labels = labels
        self.device = device

    def _calculate_variance(self, source, target):
        raise NotImplementedError

    def _preprocessing(self, features, level_name):
        dist_list = []
        logger.info(f"The shape of features: {features.shape}")
        for i in tqdm.tqdm(range(self.num_samples), desc=f"Calculating {self.metric_name}"):
            # sample 50 patches from features_class
            # idxs1, idxs2 = torch.randint(0, features.shape[0], (100,)), torch.randint( 0, features.shape[0], (100,))
            idxs1, idxs2 = self._sample_ids(features.shape[0], 100)
            ret = self._calculate_variance(
                self._sample_patches(features, level_name)[
                    idxs1].to(self.device),
                self._sample_patches(features, level_name)[
                    idxs2].to(self.device)
            )
            # free torch memory
            torch.cuda.empty_cache()
            dist_list.append(ret)

        mean, std = np.array(dist_list).mean(), np.array(dist_list).std()
        return mean, std

    def eval_variance(self, level_name: str, dataset_name: str, loader_name: str):
        if 'patch' in level_name:
            assert len(
                self.features.shape) == 3, "For patch-level, you should get features like features.shape == (batch, #patches, #features), use --mainmodel==simple"
        else:
            assert len(
                self.features.shape) == 2, "For image-level, you should get features like features.shape == (batch, #features), use --mainmodel==simple"

        logger.info(
            f"Calculating variance.. {level_name} {dataset_name} {loader_name}")
        df = pd.DataFrame(
            columns=["level", "metric_name",  "dataset", "split", "label", "mean", "None"])
        # label 마다 feature 의 variance 를 구함
        for label in np.unique(self.labels):
            # get indices of the same class
            ids = np.arange(len(self.labels))[self.labels == label]
            features_class = self.features[ids]

            mean, std = self._preprocessing(features_class, level_name)
            df.loc[len(df)] = [level_name, self.metric_name, dataset_name,
                               loader_name, label, mean, 0]
        return df

    def _sample_patches(self, features, level_name, num=10):
        if 'patch' in level_name:
            # sample 10 indices without replacement
            idxs = torch.randperm(features.shape[1])[:num]
            features = features[:, idxs]
            features = features.reshape(features.shape[0], -1)
            return features
        else:
            return features

    def _sample_ids(self, max, num):
        idxs1, idxs2 = torch.randint(0, max, (num,)), torch.randint(
            0, max, (num,))
        diff_ids = idxs1 != idxs2
        idxs1, idxs2 = idxs1[diff_ids], idxs2[diff_ids]
        """
        Remove duplicates
        >>> zip(idxs1, idxs2)
            [(0, 1), (3, 4), (3, 2), (2, 4), (3, 0), (3, 2), (3, 0), (4, 1), (2, 3), (1, 0), (4, 1), (0, 4), (4, 0), (4, 2), (0, 2), (4, 2), (3, 2), (4, 3), (0, 2), (1, 3), (2, 3), (1, 0), (2, 3), (4, 1), (3, 0), (3, 2), (4, 2), (1, 3), (2, 4), (3, 0), (2, 0), (4, 0), (2, 0), (0, 1), (0, 3), (2, 3), (1, 0), (0, 1), (4, 2), (0, 3), (0, 3), (2, 4), (3, 0), (1, 0), (1, 2), (4, 3), (4, 3), (2, 4), (2, 4), (4, 1), (3, 2), (1, 3), (3, 2), (3, 1), (1, 4), (1, 0), (1, 0), (0, 2), (1, 3), (2, 4), (4, 1), (3, 1), (2, 1), (2, 1), (4, 0), (4, 2), (2, 0), (4, 3), (3, 0), (0, 2), (2, 4), (3, 1), (4, 0), (3, 4), (3, 4), (4, 2), (1, 3), (3, 4), (2, 0)]
        >>> dedup_ids
            {(0, 1), (2, 4), (1, 2), (0, 4), (3, 4), (0, 3), (1, 4), (2, 3), (0, 2), (1, 3)}
        """
        dedup_ids = set([tuple(sorted([i1.item(), i2.item()]))
                        for i1, i2 in zip(idxs1, idxs2)])
        """
        >>> idxs1
            tensor([0, 2, 1, 0, 3, 0, 1, 2, 0, 1])
        >>> idxs2
            tensor([1, 4, 2, 4, 4, 3, 4, 3, 2, 3])
        """
        idxs1, idxs2 = torch.tensor([i1 for i1, _ in dedup_ids]), torch.tensor([
            i2 for _, i2 in dedup_ids])
        return idxs1, idxs2
