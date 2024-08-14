# https://zephyrus1111.tistory.com/417
from sklearn.cluster import KMeans
from intra_class_variance.variance import Variance
from sklearn.metrics import davies_bouldin_score
from loguru import logger
import torch


class DaviesBouldinIndex(Variance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_name = "DBI"
        self.num_samples = 10

    def _calculate_variance(self, source, target):
        features = torch.concat([source.cpu(), target.cpu()]).numpy()
        # do kmeans
        kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
        cluster = kmeans.predict(features)
        # 100 번 sample 한게 아니므로, std 는 0
        return 1/davies_bouldin_score(features, cluster)

    # def preprocessing(self, features):
    #    logger.info(f"Calculating {self.metric_name}")
    #    mean, std = self.calculate_variance(features)
    #    return mean, std
